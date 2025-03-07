import torch
import torch.nn as nn
import torch.nn.functional as F

from toolbox.depth_adapted_sampling import computeOffset
from model_builder import BaseModel, PerImageNormalization, ModelTypeEnum, PaddingLayer, NormalizationEnum
from tvdcn import deform_conv2d


class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            if type(input) == tuple:
                input = module(*input)
            else:
                input = module(input)
        return input


class OffsetsConv2d(nn.Module):
    """
    This class is used to learn the offsets for the deformable convolutional layers.
    Its structure follows the DeformableDepthModel's layers that process RGB data, to obtain the correct output shape.
    """

    def __init__(self, height, width, kernel_size, padding, deformable_groups=1):
        super(OffsetsConv2d, self).__init__()
        self.padded = padding > 0
        self.offset_channels = 2 * kernel_size[0] * kernel_size[1] * deformable_groups

        # Prepare padding layers
        self.padding_1 = PaddingLayer([height, width], kernel_size=5, stride=2)
        self.padding_2 = PaddingLayer([height // 2, width // 2], kernel_size=5, stride=2)
        self.padding_3 = PaddingLayer([height // 4, width // 4], kernel_size=3, stride=2)

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=2),
            nn.ReLU(),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(36, 48, kernel_size=(3, 3), padding=padding, stride=(2, 2)),
            nn.ReLU(),
        )

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=(5, 5), padding=padding, stride=(1, 1)),
            nn.ReLU(),
        )

        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(64, self.offset_channels, kernel_size=(5, 5), padding=padding, stride=(1, 1)),
            nn.ReLU(),
        )

    def forward(self, depth):
        if self.padded:
            depth = self.padding_1(depth)

        depth = self.conv_block_1(depth)

        if self.padded:
            depth = self.padding_2(depth)

        depth = self.conv_block_2(depth)

        if self.padded:
            depth = self.padding_3(depth)

        depth = self.conv_block_3(depth)
        depth = self.conv_block_4(depth)
        depth = self.conv_block_5(depth)

        return depth


class DeformByDepthConv2d(nn.Module):
    """
    Wrapper for tvdcn.deform_conv2d function, to use it as a normal layer in PyTorch.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, deformable_groups=1):
        super(DeformByDepthConv2d, self).__init__()
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        nn.init.xavier_uniform_(self.weight)  # Xavier initialization, similar to the other models used in the project

        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, rgb, offsets):
        return deform_conv2d(
            input=rgb,
            weight=self.weight,
            offset=offsets,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=(1, 1),
        )


class DeformableDepthModel(BaseModel):
    """
    Variant of BaseModel that uses deformable convolutions to process RGB data.
    Configurable based on the mode parameter:
    - DCN: Offsets are learned from the depth data, using convolutional layers.
    - ZACN: Offsets are computed based on https://openaccess.thecvf.com/content/ACCV2020/papers/Wu_Depth-Adapted_CNN_for_RGB-D_cameras_ACCV_2020_paper.pdf

    Layer structure: Convolutional blocks -> Deformable Convolutional block -> Linear layer -> RNN layer
    """

    def __init__(self, mode, input_shape, model_type, padded, normalized, technique=NormalizationEnum.NONE):
        batch_size, seq_len, channels, height, width = input_shape
        super().__init__(input_shape, model_type, input_size=32, padded=padded, normalized=normalized,
                         technique=technique)
        self.mode = mode
        self.build_dcn_head(channels - 1, height, width)

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        # Merge time and batch dimension into a single one (because the Conv layers require this)
        x = x.view(batch_size * seq_len, *x.shape[2:])  # torch.Size([320, 4, 120, 212])
        if self.normalized:
            x = self.normalization(x)

        # Separate RGB and depth data
        depth = x[:, 3, :, :]  # torch.Size([320, 120, 212])
        rgb = x[:, :3, :, :]  # torch.Size([320, 3, 120, 212])

        # Generate offsets once for all deformable layers
        if self.mode == "DCN":
            depth = depth.view(batch_size * seq_len, 1, height, width)  # torch.Size([320, 1, 120, 212])
            offsets = self.offset_generator(depth)
        else:
            # We need to downscale the depth data to match the input coming from conv_block_5
            depth = F.interpolate(depth.unsqueeze(1), (12, 24) if self.padded else (5, 16), mode='bilinear',
                                  align_corners=True).squeeze(1)
            offsets = computeOffset(depth, 3, 1, chunk_size=32)
            offsets = F.pad(offsets, (1, 1, 1, 1), "constant", 0)

        # Apply convolutional blocks
        if self.padded:
            rgb = self.padding_1(rgb)

        rgb = self.conv_block_1(rgb)

        if self.padded:
            rgb = self.padding_2(rgb)

        rgb = self.conv_block_2(rgb)

        if self.padded:
            rgb = self.padding_3(rgb)

        rgb = self.conv_block_3(rgb)
        rgb = self.conv_block_4(rgb)
        rgb = self.conv_block_5(rgb)
        rgb = self.conv_block_6(rgb, offsets)

        lin = self.linear(rgb)

        # Separate time and batch dimension again
        lin = lin.view(batch_size, seq_len, *lin.shape[1:])  # shape is [batch_size, seq_len, 32]
        states, h_states = self.rnn_layer(lin)  # hx is the hidden state of the RNN
        if self.model_type == ModelTypeEnum.LSTM:
            return self.fc(states)

        return states

    def build_dcn_head(self, channels, height, width):
        if self.normalized:
            # Prepare normalization layer
            self.normalization = PerImageNormalization(self.technique)

        # Shared Offset Generator
        if self.mode == "DCN":
            self.offset_generator = OffsetsConv2d(height, width, kernel_size=(3, 3), padding=1 if self.padded else 0)

        # Prepare padding layers
        self.padding_1 = PaddingLayer([height, width], kernel_size=5, stride=2)
        self.padding_2 = PaddingLayer([height // 2, width // 2], kernel_size=5, stride=2)
        self.padding_3 = PaddingLayer([height // 4, width // 4], kernel_size=3, stride=2)

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(channels, 24, kernel_size=5, stride=2, padding=1 if self.padded else 0),
            nn.ReLU(),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=1 if self.padded else 0),
            nn.ReLU(),
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(36, 48, kernel_size=(3, 3), padding=(1, 1) if self.padded else 0, stride=(2, 2)),
            nn.ReLU(),
        )

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=(5, 5), padding=(1, 1) if self.padded else 0, stride=(1, 1)),
            nn.ReLU(),
        )

        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(5, 5), padding=(1, 1) if self.padded else 0, stride=(1, 1)),
            nn.ReLU(),
        )

        self.conv_block_6 = mySequential(
            DeformByDepthConv2d(64, 8, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),  # padding=(1,1) regardless
            nn.ReLU(),
        )

        if self.padded:
            self.linear = nn.Sequential(
                nn.Flatten(),
                nn.Linear(8 * 12 * 24, 32),
                nn.ReLU(),
            )
        else:
            self.linear = nn.Sequential(
                nn.Flatten(),
                nn.Linear(8 * 5 * 16, 32),
                nn.ReLU(),
            )

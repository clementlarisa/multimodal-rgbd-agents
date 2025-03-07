import torch
import torch.nn as nn
from ncps.torch import CfC, LTC
from ncps.wirings import NCP
from enum import Enum
import math
from lrc import LRC


class ModelTypeEnum(str, Enum):
    CfC = 'CfC'
    LTC = 'LTC'
    LSTM = 'LSTM'
    LRC = 'LRC'


class ModeEnum(str, Enum):
    VANILLA = 'VANILLA'
    TWO_STREAM = 'TWO_STREAM'
    ZACN = 'ZACN'
    DCN = 'DCN'


class NormalizationEnum(str, Enum):
    NONE = 'NONE'
    MINMAX = 'MINMAX'
    ZSCORE = 'ZSCORE'
    BOTH = 'BOTH'


class PerImageNormalization(nn.Module):
    """
    From a batch of image, this layer normalizes each image independently, on each channel. If an image is RGB-D, depth is normalized separately from RGB.
    Use one of the following techniques: NormalizationEnum.MINMAX, NormalizationEnum.ZSCORE, NormalizationEnum.BOTH
    Where BOTH first normalizes the image to [0, 1], then computes z-score.
    """

    def __init__(self, technique):
        super(PerImageNormalization, self).__init__()
        if technique == NormalizationEnum.NONE or technique not in NormalizationEnum:
            raise ValueError(f"Invalid normalization technique: {technique}")

        self.norm_fn = {
            NormalizationEnum.MINMAX: self.minmax_normalization,
            NormalizationEnum.ZSCORE: self.zscore_normalization,
            NormalizationEnum.BOTH: self.chained_normalization
        }[technique]

    def forward(self, inputs):
        # Convert inputs to float32 and separate RGB and depth components
        inputs = inputs.float()
        rgb_inputs = inputs[:, :3, :, :]

        # Apply per-image normalization to the RGB channels
        normalized_rgb = self.norm_fn(rgb_inputs)

        if inputs.shape[1] <= 3:
            # also captures Late Fusion case where depth is the only channel
            return normalized_rgb
        else:
            depth = inputs[:, 3:, :, :]
            normalized_depth = self.norm_fn(depth)
            return torch.cat([normalized_rgb, normalized_depth], dim=1)

    def minmax_normalization(self, img):
        # compute minmax normalization
        # if we use dim=(1, 2, 3) we get the min/max for each image in the batch, otherwise with dim=(0,1,2,3) we get the min/max for the whole batch
        min = torch.amin(img, dim=(1, 2, 3), keepdim=True)  # torch.Size([320, 1, 1, 1])
        max = torch.amax(img, dim=(1, 2, 3), keepdim=True)  # torch.Size([320, 1, 1, 1])
        img = (img - min) / (max - min)
        return img

    def zscore_normalization(self, img):
        # Compute mean and std for each image in the batch, across C x H x W dimensions
        # dim - 0 is batch size, 1 is channel, 2 is height, 3 is width
        mean = img.mean(dim=(1, 2, 3), keepdim=True)  # torch.Size([320, 1, 1, 1])
        stddev = img.std(dim=(1, 2, 3), keepdim=True)  # torch.Size([320, 1, 1, 1])
        min_stddev = 1.0 / torch.sqrt(torch.tensor(img[0].numel(), dtype=torch.float32, device=img.device))
        adjusted_stddev = torch.maximum(stddev, min_stddev)

        # Standardize
        standardized_img = (img - mean) / adjusted_stddev
        return standardized_img

    def chained_normalization(self, img):
        return self.zscore_normalization(self.minmax_normalization(img))


class PaddingLayer(nn.Module):
    """
    Layer to be used before a Conv2d layer when stride > 1.
    """

    def __init__(self, input_shape, kernel_size, stride):
        super(PaddingLayer, self).__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride

        # Calculate padding needed for this layer
        padding_needed = self._get_padding_needed(input_shape, [kernel_size, kernel_size], [stride, stride])
        padding = self._calculate_padding(padding_needed)
        # Pads the input tensor using zeros
        self.pad_layer = nn.ZeroPad2d((padding[1][0], padding[1][1], padding[0][0], padding[0][1]))

    def forward(self, x):
        return self.pad_layer(x)

    def _get_padding_needed(self, input_spatial_shape, filter_shape, strides):
        """
        Replicates the padding behavior of TensorFlow's 'same' padding https://www.tensorflow.org/api_docs/python/tf/nn/convolution

        :param input_spatial_shape:
        :param filter_shape:
        :param strides:
        :return:
        """
        num_spatial_dim = len(input_spatial_shape)
        padding_needed = [0] * num_spatial_dim
        # The total padding applied along the height and width is computed as:
        for i in range(num_spatial_dim):
            if input_spatial_shape[i] % strides[i] == 0:
                padding_needed[i] = max(filter_shape[i] - strides[i], 0)
            else:
                padding_needed[i] = max(filter_shape[i] - (input_spatial_shape[i] % strides[i]), 0)
        # Note that the division by 2 means that there might be cases when the padding on both sides (top vs bottom, right vs left)
        # are off by one. In this case, the bottom and right sides always get the one additional padded pixel.
        return padding_needed

    def _calculate_padding(self, padding_needed):
        padding = []
        # Finally, the padding on the top, bottom, left and right are:
        for pad in padding_needed:
            pad_before = pad // 2
            pad_after = pad - pad_before
            padding.append((pad_before, pad_after))
        return padding


def build_conv_head(height, width, channels, padded, normalized, technique=NormalizationEnum.NONE):
    """
    Builds the convolutional head of the model with the following structure:
    Normalization (if applicable)
    Padding (if applicable)
    6 x Conv2d -> ReLU
    1 x Flatten
    1 x Linear -> ReLU

    :param technique:
    :param height:
    :param width:
    :param channels:
    :param padded:
    :param normalized:
    :return:
    """
    layers = []

    if normalized:
        layers.append(PerImageNormalization(technique))

    if padded:
        layers.append(PaddingLayer([height, width], kernel_size=5, stride=2))
    layers.append(
        nn.Conv2d(channels, 24, kernel_size=5, stride=2))  # should be 24x60x106 (is 24x58x104 without padding)
    layers.append(nn.ReLU())
    if padded:
        layers.append(PaddingLayer([height // 2, width // 2], kernel_size=5, stride=2))
    layers.append(nn.Conv2d(24, 36, kernel_size=5, stride=2))  # should be 36x30x53 (is 36x27x50 without padding)
    layers.append(nn.ReLU())
    if padded:
        layers.append(PaddingLayer([height // 4, width // 4], kernel_size=3, stride=2))
    layers.append(nn.Conv2d(36, 48, kernel_size=3, stride=2))  # should be 48x15x27 (is 48x13x24 without padding)
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(48, 64, kernel_size=5, padding='same' if padded else 0))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(64, 64, kernel_size=5, padding='same' if padded else 0))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(64, 8, kernel_size=3, padding='same' if padded else 0))
    layers.append(nn.ReLU())
    layers.append(nn.Flatten())
    if padded:
        # thanks to the padding layer the height and width are halved after each convolution, so we can just divide by 8
        # where 8 = 2^3 is the down sampling due to strides from convolutions 1-3
        layers.append(nn.Linear(8 * math.ceil(height / 8) * math.ceil(width / 8), 32))
    else:
        layers.append(nn.Linear(8 * 3 * 14, 32))
    layers.append(nn.ReLU())

    return nn.Sequential(*layers)


class BaseModel(nn.Module):
    """
    Base class for all the models in this project. Contains the common wiring for the RNN layer, and the FC layer.
    Initializes the Conv2d weights of the model using Xavier Uniform initialization, and bias to 0.01.
    """

    def __init__(self, input_shape, model_type, input_size, padded, normalized, technique=NormalizationEnum.NONE):
        super().__init__()
        self.normalized = normalized
        self.padded = padded
        self.model_type = model_type
        self.technique = technique
        self.rnn_layer = self._get_rnn_layer(model_type, input_size)
        self.fc = nn.Linear(64, 1)
        self._initialize_weights()

    @staticmethod
    def _get_rnn_layer(model_type, input_size):
        # Wiring for CfC or LTC models
        paper_wiring = NCP(
            inter_neurons=12, command_neurons=6, motor_neurons=1,
            sensory_fanout=6, inter_fanout=4, recurrent_command_synapses=6, motor_fanin=6, seed=20240109
        )
        # RNN layer based on the selected model type
        if model_type == ModelTypeEnum.CfC:
            return CfC(input_size=input_size, units=paper_wiring, return_sequences=True)
        elif model_type == ModelTypeEnum.LTC:
            return LTC(input_size=input_size, units=paper_wiring, return_sequences=True)
        elif model_type == ModelTypeEnum.LSTM:
            return nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True)
        elif model_type == ModelTypeEnum.LRC:
            return LRC(input_size=input_size, units=64)

    def _initialize_weights(self):
        # Iterate through the model layers and apply initializations
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # Apply Xavier (Glorot) Uniform initialization for weights
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # Set bias to a constant value of 0.01
                    nn.init.constant_(m.bias, 0.01)


class EarlyFusionModel(BaseModel):
    """
    Model that fuses RGB and Depth data early in the network.
    """

    def __init__(self, input_shape, model_type, padded, normalized, technique=NormalizationEnum.NONE):
        batch_size, seq_len, channels, height, width = input_shape
        super().__init__(input_shape, model_type, input_size=32, padded=padded, normalized=normalized,
                         technique=technique)
        self.conv_head = build_conv_head(height, width, channels, padded, normalized, technique)

    def forward(self, x):
        batch_size, seq_len, _, height, width = x.size()
        # Merge time and batch dimension into a single one (because the Conv layers require this)
        x = x.view(batch_size * seq_len, *x.shape[2:])
        x = self.conv_head(x)
        # Separate time and batch dimension again
        x = x.view(batch_size, seq_len, *x.shape[1:])  # shape is [batch_size, seq_len, 32]
        x, _ = self.rnn_layer(x)  # hx is the hidden state of the RNN
        return self.fc(x) if self.model_type == ModelTypeEnum.LSTM else x


class LateFusionModel(BaseModel):
    """
    Model that fuses RGB and Depth data late in the network, using a concatenation layer.
    """

    def __init__(self, input_shape, model_type, padded, normalized, technique=NormalizationEnum.NONE):
        batch_size, _, _, height, width = input_shape
        super().__init__(input_shape, model_type, input_size=64, padded=padded, normalized=normalized,
                         technique=technique)
        # Separate RGB (3 channels) and Depth (1 channel) streams
        self.rgb_conv_head = build_conv_head(height, width, 3, padded, normalized, technique)
        self.depth_conv_head = build_conv_head(height, width, 1, padded, normalized, technique)

    def forward(self, x):
        batch_size, seq_len, _, height, width = x.size()
        # Split input into RGB and Depth
        # First 3 channels
        # Merge batch and sequence dimensions for convolutional layers
        rgb_x = x[:, :, :3, :, :].view(batch_size * seq_len, 3, height, width)
        # Last channel
        depth_x = x[:, :, 3:, :, :].view(batch_size * seq_len, 1, height, width)
        # Apply separate conv heads
        rgb_x, depth_x = self.rgb_conv_head(rgb_x), self.depth_conv_head(depth_x)
        # Concatenate RGB and Depth features, Reshape back for RNN input
        combined_x = torch.cat((rgb_x, depth_x), dim=-1).view(batch_size, seq_len, -1)
        # Pass through the RNN layer
        x, _ = self.rnn_layer(combined_x)
        return self.fc(x) if self.model_type == ModelTypeEnum.LSTM else x


class RGBModel(EarlyFusionModel):
    def __init__(self, input_shape, model_type, padded, normalized, technique=NormalizationEnum.NONE):
        super().__init__(input_shape, model_type, padded, normalized, technique)

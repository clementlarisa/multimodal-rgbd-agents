import argparse
import datetime
import math
import os
import random
import time
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset_loader import TarNPZLoader, ImageSize, CustomDataset, collate_fn
from deformable_model_builder import DeformableDepthModel
from model_builder import ModelTypeEnum, EarlyFusionModel, LateFusionModel, RGBModel, NormalizationEnum
from tqdm import tqdm


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_step(model, train_loader, loss_fn, optimizer, device):
    # Put model in train mode
    model.train()
    # Setup train loss
    running_loss = 0.0
    # Loop through data loader data batches
    progress_bar = tqdm(train_loader, unit="batch", desc="Training", file=sys.stdout)  # Progress bar for training
    for images, labels in progress_bar:
        # Send data to target device
        images, labels = images.to(device), labels.to(device)
        # with torch.autocast(device_type="cuda"):
        # 1. Forward pass
        outputs = model(images).squeeze(-1)
        # 2. Calculate  and accumulate loss
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()

        # 3. Optimizer zero grad clears old gradients from the last step
        optimizer.zero_grad()

        # 4. Loss backward computes the derivative of the loss w.r.t. the parameters
        loss.backward()

        # 5. Optimizer step causes the optimizer to take a step based on the gradients of the parameters.
        optimizer.step()

        progress_bar.set_postfix(loss=loss.item())

    return running_loss / len(train_loader)


def validation_step(model, dev_loader, loss_fn, device):
    model.eval()
    val_loss = 0.
    progress_bar = tqdm(dev_loader, unit="batch", desc="Validation", file=sys.stdout)  # Progress bar for validation
    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            # with torch.autocast(device_type="cuda"):
            outputs = model(images).squeeze(-1)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())

    return val_loss / len(dev_loader)


def test_step(model, test_loader, loss_fn, device, summary_writer):
    model.eval()
    test_loss = 0.0
    test_progress_bar = tqdm(test_loader, unit="batch", desc="Test", file=sys.stdout)
    with torch.no_grad():
        for images, labels in test_progress_bar:
            images, labels = images.to(device), labels.to(device)
            # with torch.autocast(device_type="cuda"):
            outputs = model(images).squeeze(-1)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()

            test_progress_bar.set_postfix(loss=loss.item())

    # Add to tensorboard
    summary_writer.add_scalar('Loss/test', test_loss / len(test_loader), 1)
    print(f"Test loss: {test_loss / len(test_loader)}")


def train(mode, epochs, early_stop_patience, train_loader, dev_loader, model, loss_fn, optimizer, device,
          summary_writer):
    best_val_loss = float('inf')
    early_stop_counter = 0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        # Train step
        avg_train_loss = train_step(model, train_loader, loss_fn, optimizer, device)

        # Validation step
        avg_val_loss = validation_step(model, dev_loader, loss_fn, device)

        # Add to tensorboard
        summary_writer.add_scalar('Loss/train', avg_train_loss, epoch)
        summary_writer.add_scalar('Loss/validation', avg_val_loss, epoch)

        print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.8f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            print(f"Validation loss decreased from {best_val_loss:.8f} to {avg_val_loss:.8f}")
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model, model_path)  # Save the best model
            print(f"Model saved at {model_path}")
        else:
            early_stop_counter += 1
            print("Increased early stop counter to ", early_stop_counter)
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break


def load_dataset(recordings_dir, seed, test_size, dev_size,
                 image_size, depth, sequence_length, batch_size):
    loader = TarNPZLoader(recordings_dir=recordings_dir, seed=seed, test_size=test_size, dev_size=dev_size,
                          image_size=image_size,
                          depth=depth)

    train_data, dev_data, test_data = loader.load_datasets()

    train_dataset = CustomDataset(dataset=train_data, sequence_length=sequence_length)
    dev_dataset = CustomDataset(dataset=dev_data, sequence_length=sequence_length)
    test_dataset = CustomDataset(dataset=test_data, sequence_length=sequence_length)

    steps_per_epoch = math.ceil(len(train_dataset) // batch_size)
    validation_steps = math.ceil(len(dev_dataset) // batch_size)
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                               collate_fn=collate_fn, num_workers=1, pin_memory=True)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                                             num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                                              num_workers=1, pin_memory=True)

    return train_loader, dev_loader, test_loader


def build_parser():
    parser.add_argument('--seed', type=int, default=20250703, help='The length of the sequence to use.')
    parser.add_argument('--model_type', type=str, default='CFC', help='Which model to use: CFC, LSTM, LTC or LRC.')
    parser.add_argument('--mode', type=str, default='EARLY',
                        help='Which depth algorithm to use: RGB, EARLY, LATE, ZACN or DCN.')
    parser.add_argument('--depth', action='store_true', default=False,
                        help='If True, loads RGB-D images in the dataset. Otherwise, loads only RGB images.')
    parser.add_argument('--normalized', action='store_true', default=False,
                        help='If True, normalizes each image in the dataset.')
    parser.add_argument('--padded', action='store_true', default=False,
                        help='If True, uses padding in between convolutional layers.')
    parser.add_argument('--load', action='store_true',
                        help='If True, tries to load a saved model with these parameters.')
    parser.add_argument('--sequence_length', type=int, default=16, help='The length of the sequence to use.')
    parser.add_argument('--image_scale', type=str, default='QUARTER',
                        help='The scale of the images to load: FULL, HALF, or QUARTER.')
    parser.add_argument('--recordings_dir', type=str, default='datasets',
                        help='The directory containing the tar files.')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs to train the model.')
    parser.add_argument('--batch_size', type=int, default=20, help='The batch size to use.')
    parser.add_argument('--test_size', type=float, default=0.2, help='The test size to use.')
    parser.add_argument('--dev_size', type=float, default=0.2, help='The validation size to use.')
    parser.add_argument('--lr', type=float, default=0.0005, help='The learning rate of the Adam optimizer.')
    parser.add_argument('--early_stop_patience', type=int, default=3, help='The patience for early stopping.')
    parser.add_argument('--normalization_technique', type=str, default='NONE',
                        help='The normalization technique to use: MINMAX for [0, 1] normalization, ZSCORE for z-score normalization, or BOTH for MINMAX followed by ZSCORE.')


if __name__ == "__main__":
    # Device configuration
    cuda_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------------------- Parse command line arguments ------------------------
    parser = argparse.ArgumentParser(description='Train and test model')
    build_parser()
    args = parser.parse_args()
    args_dict = vars(args)
    print(f"Arguments: {args_dict}")

    # ---------------------------- Seed ------------------------------------------
    seed_everything(args.seed)

    if args.mode not in ["EARLY", "LATE", "RGB", "ZACN", "DCN"]:
        raise ValueError("Invalid mode. Choose from EARLY, LATE, RGB, ZACN, DCN")

    image_scale = {
        "FULL": ImageSize.FULL,
        "HALF": ImageSize.HALF,
        "QUARTER": ImageSize.QUARTER,
    }[args.image_scale]

    normalization_technique = {
        "MINMAX": NormalizationEnum.MINMAX,
        "ZSCORE": NormalizationEnum.ZSCORE,
        "BOTH": NormalizationEnum.BOTH,
        "NONE": NormalizationEnum.NONE,
    }[args.normalization_technique]

    # ---------------------------- Tick ------------------------------------------
    start_time = time.time()

    # --------------------- Callbacks for training ------------------------------
    now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = (
                     f"{normalization_technique}_" if args.normalized else "") + f"{args_dict['model_type']}_{args_dict['mode']}_{args_dict['lr']}lr_padded{args_dict['padded']}_normalized{args_dict['normalized']}.pt"
    fit_log_dir = os.path.join(f"outputs_{args.seed}/logs/fit", model_name, now_str)
    writer = SummaryWriter(log_dir=fit_log_dir)
    eval_log_dir = os.path.join(f"outputs_{args.seed}/logs/evaluate", model_name, now_str)
    test_writer = SummaryWriter(log_dir=eval_log_dir)

    # ----------------------- Load and prepare the dataset ---------------------
    train_ds_loader, dev_ds_loader, test_ds_loader = load_dataset(recordings_dir=args.recordings_dir,
                                                                  seed=args.seed,
                                                                  test_size=args.test_size, dev_size=args.dev_size,
                                                                  image_size=image_scale, depth=args.depth,
                                                                  sequence_length=args.sequence_length,
                                                                  batch_size=args.batch_size)

    # CHW format
    input_shape = (args.batch_size,
                   args.sequence_length,
                   3 if not args.depth else 4,
                   image_scale[1],
                   image_scale[0],
                   )

    # --------------------- If using saved model --------------------------
    model_path = os.path.join(f"outputs_{args.seed}/models", model_name)
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        model_path = model_path.replace(".pt", now_str + ".pt")
    if args.load and os.path.exists(model_path):
        stacked_model = torch.load(model_path)
        print(f"Loaded model from {model_path}")
        test_step(stacked_model, test_ds_loader, nn.MSELoss(), cuda_device, test_writer)
        exit()

    # --------------------- Create and compile the model ------------------------
    model_type = {
        "CFC": lambda: ModelTypeEnum.CfC,
        "LSTM": lambda: ModelTypeEnum.LSTM,
        "LTC": lambda: ModelTypeEnum.LTC,
        "LRC": lambda: ModelTypeEnum.LRC,
    }[args.model_type]()

    stacked_model = {
        "EARLY": lambda: EarlyFusionModel(input_shape, model_type=model_type, padded=args.padded,
                                          normalized=args.normalized, technique=normalization_technique),
        "LATE": lambda: LateFusionModel(input_shape, model_type=model_type, padded=args.padded,
                                        normalized=args.normalized, technique=normalization_technique),
        "RGB": lambda: RGBModel(input_shape, model_type=model_type, padded=args.padded, normalized=args.normalized,
                                technique=normalization_technique),
        "ZACN": lambda: DeformableDepthModel(mode="ZACN", input_shape=input_shape, model_type=model_type,
                                             padded=args.padded, normalized=args.normalized,
                                             technique=normalization_technique),
        "DCN": lambda: DeformableDepthModel(mode="DCN", input_shape=input_shape, model_type=model_type,
                                            padded=args.padded, normalized=args.normalized,
                                            technique=normalization_technique),
    }[args.mode]()

    stacked_model = stacked_model.to(cuda_device)
    mse_loss_fn = nn.MSELoss()
    adam_opt = optim.Adam(stacked_model.parameters(), lr=args.lr, eps=1e-7)  # use same eps as in Keras

    # --------------------- Training and validation -----------------------------
    train(mode=args.mode, epochs=args.epochs, early_stop_patience=args.early_stop_patience,
          train_loader=train_ds_loader,
          dev_loader=dev_ds_loader, model=stacked_model, loss_fn=mse_loss_fn, optimizer=adam_opt, device=cuda_device,
          summary_writer=writer)

    # --------------------- Test the model ---------------------------------------
    test_step(stacked_model, test_ds_loader, mse_loss_fn, cuda_device, test_writer)

    # -------------------------------- Tock ---------------------------------------
    elapsed_time = time.time() - start_time
    print(f"Total time: {elapsed_time:.2f} seconds")

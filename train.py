import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from loader import FeatureLoader
from model import AttentionMIL
from utils import phase_step


def phase_step(model, dataloader, optimizer, criterion, phase):

    if phase == 'train':
        train=True
        model.train()

    if phase == 'valid':
        train=False
        model.eval()

    with torch.set_grad_enabled(train):

        phase_loss, phase_metr = 0.0, 0.0

        for data in dataloader:

            X = data['X'].to(device)
            Y = data['Y'].to(device)

            # Forward pass
            optimizer.zero_grad()
            P, A = model(X)
            loss = criterion(P, Y.long())

            # Backward Pass
            if train:
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                P = torch.argmax(P, dim=-1)
                metr = P.eq(Y).sum()

            phase_loss += loss.item()
            phase_metr += metr.item()

        phase_loss = phase_loss/len(dataloader)
        phase_metr = phase_metr/len(dataloader)

    return phase_loss, phase_metr


def main(config):

    print(f'\nFold-{config["fold"]} ...', flush=True)

    # Arange files and labels
    files = sorted(glob.glob(f'{config["data_dir"]}/*.h5'))
    labels = np.array([int(f.split('/')[-1][5]) for f in files]) - 1

    # K-Fold Split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    train_indices, valid_indices = list(skf.split(files, labels))[config["fold"]]
    train_samples = [{"X": files[i], "Y":labels[i]} for i in train_indices]
    valid_samples = [{"X": files[i], "Y":labels[i]} for i in valid_indices]
    np.random.shuffle(train_samples)
    print(f'\nNumber of train files: {len(train_samples)}', flush=True)
    print(f'Number of valid_files: {len(valid_samples)}', flush=True)

    # Create dataset
    train_ds = FeatureLoader(train_samples)
    valid_ds = FeatureLoader(valid_samples)
    train_ds = DataLoader(train_ds, batch_size=None, shuffle=True, num_workers=config["workers"], pin_memory=True)
    valid_ds = DataLoader(valid_ds, batch_size=None, shuffle=False, num_workers=config["workers"], pin_memory=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'\nDevice: {device}')

    # Create model
    model = AttentionMIL(input_shape=1024, classes=3).to(device)
    print(f'\nModel compiled', flush=True)

    # Loss, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    checkpoint_path = f'{config["save_dir"]}/model_f{config["fold"]}.pt'

    # Training
    print(f'\nTraining ...\n')

    monitor_metr = 0
    for epoch in range(0, config["epochs"]):

        print(f'\nEpoch {epoch:03}/{config["epochs"]}')

        for phase in ['train', 'valid']:
            dataloader = train_ds if phase == 'train' else valid_ds
            phase_loss, phase_metr = phase_step(phase)
            print(f'{phase}_loss: {phase_loss:0.4f} - {phase}_accuracy: {phase_metr:0.4f}')

        scheduler.step(phase_loss)

        if phase_metr>monitor_metr:
            torch.save(model, checkpoint_path)
            monitor_metr = phase_metr
            print(f'checkpoint saved: {checkpoint_path}')

    print('\nComplete!')



if __name__ == "__main__":

    config = {
        "data_dir": '/storage/features',
        "save_dir": '/storage/models',
        "epochs": 50,
        "fold": 1,
        "seed": 0,
        "workers": 2,
    }

    main(config)

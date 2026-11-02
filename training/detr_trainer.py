"""GalaxyDETR optimizer setup and train loop (best-validation-loss checkpointing, as in the VGG notebook)."""
import time

import torch

from training.detr_validater import validation_loss
from utils.device import targets_to_device


def build_optimizer(model, cfg):
    """AdamW with a lower LR on the (pretrained) backbone so its ImageNet features aren't wrecked early."""
    t = cfg['training']
    return torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if n.startswith('backbone')], 'lr': t['lr_backbone']},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('backbone')], 'lr': t['lr']},
    ], weight_decay=t['weight_decay'])


def train_model(model, criterion, optimizer, train_loader, val_loader, device,
                num_epochs, checkpoint_path, grad_clip=0.1):
    """Train, keep the checkpoint with the lowest validation loss. Returns (train_losses, val_losses)."""
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    training_losses, validation_losses = [], []
    best_validation_loss, best_validation_epoch = float('inf'), -1
    training_start = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, num_batches = 0.0, 0
        for images, targets in train_loader:
            loss, _ = criterion(model(images.to(device)), targets_to_device(targets, device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        sched.step()
        avg_loss = epoch_loss / num_batches
        val_loss = validation_loss(model, criterion, val_loader, device)
        training_losses.append(avg_loss)
        validation_losses.append(val_loss)

        best_marker = ''
        if val_loss < best_validation_loss:
            best_validation_loss, best_validation_epoch = val_loss, epoch
            torch.save(model.state_dict(), checkpoint_path)
            best_marker = ' *'
        print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {avg_loss:.5f}, Val Loss: {val_loss:.5f}{best_marker}',
              flush=True)

    print(f'Average time per epoch: {(time.time() - training_start) / num_epochs:.2f}s')
    print(f'Best validation loss: {best_validation_loss:.5f} (epoch {best_validation_epoch})')
    return training_losses, validation_losses

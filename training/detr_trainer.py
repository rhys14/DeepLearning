"""GalaxyDETR optimizer setup and train loop (best-validation-loss checkpointing, as in the VGG notebook)."""
import json
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
                num_epochs, save_fn, grad_clip=0.1, log_every=50, history_path=None):
    """Train; call save_fn(model, epoch, val_loss) every time validation loss reaches a new best.
    Returns a history dict: train_loss, val_loss, epoch_time (lists), best_epoch, best_val_loss,
    avg_epoch_time, total_time.
    history_path: if given, the history so far is written there as JSON after every epoch, so it
    survives a crash or a failed report (make_report_detr.py can rebuild the report from it)."""
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    training_losses, validation_losses, epoch_times = [], [], []
    best_validation_loss, best_validation_epoch = float('inf'), -1
    training_start = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, num_batches = 0.0, 0
        epoch_start = time.time()
        for images, targets in train_loader:
            loss, logs = criterion(model(images.to(device)), targets_to_device(targets, device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
            if log_every and num_batches % log_every == 0:
                elapsed = time.time() - epoch_start
                print(f'  Epoch [{epoch}/{num_epochs}] Batch [{num_batches}/{len(train_loader)}] '
                      f'Loss: {epoch_loss / num_batches:.4f} (ce {logs["ce"]:.3f} l1 {logs["bbox"]:.3f} '
                      f'giou {logs["giou"]:.3f})  {elapsed / num_batches:.2f}s/batch', flush=True)
        sched.step()
        avg_loss = epoch_loss / num_batches
        val_loss = validation_loss(model, criterion, val_loader, device)
        training_losses.append(avg_loss)
        validation_losses.append(val_loss)

        best_marker = ''
        if val_loss < best_validation_loss:
            best_validation_loss, best_validation_epoch = val_loss, epoch
            save_fn(model, epoch, val_loss)
            best_marker = ' *'
        epoch_times.append(time.time() - epoch_start)
        print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {avg_loss:.5f}, Val Loss: {val_loss:.5f}{best_marker} '
              f'({epoch_times[-1]:.0f}s)', flush=True)
        if history_path:
            elapsed = time.time() - training_start
            with open(history_path, 'w') as f:
                json.dump({'train_loss': training_losses, 'val_loss': validation_losses, 'epoch_time': epoch_times,
                           'best_epoch': best_validation_epoch, 'best_val_loss': best_validation_loss,
                           'avg_epoch_time': elapsed / (epoch + 1), 'total_time': elapsed}, f, indent=1)

    total_time = time.time() - training_start
    print(f'Average time per epoch: {total_time / num_epochs:.2f}s')
    print(f'Best validation loss: {best_validation_loss:.5f} (epoch {best_validation_epoch})')
    return {'train_loss': training_losses, 'val_loss': validation_losses, 'epoch_time': epoch_times,
            'best_epoch': best_validation_epoch, 'best_val_loss': best_validation_loss,
            'avg_epoch_time': total_time / num_epochs, 'total_time': total_time}

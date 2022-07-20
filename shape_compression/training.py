'''Implements a generic training loop (based on https://github.com/vsitzmann/siren).'''
import copy
import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

import utils
from losses import model_l1
from utils import ReduceLROnPlateauWithWarmup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn, val_dataloader=None, double_precision=False, loss_schedules=None, weight_decay=0, l1_reg=0,
          l1_loss_fn=model_l1, use_amp=True, ref_model=None, patience=500, warmup=0, early_stop_epochs=5000):
    optim = torch.optim.Adam(lr=lr, params=model.parameters(), weight_decay=weight_decay)

    if warmup > 0:
        scheduler = ReduceLROnPlateauWithWarmup(optim, warmup_end_lr=lr, warmup_steps=warmup, mode='min', factor=0.5,
                                                patience=patience,
                                                threshold=0.0001,
                                                threshold_mode='rel', cooldown=0, eps=1e-08, verbose=True)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=patience,
                                                               threshold=0.0001,
                                                               threshold_mode='rel', cooldown=0, eps=1e-08,
                                                               verbose=True)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)
    writer = SummaryWriter(summaries_dir)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    total_steps = 0

    best_state_dict = copy.deepcopy(model.state_dict())

    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        epochs_since_improvement = 0
        best_total_epoch_loss = float("Inf")
        for epoch in range(epochs):
            total_epoch_loss = 0
            if epochs_since_improvement > early_stop_epochs:
                break  # stop early if no improvement since early_stop_epochs epochs

            if not epoch % epochs_til_checkpoint and epoch:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                with torch.cuda.amp.autocast(enabled=use_amp):
                    start_time = time.time()

                    model_input = {key: value.cuda() for key, value in model_input.items()}
                    gt = {key: value.cuda() for key, value in gt.items()}

                    model_output = model(model_input)
                    losses = loss_fn(model_output, gt)
                    if l1_reg > 0:
                        l1_loss = l1_loss_fn(model, l1_reg)
                        losses = {**losses, **l1_loss}

                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()

                        if loss_schedules is not None and loss_name in loss_schedules:
                            writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps),
                                              total_steps)
                            single_loss *= loss_schedules[loss_name](total_steps)

                        writer.add_scalar(loss_name, single_loss, total_steps)
                        train_loss += single_loss

                    train_losses.append(train_loss.item())
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                    if not total_steps % steps_til_summary:
                        torch.save(model.state_dict(),
                                   os.path.join(checkpoints_dir, 'model_current_.pth'))

                        summary_fn(model, model_input, gt, model_output, writer, total_steps)

                    if not total_steps % steps_til_summary:
                        tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f, best epoch loss %0.6f" % (
                            epoch, train_loss, time.time() - start_time, best_total_epoch_loss))

                        if val_dataloader is not None:
                            print("Running validation set...")
                            model.eval()
                            with torch.no_grad():
                                val_losses = []
                                for (model_input, gt) in val_dataloader:
                                    model_output = model(model_input)
                                    val_loss = loss_fn(model_output, gt)
                                    val_losses.append(val_loss)

                                writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                            model.train()
                    optim.zero_grad()
                    scaler.scale(train_loss).backward()
                    scaler.step(optim)
                    scaler.update()
                    pbar.update(1)
                    scheduler.step(train_loss)
                    total_steps += 1
                    total_epoch_loss += train_loss.item()

                total_steps += 1
                total_epoch_loss += train_loss.item()

            epochs_since_improvement += 1
            if total_epoch_loss < best_total_epoch_loss:
                best_total_epoch_loss = total_epoch_loss
                best_state_dict = copy.deepcopy(model.state_dict())
                best_adam_state_dict = copy.deepcopy(optim.state_dict())
                epochs_since_improvement = 0
            if total_epoch_loss > 10 * best_total_epoch_loss:
                model.load_state_dict(best_state_dict, strict=True)
                optim.load_state_dict(best_adam_state_dict)

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        torch.save(best_state_dict,
                   os.path.join(checkpoints_dir, 'model_best_.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))

        model.load_state_dict(best_state_dict, strict=True)

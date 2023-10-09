import json
import os
from typing import Any, Dict, Optional
from tqdm.autonotebook import tqdm

import numpy as np
import torch
import torch.utils.data.distributed
from torch import nn
from torch.utils.data import DataLoader

from PIL import Image
from tqdm.autonotebook import tqdm
import torchvision.transforms.functional as f

from model.idisc.utils.metrics import RunningMetric
from model.idisc.utils.misc import is_main_process
from model.idisc.utils.visulization import gray_out


def log_losses(losses_all):
    for loss_name, loss_val in losses_all.items():
        print(f"Test/{loss_name}: ", loss_val)


def update_best(metrics_all, metrics_best="abs_rel"):
    curr_loss = []
    for metrics_name, metrics_value in metrics_all.items():
        if metrics_best in metrics_name:
            curr_loss.append(metrics_value)

    curr_loss = np.mean(curr_loss)
    if curr_loss < validate.best_loss:
        validate.best_loss = curr_loss
        validate.best_metrics = metrics_all

    for metrics_name, metrics_value in metrics_all.items():
        try:
            print(
                f"{metrics_name} {round(validate.best_metrics[metrics_name], 4)} ({round(metrics_value, 4)})"
            )
        except:
            print(f"Error in best. {metrics_name} ({round(metrics_value, 4)})")


def save_model(
    metrics_all, state_dict, run_save_dir, step, config, metrics_best="abs_rel"
):
    curr_loss = []
    curr_dataset = config["data"]["train_dataset"]
    for metrics_name, metrics_value in metrics_all.items():
        if metrics_best in metrics_name:
            curr_loss.append(metrics_value)
    curr_loss = np.mean(curr_loss)

    if curr_loss == validate.best_loss:
        try:
            torch.save(
                state_dict, os.path.join(run_save_dir, f"{curr_dataset}-best.pt")
            )
            with open(
                os.path.join(run_save_dir, f"{curr_dataset}-config.json"), "w+"
            ) as fp:
                json.dump(config, fp)
        except OSError as e:
            print(f"Error while saving model: {e}")
        except:
            print("Generic error while saving")


def validate(
    model: nn.Module,
    test_loader: DataLoader,
    metrics_tracker: RunningMetric,
    context: torch.autocast,
    config: Dict[str, Any],
    run_id: Optional[str] = None,
    save_dir: Optional[str] = None,
    step: int = 0,
):
    ds_losses = {}
    device = model.device
    if save_dir is not None:
        run_save_dir = os.path.join(save_dir, run_id)
        os.makedirs(run_save_dir, exist_ok=True)

    progress_bar = tqdm(test_loader, colour="green")

    # Load over iteration
    for i, batch in enumerate(progress_bar):
        with context:
            gt, mask = batch["gt"].to(device), batch["mask"].to(device)
            preds, losses, _ = model(batch["image"].to(device), gt, mask)

        losses = {k: v for l in losses.values() for k, v in l.items()}
        for loss_name, loss_val in losses.items():
            ds_losses[loss_name] = (
                loss_val.detach().cpu().item() + i * ds_losses.get(loss_name, 0.0)
            ) / (i + 1)

        progress_bar.set_description(
            f"Iteration {step}: Loss {ds_losses}")

        metrics_tracker.accumulate_metrics(
            gt.permute(0, 2, 3, 1), preds.permute(0, 2, 3, 1), mask.permute(0, 2, 3, 1)
        )

    losses_all = ds_losses
    metrics_all = metrics_tracker.get_metrics()
    metrics_tracker.reset_metrics()

    if is_main_process():
        log_losses(losses_all=losses_all)
        update_best(metrics_all=metrics_all, metrics_best="abs_rel")
        if save_dir is not None:
            with open(os.path.join(run_save_dir, f"metrics_{step}.json"), "w") as f:
                json.dump({**losses_all, **metrics_all}, f)
            save_model(
                metrics_all=metrics_all,
                state_dict=model.state_dict(),
                config=config,
                metrics_best="abs_rel",
                run_save_dir=run_save_dir,
                step=step,
            )

def visual(
        model: nn.Module,
        test_loader: DataLoader,
        save_dir: str
):
    device = model.device

    os.makedirs(save_dir, exist_ok=True)

    # Use tqdm for visual training process
    progress_bar = tqdm(test_loader, colour="green")

    for i, batch in enumerate(progress_bar):
        gt, mask = batch["gt"].to(device), batch["mask"].to(device)
        preds, losses, _ = model(batch["image"].to(device), gt, mask)


        imgs = gray_out(preds)

        for img, base_path, img_name in zip(imgs, batch['base_path'], batch['img_name']):
            img = f.to_pil_image(img)
            os.makedirs(f'{save_dir}/{base_path}', exist_ok=True)
            img.save(f'{save_dir}/{base_path}/{img_name}')


def save(
        model: nn.Module,
        test_loader: DataLoader,
        save_dir: str
):
    device = model.device

    os.makedirs(save_dir, exist_ok=True)

    for i, batch in enumerate(test_loader):
        gt, mask = batch["gt"].to(device), batch["mask"].to(device)
        preds, losses, _ = model(batch["image"].to(device), gt, mask)

        imgs = colorize(preds)

        for img, file_name in zip(imgs, batch['file_name']):
            img = f.to_pil_image(img)
            os.makedirs(f'{save_dir}/{file_name[:-11]}', exist_ok=True)
            img.save(f'{save_dir}/{file_name}')


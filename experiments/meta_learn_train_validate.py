# Configs
import logging
import os
import pdb

# time management
# time management
import time
from pathlib import Path

import hydra

# Numpy
import numpy as np
import torch.cuda

# Loss
import torch.nn.functional as F

# Logging
# Logging
import wandb
from omegaconf import OmegaConf

# Dataset
# from ct_dataset import CTDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.cbct_recon_dataset import MultiVolumeConebeamReconDataset
from dataset.cbct_dataset import MultiVolumeConebeamDataset

# Nefs
from nef.modulation import get_modulated_neural_field
from nn_utils.checkpointer import MamlCheckpointMaker
from nn_utils.loggers import MultiCBCTReconLogger, MultiCBCTLogger
from nn_utils.early_stopping import EarlyStopping
from nn_utils.optimizers import get_optimizer

from rendering.rendering import ConebeamRenderer

from dataset import get_dataloader

# Meta learning
from nn_utils.meta_learning import _clone_module, _retain_grads_module, _set_requires_grad


# partial
from functools import partial
import subprocess as sp

log = logging.getLogger(__name__)


@hydra.main(
    config_path="../config", config_name="meta_learning.yaml", version_base=None
)
def main(cfg):
    experiment_path = Path(os.getcwd())
    log.info("Experiment path: {}".format(experiment_path))
    # We possibly want to add fields to the config file. Thus, we set struct to False.
    OmegaConf.set_struct(cfg, False)

    cfg_path = experiment_path / "hydra_config.yaml"
    if cfg_path.exists() and cfg.load_config:
        log.info("Found hydra config file. Loading it.")
        old_cfg = OmegaConf.load(cfg_path)
        # merge the two cfgs and ignore parameters for dataset and val_dataset
        new_cfg = OmegaConf.merge(cfg, old_cfg)
        new_cfg.dataset = cfg.dataset
        new_cfg.val_dataset = cfg.val_dataset
        new_cfg.load_weights = cfg.load_weights
        new_cfg.training.do = cfg.training.do
        new_cfg.load_weights = cfg.load_weights
        new_cfg.project_name = cfg.project_name
        new_cfg.validation.max_time = cfg.validation.max_time
        new_cfg.validation.lr = cfg.validation.lr

        noisy_name = "noisy" if cfg.val_dataset.noisy_projections else "clean"
        new_cfg.run_name = f"{cfg.run_name}_{cfg.val_dataset.stage}_{cfg.val_dataset.projs_sample_step}_{noisy_name}"
        cfg = new_cfg

        # print the configuration using logging
        log.info(cfg)

    else:
        if not cfg.load_config:
            log.info("load_config is set to False, so we overwrite the config file.")
        elif not cfg_path.exists():
            log.info(
                "No hydra config file found. Using provided config and storing it."
            )

        # store hydra config
        with open("hydra_config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

    # set all the seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    # Get device
    device = torch.device("cuda") if (cfg.cuda and torch.cuda.is_available()) else "cpu"
    log.info(f"Using device: {device}.")

    # Initialize logging.
    # wandb.login(key="da05829c15c052ce21ea676a2050405df8abf981")
    wandb.init(
        project=cfg.project_name,
        name=cfg.run_name,
        mode="online" if cfg.wandb_log else "offline",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    ############
    # Training #
    ############
    # Construct dataloaders
    log.info(f"Loading dataset {cfg.dataset.name}.")

    # Because we're using meta-learning, and we want to do multiple optimization steps per patient, we need to make
    # sure to multiply the batch size by number of inner loop steps.
    cfg.training.batch_size = cfg.meta_learning.inner_steps * cfg.training.batch_size
    train_loader = get_dataloader(cfg.dataset, cfg.training, "train")

    # Construct neural field model
    model = get_modulated_neural_field(
        nef_cfg=cfg.nef,
        num_in=train_loader.dataset.dim,
        num_out=train_loader.dataset.channels,
        num_signals=cfg.meta_learning.outer_batch_size,  # Number of patients per outer step.
    )

    model = model.to(device)
    log.info(f"Number of parameters: {sum([p.numel() for p in model.parameters()])}. ")

    logger = MultiCBCTReconLogger(train_loader.dataset, cfg.log_steps)
    checkpointer = MamlCheckpointMaker(
        experiment_path / "checkpoints", cfg.training.checkpoint_steps
    )
    if cfg.load_weights:
        checkpointer.load_best_checkpoint(model)
    if cfg.training.do:
        train_meta_init(
            model=model,
            optimizer_cfg=cfg.optimizer,
            meta_learning_cfg=cfg.meta_learning,
            train_loader=train_loader,
            device=device,
            logger=logger,
            checkpointer=checkpointer,
        )

    log.info(f"Used GPU memory before loading model: {torch.cuda.memory_allocated()}.")
    model = model.to(device)
    log.info(f"Used GPU memory after loading model: {torch.cuda.memory_allocated()}.")
    for i, p in enumerate(model.conditioning.codes):
        if i != cfg.val_dataset.volume_id:
            p.to("cpu")
    log.info(
        f"Used GPU memory after unloading part of the model: {torch.cuda.memory_allocated()}."
    )
    log.info(f"Number of parameters: {sum([p.numel() for p in model.parameters()])}. ")
    model = model.to(device)

    logger = MultiCBCTReconLogger(train_loader.dataset, cfg.log_steps)
    checkpointer = MamlCheckpointMaker(
        experiment_path / "checkpoints", cfg.training.checkpoint_steps
    )

    ##############
    # Validation #
    ##############
    # # where all results should be placed for this specific run
    # noisy_name = "noisy" if cfg.val_dataset.noisy_projections else "clean"
    # output_dir = (
    #     experiment_path
    #     / f"{cfg.val_dataset.stage}_{cfg.val_dataset.projs_sample_step}_{noisy_name}"
    # )
    # output_dir.mkdir(exist_ok=True)
    #
    # # allows for some logging to happen in wandb, because it needs incremental steps
    # starting_point = 0
    # for patient in range(25):
    #     checkpointer.load_best_checkpoint(model)
    #
    #     cfg.val_dataset.volume_id = patient
    #     val_loader = get_dataloader(
    #         cfg.val_dataset, cfg.validation, cfg.val_dataset.stage
    #     )
    #     val_logger = MultiCBCTLogger(
    #         val_loader.dataset,
    #         cfg.log_steps,
    #         val_log_steps=cfg.val_log_steps,
    #         stage=cfg.val_dataset.stage,
    #         starting_step=starting_point,
    #         metrics_folder=output_dir,
    #     )
    #
    #     validate(
    #         model,
    #         val_loader=val_loader,
    #         optimizer_cfg=cfg.val_optimizer,
    #         validation_cfg=cfg.validation,
    #         device=device,
    #         logger=val_logger,
    #         output_dir=output_dir,
    #     )
    #
    #     # Save the metrics from the logger
    #     val_logger.save_metrics(output_dir / f"patient_{patient}.csv")
    #     starting_point = val_logger.val_step_number

    wandb.finish()

    return None


# def process_batch(data, dset, device, extent):
#     target_pixels, sources, rays, vol_bboxes, patient_idx, noiseless_projs = data
#     # move all data to the device
#     target_pixels = target_pixels.to(device)
#     sources = sources.to(device)
#     rays = rays.to(device)
#     vol_bboxes = vol_bboxes.to(device)
#     noiseless_projs = noiseless_projs.to(device)
#
#     # compute intersection points with the bounding box
#     tmin, tmax = ConebeamRenderer.ray_box_intersection(
#         sources[None, ...],
#         rays[None, ...],
#         vol_bboxes[:, 0],
#         vol_bboxes[:, 1],
#     )
#
#     small_delta = dset.geometries[0].vol_spacing[0] / 2
#
#     # sample the points along the rays
#     samples = (
#         torch.linspace(0, 1, dset.num_steps, device=device)[None, None, :]
#         * (tmax - small_delta - tmin)[..., None]
#         + tmin[..., None]
#         + small_delta / 2
#     )
#
#     samples = samples[0]
#     samples = samples.expand([rays.shape[0], dset.num_steps])
#
#     # get intervals between samples
#     mids = 0.5 * (samples[..., 1:] + samples[..., :-1])
#     upper = torch.cat([mids, samples[..., -1:]], -1)
#     lower = torch.cat([samples[..., :1], mids], -1)
#     # stratified samples in those intervals
#     t_rand = torch.rand(samples.shape, device=lower.device)
#     samples = lower + (upper - lower) * t_rand
#     sampled_points = sources[:, None, :] + samples[:, :, None] * rays[:, None, :]
#
#     # compute the weights for the integral calculation
#     weights = samples[..., 1:] - samples[..., :-1]  # [B, R, P]
#     # make the last weight to 0
#     weights = torch.cat(
#         [
#             weights,
#             torch.Tensor([1e-10]).expand(weights[..., :1].shape).to(weights.device),
#         ],
#         -1,
#     )
#
#     # normalize points between 0 and 1 inside the reference bounding box
#     coords = 2 * sampled_points / extent[None, None, :]
#     pixels_mask = noiseless_projs < 1e-8
#     coords[pixels_mask] = 0
#     weights[pixels_mask] = 0
#
#     return coords, target_pixels, weights, patient_idx, pixels_mask


def train_meta_init(
    model,
    optimizer_cfg,
    meta_learning_cfg,
    train_loader,
    device,
    logger,
    checkpointer,
):
    # Since we're doing meta learning, we initialize only the weights for a single patient. This set of weights becomes
    # the initialization for all patients in the inner loop, and is optimized in the outer loop.
    outer_nmf = model.conditioning.neural_modulation_fields[0]

    # Get some memory back
    model.conditioning.neural_modulation_fields = None

    # Create a copy of the weights for each patient in the inner loop.
    model.conditioning.neural_modulation_fields = torch.nn.ModuleList(
        [_clone_module(outer_nmf) for _ in range(meta_learning_cfg.outer_batch_size)]
    )
    # Make sure grad is retained for inner parameters, otherwise SGD isn't updating them.
    _retain_grads_module(model.conditioning)

    # Set model to train mode
    model.train()

    nef_optimizer = get_optimizer(
        optimizer_cfg.nef,
        model.nef.parameters(),
        lr=meta_learning_cfg.lr,
    )
    conditioning_outer_optimizer = get_optimizer(
        optimizer_cfg.conditioning.outer_optimizer,
        outer_nmf.parameters(),
        lr=meta_learning_cfg.lr_conditioning_outer
    )
    conditioning_inner_optimizer = get_optimizer(
        optimizer_cfg.conditioning.inner_optimizer,
        model.conditioning.parameters(),
        lr=meta_learning_cfg.lr_conditioning_inner,
        differentiable=True,  # Muchos importantos, allows for optimizing non-leaf tensors.
    )

    max_time = meta_learning_cfg.max_time
    start_time = time.time()
    interrupted = False
    epoch = 0
    while not interrupted:
        if epoch > meta_learning_cfg.epochs:
            break

        # We keep track of which patients have already been used for the inner loop, so we can recover the correct
        # weights for the inner loop.
        patients_seen_this_inner_loop = dict()

        # training loop dataloader with progress bar using tqdm
        for i, batch in enumerate(tqdm(train_loader, desc="Training")):

            # Get patient idx
            patient_idx = batch[2][0].item()

            # If we've already seen this patient in this inner loop, we recover their inner loop patient_idx.
            if patient_idx in patients_seen_this_inner_loop:
                inner_loop_patient_idx = patients_seen_this_inner_loop[patient_idx]
            else:
                inner_loop_patient_idx = int(i % meta_learning_cfg.outer_batch_size)
                patients_seen_this_inner_loop[patient_idx] = inner_loop_patient_idx

            # Set requires_grad to False for all nef parameters, as well as the outer loop weights, so we don't
            # accumulate gradients for them
            _set_requires_grad(model.nef, False)
            _set_requires_grad(outer_nmf, False)

            # Perform inner loop optimization steps
            for j in range(meta_learning_cfg.inner_steps):

                # In case this is the last inner loop step, we need to make sure that the gradients are retained for the
                # outer loop weights as well as the nef.
                if j == meta_learning_cfg.inner_steps - 1:
                    _set_requires_grad(model.nef, True)
                    _set_requires_grad(outer_nmf, True)

                # We divide the batch up into meta_learning_cfg.inner_steps chunks, and perform an optimization step
                # for each chunk.
                inner_batch_size = batch[0].shape[0] // meta_learning_cfg.inner_steps
                coords = batch[0][j * inner_batch_size : (j + 1) * inner_batch_size]
                target_pixels = batch[1][j * inner_batch_size : (j + 1) * inner_batch_size]
                coords = coords.to(device)
                target_pixels = target_pixels.to(device)

                # Get projection data
                out = model(
                    coords,
                    torch.tensor(inner_loop_patient_idx).repeat(coords.shape[0]).to(device)
                )
                # compute the loss and update the model
                loss = torch.nn.functional.mse_loss(out[:, 0], target_pixels)

                # Compute gradients, only retain them for inner loop steps, otherwise we'll run out of memory.
                if (((i > 0 and i % meta_learning_cfg.outer_batch_size == 0) or i == len(train_loader) - 1)
                        and (j == meta_learning_cfg.inner_steps - 1)):
                    loss.backward(retain_graph=False)
                else:
                    loss.backward(retain_graph=True)

                # Perform gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update weights of inner loops
                conditioning_inner_optimizer.step()
                conditioning_inner_optimizer.zero_grad()

            # This is an outer loop step. We compute the loss for the entire batch, and perform an optimization step.
            if (i > 0 and i % meta_learning_cfg.outer_batch_size == 0) or i == len(train_loader) - 1:

                # Reset patient seen set
                patients_seen_this_inner_loop = dict()

                # Perform outer step for 5 epochs followed by nef step for 1 epoch
                if epoch % 5 == 0:
                    nef_optimizer.step()
                    nef_optimizer.zero_grad()
                else:
                    conditioning_outer_optimizer.step()
                    conditioning_outer_optimizer.zero_grad()

                # Do some logging.
                logger.train_step(model, loss.detach())

                nef_optimizer.zero_grad()
                conditioning_outer_optimizer.zero_grad()

                # Remove previous inner optimizer
                del conditioning_inner_optimizer

                # Reinitialize the inner weights
                model.conditioning.neural_modulation_fields = torch.nn.ModuleList(
                    [_clone_module(outer_nmf) for _ in range(meta_learning_cfg.outer_batch_size)]
                )
                # Make sure grad is retained for inner parameters
                _retain_grads_module(model.conditioning)

                conditioning_inner_optimizer = get_optimizer(
                    optimizer_cfg.conditioning.inner_optimizer,
                    model.conditioning.parameters(),
                    lr=meta_learning_cfg.lr_conditioning_inner,
                    differentiable=True,
                )

                checkpointer.step(
                    model, nef_optimizer, loss.detach(), metric=-logger.best_psnr
                )

            # elapsed time in seconds
            elapsed_time = time.time() - start_time
            if elapsed_time > max_time:
                logger.log_metrics(model, loss, stage="train")
                logger.log_volume(model, loss, stage="train")
                log.info("Time limit reached. Finishing training.")
                interrupted = True
                break
        epoch += 1


# def validate(
#     model,
#     optimizer_cfg,
#     validation_cfg,
#     val_loader,
#     device,
#     logger,
#     output_dir,
# ):
#     model.train()
#     optimizer = get_optimizer(
#         optimizer_config=optimizer_cfg,
#         parameters=model.conditioning.codes[
#             val_loader.dataset.volume_id + 200
#         ].parameters(),
#         lr=validation_cfg.lr,
#     )
#
#     # Early stopping
#     early_stopping = EarlyStopping(
#         patience=validation_cfg.patience,
#         delta=validation_cfg.delta,
#         step_interval=validation_cfg.early_stopping_step_interval,
#     )
#
#     max_time = validation_cfg.max_time
#
#     dset = val_loader.dataset
#     extent = (dset.rendering_bbox[1] - dset.rendering_bbox[0]).to(device)
#
#     start_time = time.time()
#     logging_time = 0
#     interrupted = False
#     epoch = 0
#     log.info(f"GPU memory: {torch.cuda.memory_allocated()}")
#     memory_file = output_dir / f"memory_{dset.volume_id}.txt"
#     memory_file.write_text(f"GPU memory: {torch.cuda.memory_allocated()/1024**3} GB")
#     itercount = 0
#     while not interrupted:
#         for i, batch in enumerate(tqdm(val_loader, desc="Validation")):
#             # Get batch of projections
#             (
#                 coords,
#                 target_pixels,
#                 weights,
#                 patient_idx,
#                 pixels_mask,
#             ) = process_batch(batch, dset, device, extent)
#
#             log.info(f"GPU memory: {torch.cuda.memory_allocated()}")
#
#             # Get projection data
#             out = model(coords, patient_idx)
#             # Sum over height to compute projection
#             y_pred = (out * weights[:, :, None]).sum(dim=1)
#             y_pred[pixels_mask] = 0
#             loss = torch.nn.functional.mse_loss(y_pred[:, 0], target_pixels)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             logging_time_start = time.time()
#             logger.val_step(model, loss)
#             logging_time += time.time() - logging_time_start
#             # offset the time to make sure that the logging time is not counted
#             logger.start_time += time.time() - logging_time_start
#
#             # elapsed time in seconds
#             elapsed_time = time.time() - start_time - logging_time
#             if (
#                 (elapsed_time > max_time)
#                 or early_stopping.early_stop
#                 or itercount >= validation_cfg.max_iters
#             ):
#                 logger.log_metrics(model, loss, stage=dset.stage, store_volume=True)
#                 logger.log_volume(model, loss, stage=dset.stage)
#                 if elapsed_time > max_time:
#                     log.info("Time limit reached. Finishing optimizing.")
#                 elif early_stopping.early_stop:
#                     log.info("Early stopping. Finishing optimizing.")
#                 elif itercount >= validation_cfg.max_iters:
#                     log.info("Max iterations reached. Finishing optimizing.")
#                 interrupted = True
#                 break
#
#             itercount += 1
#
#         epoch += 1
#         if epoch >= validation_cfg.max_epochs:
#             log.info("Max epochs reached. Finishing training.")
#             break


if __name__ == "__main__":
    main()

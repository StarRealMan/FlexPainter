import os
import shutil
import math
import argparse
import logging
from tqdm.auto import tqdm
from omegaconf import OmegaConf

import torch
from torch.optim.swa_utils import AveragedModel
import datasets
import diffusers
import transformers
from torch.utils.tensorboard import SummaryWriter
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from accelerate import Accelerator
from diffusers.optimization import get_scheduler

from spuv.rasterize import NVDiffRasterizerContext
from model.weighter_net import WeighterNet

from data.objv_dataset import loader
from utils.validation import load_validation
from utils.renderer import data_process
from utils.video import render_video
from utils.loss import Loss_Func

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()

    return args.config

def main():
    torch.multiprocessing.set_start_method('spawn')
    args = OmegaConf.load(parse_args())
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(logging_dir)

    train_dataloader = loader(**args.data_config)

    model = WeighterNet(args.model_config)
    model.train()
    if args.use_ema:
        ema_model = AveragedModel(model)
        ema_model.eval()
        ema_model.to(accelerator.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    global_step = 0
    first_epoch = 0

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    total_batch_size = args.data_config.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.data_config.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

            if args.use_ema:
                ema_model.load_state_dict(torch.load(os.path.join(args.output_dir, path, "ema_model.pth")))
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    if not args.disable_sampling:
        camera_dist = torch.tensor(4)
        fovy = math.radians(30)
        bg = torch.tensor(args.bg_color, device=accelerator.device)
        ctx = NVDiffRasterizerContext('cuda', accelerator.device)
        loss_func = Loss_Func(args.perceptual_weight, bg, accelerator.device)
        val_samples = load_validation(ctx, args.sample_path, args.val_mesh_scale, bg)

        if args.use_ema:
            val_model = ema_model
        else:
            val_model = model

    for _ in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for _, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                meshes, uv_maps, gen_rgbs, mvps, timestep = batch
                with torch.no_grad():
                    uv_position, uv_normal, uv_mask, uv_bakes, uv_bake_masks, rgbs = data_process(
                        ctx, meshes, uv_maps, mvps, gen_rgbs, bg
                    )
                
                for single_uv_mask in uv_mask:
                    if single_uv_mask.sum() == 0:
                        print("skip currupted data")
                        continue

                uv_maps = torch.stack(uv_maps).permute(0, 3, 1, 2).to(torch.float32)
                uv_gen_bakes = uv_bakes[:, :, 3:6]
                uv_rays = uv_bakes[:, :, 6:9]
                uv_scores = uv_bakes[:, :, 9:10]
                uv_bakes = uv_bakes[:, :, :3]

                output = model(
                    gen_bakes = uv_gen_bakes,
                    rays = uv_rays,
                    ref_scores = uv_scores,
                    position_map = uv_position,
                    normal_map = uv_normal,
                    timestep = timestep,
                    mesh = meshes,
                    uv_bake_masks = uv_bake_masks,
                )

                loss = loss_func(
                    model_out = output,
                    uv_maps = uv_maps,
                    ctx = ctx,
                    meshes = meshes,
                    mvps = mvps,
                    timestep = timestep,
                )

                avg_loss = accelerator.gather(loss.repeat(args.data_config.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if args.use_ema:
                    ema_model.update_parameters(model)

                os.system('nvidia-smi')

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    writer.add_scalar("train_loss", train_loss, global_step)
                train_loss = 0.0

                if not args.disable_sampling and (global_step % args.sample_every == 0 or (args.sanity_check and global_step == 1)):
                    if accelerator.is_main_process:
                        print(f"Sampling for step {global_step}...")
                        
                        for i, val_sample in enumerate(val_samples):
                            uv_maps = torch.stack(uv_maps).permute(0, 3, 1, 2).to(torch.float32)
                            uv_gen_bakes = uv_bakes[:, :, 3:6]
                            uv_rays = uv_bakes[:, :, 6:9]
                            uv_scores = uv_bakes[:, :, 9:10]
                            uv_bakes = uv_bakes[:, :, :3]

                            output = val_model(**val_samples)

                            result = output

                            writer.add_image(f"result #{i}", result.squeeze(0), global_step)
                            writer.add_image(f"gt #{i}", val_sample["uv_map"].permute(2, 0, 1), global_step)
                            print(f"Result for prompt #{i} is generated")

                            video_path = os.path.join(args.output_dir, "video")
                            if not os.path.exists(video_path):
                                os.makedirs(video_path)
                            video_file = os.path.join(video_path, f"{global_step}_{i}.mp4")
                            render_video(args.render_frame_num, args.render_ele, camera_dist, fovy, accelerator.device, 
                                         ctx, val_sample["mesh"], result.squeeze(0), args.render_size, bg, video_file)

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    if args.use_ema:
                        torch.save(ema_model.state_dict(), os.path.join(save_path, "ema_model.pth"))
                    logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()
        accelerator.end_training()

if __name__ == "__main__":
    main()
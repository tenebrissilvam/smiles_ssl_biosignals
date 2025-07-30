import os
import time

import torch
import util.misc as misc
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from util.dataset import build_dataset, get_dataloader
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.optimizer import get_optimizer_from_config


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    epoch,
    loss_scaler,
    log_writer=None,
    config=None,
):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 5

    accum_iter = config["accum_iter"]
    optimizer.zero_grad()

    for data_iter_step, samples in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        samples = samples[0].type(torch.FloatTensor).to(device, non_blocking=True)

        _, loss, vq_loss, indices = model(samples)

        codebook_usage = len(torch.unique(indices))
        usage_ratio = codebook_usage / model.vq.num_embeddings
        diversity_loss = -torch.log(torch.tensor(usage_ratio, device=device))

        loss = loss + 0.2 * diversity_loss

        loss_value = loss.item()
        vq_loss_value = vq_loss.item()
        recon_loss_value = loss_value - vq_loss_value - 0.2 * diversity_loss.item()

        loss = loss / accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        metric_logger.update(loss=loss_value)
        metric_logger.update(recon_loss=recon_loss_value)
        metric_logger.update(vq_loss=vq_loss_value)
        metric_logger.update(codebook_usage=usage_ratio)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((epoch + data_iter_step / len(data_loader)) * 1000)
            log_writer.add_scalar("train/total_loss", loss_value, epoch_1000x)
            log_writer.add_scalar("train/recon_loss", recon_loss_value, epoch_1000x)
            log_writer.add_scalar("train/vq_loss", vq_loss_value, epoch_1000x)
            log_writer.add_scalar("train/codebook_usage", usage_ratio, epoch_1000x)

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_vqvae_model(model, config, device):
    """Train VQ-VAE model."""
    dataset_train = build_dataset(config["dataset"], split="train")
    data_loader_train = get_dataloader(
        dataset_train, mode="train", **config["dataloader"]
    )

    optimizer = get_optimizer_from_config(config["train"], model)
    loss_scaler = NativeScaler()
    log_writer = SummaryWriter(log_dir=config["output_dir"])

    print(f"Start training for {config['train']['epochs']} epochs")
    start_time = time.time()

    for epoch in tqdm(range(config["train"]["epochs"])):
        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer,
            config["train"],
        )

        if epoch % 5 == 0 or epoch + 1 == config["train"]["epochs"]:
            misc.save_model(
                config,
                os.path.join(config["output_dir"], f"checkpoint-{epoch}.pth"),
                epoch,
                model,
                optimizer,
                loss_scaler,
            )

    print(f"Training completed in {time.time() - start_time:.2f}s")

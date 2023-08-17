import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import tqdm

import commons
import utils
from data_utils import (
  TextAudioLoader,
  TextAudioCollate,
  DistributedBucketSampler
)
from models import (
  SynthesizerTrn,
  DurationDiscriminator
)
from losses import (
  dur_generator_loss,
  dur_discriminator_loss
)

import monotonic_align

from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols


torch.backends.cudnn.benchmark = True
global_step = 0


def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '6060'

  hps = utils.get_hparams()
  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)
  
  if "use_mel_posterior_encoder" in hps.model.keys() and hps.model.use_mel_posterior_encoder == True:
    print("Using mel posterior encoder for VITS2")
    posterior_channels = 80 #vits2
    hps.data.use_mel_posterior_encoder = True
  else:
    print("Using lin posterior encoder for VITS1")
    posterior_channels = hps.data.filter_length // 2 + 1  
    hps.data.use_mel_posterior_encoder = False

  train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
  train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
      [32,300,400,500,600,700,800,900,1000],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextAudioCollate()
  train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler)
  if rank == 0:
    eval_dataset = TextAudioLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(eval_dataset, num_workers=8, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)
  
  if "use_transformer_flows" in hps.model.keys() and hps.model.use_transformer_flows == True:
    print("Using transformer flows for VITS2")
    use_transformer_flows = True
    transformer_flow_type = hps.model.transformer_flow_type
    assert transformer_flow_type in ["pre_conv", "fft"], "transformer_flow_type must be one of ['pre_conv', 'fft']"
  else:
    print("Using normal flows for VITS1")
    use_transformer_flows = False

  if "use_spk_conditioned_encoder" in hps.model.keys() and hps.model.use_spk_conditioned_encoder == True:
    if hps.data.n_speakers == 0:
      print("Warning: use_spk_conditioned_encoder is True but n_speakers is 0")
    print("Setting use_spk_conditioned_encoder to False as model is a single speaker model")
    use_spk_conditioned_encoder = False
  else:
    print("Using normal encoder for VITS1")
    use_spk_conditioned_encoder = False

  if "use_noise_scaled_mas" in hps.model.keys() and hps.model.use_noise_scaled_mas == True:
    print("Using noise scaled MAS for VITS2")
    use_noise_scaled_mas = True
    mas_noise_scale_initial = 0.01
    noise_scale_delta = 2e-6
  else:
    print("Using normal MAS for VITS1")
    use_noise_scaled_mas = False
    mas_noise_scale_initial = 0.0
    noise_scale_delta = 0.0
  
  net_g = SynthesizerTrn(
      len(symbols),
      posterior_channels,
      hps.train.segment_size // hps.data.hop_length,
      mas_noise_scale_initial = mas_noise_scale_initial,
      noise_scale_delta = noise_scale_delta,
      **hps.model).cuda(rank)
  
  # load VITS2 model
  _, _, _, _ = utils.load_checkpoint(hps.vits2_checkpoint_path, net_g, None)
    
  for param in net_g.parameters():
    param.requires_grad = False  # Freeze all parameters by default
    
  # Unfreeze parameters in the Duration Predictor module
  for param in net_g.dp.parameters():
      param.requires_grad = True

  net_d = DurationDiscriminator(hps.model.hidden_channels+1, hps.model.hidden_channels).cuda(rank)
    
  optim_g = torch.optim.AdamW(
      net_g.dp.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
  net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)

  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    epoch_str = 1
    global_step = 0

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  scaler = GradScaler(enabled=hps.train.fp16_run)

  for epoch in range(epoch_str, hps.train.dp_epochs + 1):
    if rank==0:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None)
    scheduler_g.step()
    scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
  net_g, net_d = nets
  optim_g, optim_d = optims
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers

  train_loader.batch_sampler.set_epoch(epoch)
  global global_step
  
  net_g.train()
  net_d.train()
  if rank == 0:
      loader = tqdm.tqdm(train_loader, desc='Loading train data')
  else:
      loader = train_loader
  for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(loader):
    if net_g.module.use_noise_scaled_mas:
      current_mas_noise_scale = net_g.module.mas_noise_scale_initial - net_g.module.noise_scale_delta * global_step
      net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)

    with autocast(enabled=hps.train.fp16_run):
      
      with torch.no_grad():
        h_text, m_p, logs_p, h_text_mask = net_g.module.enc_p(x, x_lengths)
        z, m_q, logs_q, spec_mask = net_g.module.enc_q(spec, spec_lengths, g=None)
        z_p = net_g.module.flow(z, spec_mask, g=None)
        
        # estimate d
        # negative cross-entropy
        s_p_sq_r = torch.exp(-2 * logs_p) # [b, d, t]
        neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True) # [b, 1, t_s]
        neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
        neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r)) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
        neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True) # [b, 1, t_s]
        neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

        if net_g.module.use_noise_scaled_mas:
          epsilon = torch.std(neg_cent) * torch.randn_like(neg_cent) * net_g.module.current_mas_noise_scale
          neg_cent = neg_cent + epsilon

        attn_mask = torch.unsqueeze(h_text_mask, 2) * torch.unsqueeze(spec_mask, -1)
        attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()
        
        d = attn.sum(2)
      
      # d_hat  
      d_hat = net_g.module.dp(h_text, h_text_mask, g=None, reverse=True, noise_scale=1.0)
      d_hat = torch.exp(d_hat) * h_text_mask
      
      # Discriminator
      d_disc_r, d_disc_g = net_d(h_text, d, d_hat.detach())
      with autocast(enabled=False):
        loss_disc = dur_discriminator_loss(d_disc_r, d_disc_g)
        loss_disc_all = loss_disc
    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    with autocast(enabled=hps.train.fp16_run):
      # Generator
      d_disc_r, d_disc_g = net_d(h_text, d, d_hat.detach())
      with autocast(enabled=False):
        loss_mse = F.l1_loss(d_hat, d)
        loss_gen = dur_generator_loss(d_disc_g)
        loss_gen_all = loss_gen + loss_mse
    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()

    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses = [loss_disc, loss_gen, loss_mse]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([x.item() for x in losses] + [global_step, lr])
        
        scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
        scalar_dict.update({"loss/g/mse": loss_mse})
        scalar_dict.update({"loss/gen": loss_gen})
        image_dict = { 
            "all/attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy())
        }
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images=image_dict,
          scalars=scalar_dict)

      if global_step % hps.train.eval_interval == 0:
        evaluate(hps, net_g, eval_loader, writer_eval)
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
        utils.clean_checkpoints(path_to_models=hps.model_dir, n_ckpts_to_keep=2, sort_by_time=True)
    global_step += 1
  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))

 
def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
      for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(eval_loader):
        x, x_lengths = x.cuda(0), x_lengths.cuda(0)
        spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
        y, y_lengths = y.cuda(0), y_lengths.cuda(0)

        # remove else
        x = x[:1]
        x_lengths = x_lengths[:1]
        spec = spec[:1]
        spec_lengths = spec_lengths[:1]
        y = y[:1]
        y_lengths = y_lengths[:1]
        break
      y_hat, attn, mask, *_ = generator.module.infer(x, x_lengths, max_len=1000)
      y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length

      if hps.model.use_mel_posterior_encoder or hps.data.use_mel_posterior_encoder:
        mel = spec
      else:
        mel = spec_to_mel_torch(
            spec, 
            hps.data.filter_length, 
            hps.data.n_mel_channels, 
            hps.data.sampling_rate,
            hps.data.mel_fmin, 
            hps.data.mel_fmax)
      y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
      )
    image_dict = {
      "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {
      "gen/audio": y_hat[0,:,:y_hat_lengths[0]]
    }
    if global_step == 0:
      image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
      audio_dict.update({"gt/audio": y[0,:,:y_lengths[0]]})

    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      images=image_dict,
      audios=audio_dict,
      audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()

                           
if __name__ == "__main__":
  main()

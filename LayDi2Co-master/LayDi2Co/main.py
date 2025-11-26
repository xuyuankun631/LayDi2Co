import os
import torch
from absl import flags, app
from accelerate import Accelerator
from ml_collections import config_flags

from logger_set import LOG
from utils import set_seed
from data_loaders.publaynet import PublaynetLayout
from data_loaders.rico import RicoLayout
from data_loaders.magazine import MagazineLayout

from models.discrete_model import DiscreteModel
from models.continuous_model import ContinuousModel
from diffusion import DiscreteDiffusionScheduler, ContinuousDiffusionScheduler
from trainers.trainer import TrainLoopLayDi2CoTwoStage



FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "configs/remote/publaynet_config.py", lock_config=False)
flags.DEFINE_string("workdir", default='test', help="Work unit directory.")
flags.mark_flags_as_required(["config"])


def main(*args, **kwargs):
    config = init_job()
    LOG.info("Loading data...")

    if config.dataset == 'publaynet':
        train_data = PublaynetLayout(config.train_json, max_num_com=config.max_num_comp)
        val_data = PublaynetLayout(config.val_json, train_data.max_num_comp)
    elif config.dataset == 'rico':
        train_data = RicoLayout(config.dataset_path, 'train', max_num_comp=config.max_num_comp)
        val_data = RicoLayout(config.dataset_path, 'val', train_data.max_num_comp)
    elif config.dataset == 'magazine':
        train_data = MagazineLayout(config.train_json, max_num_com=config.max_num_comp)
        val_data = MagazineLayout(config.val_json, train_data.max_num_comp)
    else:
        raise NotImplementedError

    assert config.categories_num == train_data.categories_num
    accelerator = Accelerator(
        split_batches=config.optimizer.split_batches,
        gradient_accumulation_steps=config.optimizer.gradient_accumulation_steps,
        mixed_precision=config.optimizer.mixed_precision,
        project_dir=config.log_dir,
    )
    LOG.info(accelerator.state)
    LOG.info("Creating discrete & continuous models...")
    discrete_model = DiscreteModel(categories_num=config.categories_num,
                                 latent_dim=config.latent_dim,
                                 num_layers=config.num_layers, num_heads=config.num_heads,
                                 dropout_r=config.dropout_r, activation=config.activation,
                                 cond_emb_size=config.cond_emb_size,
                                 cat_emb_size=config.cls_emb_size).to(accelerator.device)

    continuous_model = ContinuousModel(categories_num=config.categories_num,
                                     latent_dim=config.latent_dim,
                                     num_layers=config.num_layers, num_heads=config.num_heads,
                                     dropout_r=config.dropout_r, activation=config.activation,
                                     cond_emb_size=config.cond_emb_size,
                                     cat_emb_size=config.cls_emb_size).to(accelerator.device)
    LOG.info("Creating discrete & continuous diffusion schedulers...")
    discrete_diffusion = DiscreteDiffusionScheduler(
        alpha=0.1, beta=0.15, seq_max_length=config.max_num_comp,
        device=accelerator.device,
        discrete_features_names=[('cat', config.categories_num)],
        num_discrete_steps=[config.num_discrete_steps],
        num_cont_steps=config.num_cont_timesteps
    )

    continuous_diffusion = ContinuousDiffusionScheduler(
        num_train_timesteps=config.num_cont_timesteps,
        beta_schedule=config.beta_schedule,
        device=accelerator.device
    )

    LOG.info(f"Discrete model params: {sum(p.numel() for p in discrete_model.parameters())}")
    LOG.info(f"Continuous model params: {sum(p.numel() for p in continuous_model.parameters())}")

    LOG.info("Starting two-stage training...")
    trainer = TrainLoopLayDi2CoTwoStage(
        accelerator=accelerator,
        discrete_model=discrete_model,
        continuous_model=continuous_model,
        discrete_diffusion=discrete_diffusion,
        continuous_diffusion=continuous_diffusion,
        train_data=train_data, val_data=val_data,
        opt_conf=config.optimizer,
        log_interval=config.log_interval,
        save_interval=config.save_interval,
        categories_num=config.categories_num,
        device=accelerator.device,
        resume_from_checkpoint=config.resume_from_checkpoint,
        full_config=config,
        discrete_ckpt_path= "/root/private/LayDi2Co/logs/publaynet_final/checkpoints2/discrete_checkpoint-299/model.safetensors",
    )
    trainer.train()


def init_job():
    config = FLAGS.config
    config.log_dir = config.log_dir / FLAGS.workdir
    config.optimizer.ckpt_dir = config.log_dir / 'checkpoints'
    config.optimizer.samples_dir = config.log_dir / 'samples'
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.optimizer.samples_dir, exist_ok=True)
    os.makedirs(config.optimizer.ckpt_dir, exist_ok=True)
    set_seed(config.seed)
    return config


if __name__ == '__main__':
    app.run(main)

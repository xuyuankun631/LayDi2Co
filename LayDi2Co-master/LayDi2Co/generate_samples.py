import os
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from data_loaders.publaynet import PublaynetLayout
from data_loaders.rico import RicoLayout
from data_loaders.magazine import MagazineLayout
from logger_set import LOG
from absl import flags, app
from ml_collections import config_flags
from models.discrete_model import DiscreteModel
from models.continuous_model import ContinuousModel
from diffusion import DiscreteDiffusionScheduler, ContinuousDiffusionScheduler
from utils import set_seed, draw_layout_opacity

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "/root/private/LayDi2Co-master/LayDi2Co/configs/remote/publaynet_config.py",
                                lock_config=False)
flags.DEFINE_string("workdir", default='test', help="Work unit directory.")
flags.DEFINE_string("discrete_ckpt", None, help="Path to discrete ckpt.")
flags.DEFINE_string("continuous_ckpt", None, help="Path to continuous ckpt.")
flags.DEFINE_string("cond_type", default='all', help="Condition type to sample from.")
flags.DEFINE_bool("save", default=False, help="Whether to save generated images.")
flags.mark_flags_as_required(["config", "discrete_ckpt", "continuous_ckpt"])



def sample_discrete_labels(model, diffusion, batch, num_steps, categories_num, device):
    shape = batch['cat'].shape
    cat = (categories_num - 1) * torch.ones(shape, dtype=torch.long, device=device)
    for step in reversed(range(num_steps)):
        t = torch.full((shape[0],), step, dtype=torch.long, device=device)

        noisy_batch = {'cat': cat}
        with torch.no_grad():
            logits = model(batch, noisy_batch, t)
        desc_pred = {'cat': logits}
        _, step_cat = diffusion.step_jointly(None, desc_pred, timestep=t, sample=None)
        cat = step_cat['cat']
    return cat




def sample_continuous_boxes_cfg(model, diffusion, batch, pred_labels, num_steps, device, guidance_w):
    shape = batch['box'].shape
    noisy_boxes = torch.randn(shape, device=device)
    for step in reversed(range(100)):  # 100

        t_batch = torch.full((shape[0],), step, dtype=torch.long, device=device)
        t_scalar = torch.tensor([step], device=device)

        null_labels = torch.full_like(pred_labels, -1)
        with torch.no_grad():
            pred_cond = model(batch, noisy_boxes, t_batch, pred_labels)
            pred_uncond = model(batch, noisy_boxes, t_batch, null_labels)
            pred = (1 + 0) * pred_cond - 0 * pred_uncond
        scheduler_output, _ = diffusion.step_jointly(pred, None, timestep=t_scalar, sample=noisy_boxes)
        noisy_boxes = scheduler_output.prev_sample
    return scheduler_output.pred_original_sample







def main(_):
    config = init_job()
    config.optimizer.batch_size = 64

    LOG.info("Loading dataset...")
    if config.dataset == 'publaynet':
        val_data = PublaynetLayout(config.val_json, config.max_num_comp, config.cond_type)
    elif config.dataset == 'rico':
        val_data = RicoLayout(config.dataset_path, 'test', config.max_num_comp, config.cond_type)
    elif config.dataset == 'magazine':
        val_data = MagazineLayout(config.val_json, config.max_num_comp, config.cond_type)
    else:
        raise NotImplementedError

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOG.info("Loading models...")
    discrete_model = DiscreteModel(categories_num=config.categories_num, latent_dim=config.latent_dim,
                                 num_layers=config.num_layers, num_heads=config.num_heads,
                                 dropout_r=config.dropout_r, activation=config.activation,
                                 cond_emb_size=config.cond_emb_size, cat_emb_size=config.cls_emb_size).to(device)
    continuous_model = ContinuousModel(categories_num=config.categories_num, latent_dim=config.latent_dim,
                                     num_layers=config.num_layers, num_heads=config.num_heads,
                                     dropout_r=config.dropout_r, activation=config.activation,
                                     cond_emb_size=config.cond_emb_size, cat_emb_size=config.cls_emb_size).to(device)

    LOG.info(f"Loading discrete weights from {FLAGS.discrete_ckpt}")
    discrete_model.load_state_dict(torch.load(FLAGS.discrete_ckpt, map_location=device))

    LOG.info(f"Loading continuous weights from {FLAGS.continuous_ckpt}")
    continuous_model.load_state_dict(torch.load(FLAGS.continuous_ckpt, map_location=device))

    discrete_model.eval()
    continuous_model.eval()

    LOG.info("Creating diffusion schedulers...")
    discrete_diffusion = DiscreteDiffusionScheduler(
        alpha=0.1,
        beta=0.15,
        seq_max_length=config.max_num_comp,
        device=device,
        discrete_features_names=[('cat', config.categories_num)],
        num_discrete_steps=[config.num_discrete_steps],
        num_cont_steps=config.num_cont_timesteps
    )
    continuous_diffusion = ContinuousDiffusionScheduler(
        num_train_timesteps=config.num_cont_timesteps,
        beta_schedule=config.beta_schedule,
        device=device
    )

    val_loader = DataLoader(val_data, batch_size=config.optimizer.batch_size,
                            shuffle=False, num_workers=config.optimizer.num_workers)

    all_results = {'dataset_val': [], 'predicted_val': []}

    LOG.info("Start generating samples...")
    #
    for i, batch in enumerate(tqdm(val_loader, ascii=True)):
        batch = {k: v.to(device) for k, v in batch.items()}
        pred_labels = sample_discrete_labels(
            model=discrete_model,
            diffusion=discrete_diffusion,
            batch=batch,
            num_steps=config.num_discrete_steps,
            categories_num=config.categories_num,
            device=device
        )

        pred_boxes = sample_continuous_boxes_cfg(
            model=continuous_model,
            diffusion=continuous_diffusion,
            batch=batch,
            pred_labels=pred_labels,

            num_steps=config.num_cont_timesteps,
            device=device,
            guidance_w=0.02
        )

        box = batch['mask_box'] * pred_boxes + (1 - batch['mask_box']) * batch['box_cond']
        cat = batch['mask_cat'] * pred_labels + (1 - batch['mask_cat']) * batch['cat']
        box = box.cpu().numpy()
        cat = cat.cpu().numpy()
        all_results['dataset_val'].append(
            np.concatenate([batch['box_cond'].cpu().numpy(), np.expand_dims(batch['cat'].cpu().numpy(), -1)], axis=-1))
        all_results['predicted_val'].append(
            np.concatenate([box, np.expand_dims(cat, -1)], axis=-1))

        # 保存可视化
        if config.save:
            for b in range(box.shape[0]):
                tmp_box = box[b]
                tmp_cat = cat[b]
                tmp_box = tmp_box[~(tmp_box == 0.).all(1)]
                tmp_cat = tmp_cat[~(tmp_cat == 0)]
                tmp_box = (tmp_box / 2 + 1) / 2
                canvas = draw_layout_opacity(tmp_box, tmp_cat, None, val_data.idx2color_map, height=512)
                Image.fromarray(canvas).save(config.optimizer.samples_dir / f'{i}_{b}.jpg')
                tmp_box_gt = batch['box_cond'][b].cpu().numpy()
                tmp_cat_gt = batch['cat'][b].cpu().numpy()
                tmp_box_gt = tmp_box_gt[~(tmp_box_gt == 0.).all(1)]
                tmp_cat_gt = tmp_cat_gt[~(tmp_cat_gt == 0)]
                tmp_box_gt = (tmp_box_gt / 2 + 1) / 2
                canvas_gt = draw_layout_opacity(tmp_box_gt, tmp_cat_gt, None, val_data.idx2color_map, height=512)
                Image.fromarray(canvas_gt).save(config.optimizer.samples_dir / f'{i}_{b}_gt.jpg')

    with open(config.optimizer.samples_dir / f'results_{config.cond_type}.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    LOG.info(f"Saved results to {config.optimizer.samples_dir / f'results_{config.cond_type}.pkl'}")


def init_job():
    config = FLAGS.config
    config.log_dir = config.log_dir / FLAGS.workdir
    config.optimizer.ckpt_dir = config.log_dir / 'checkpoints'
    config.optimizer.samples_dir = config.log_dir / 'samples'
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.optimizer.samples_dir, exist_ok=True)
    os.makedirs(config.optimizer.ckpt_dir, exist_ok=True)
    set_seed(config.seed)


    assert FLAGS.cond_type in ['whole_box', 'loc', 'all']
    config.cond_type = FLAGS.cond_type
    config.save = FLAGS.save
    return config


if __name__ == '__main__':
    app.run(main)

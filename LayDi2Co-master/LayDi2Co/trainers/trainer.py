import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ml_collections import config_dict


from logger_set import LOG
from utils import masked_l2, masked_cross_entropy
from diffusers import get_scheduler
import matplotlib.pyplot as plt



class TrainLoopLayDi2CoTwoStage:
    def __init__(self, accelerator, discrete_model, continuous_model,
                 discrete_diffusion, continuous_diffusion,
                 train_data, val_data, opt_conf,
                 log_interval, save_interval, categories_num,
                 device='cpu',
                 resume_from_checkpoint=None,
                 full_config=None,
                 discrete_ckpt_path=None):
        """
        :param discrete_ckpt_path: 离散模型训练后保存的权重，用于连续阶段加载
        """
        self.accelerator = accelerator
        self.device = device
        self.train_data = train_data
        self.val_data = val_data
        self.opt_conf = opt_conf
        self.categories_num = categories_num
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.config = full_config
        self.resume_from_checkpoint = resume_from_checkpoint

        # 两个模型
        self.discrete_model = discrete_model
        self.continuous_model = continuous_model
        self.discrete_diffusion = discrete_diffusion
        self.continuous_diffusion = continuous_diffusion

        # 优化器
        self.optimizer_discrete = torch.optim.AdamW(discrete_model.parameters(), lr=opt_conf.lr,
                                                    betas=opt_conf.betas, weight_decay=opt_conf.weight_decay,
                                                    eps=opt_conf.epsilon)
        self.optimizer_continuous = torch.optim.AdamW(continuous_model.parameters(), lr=opt_conf.lr,
                                                      betas=opt_conf.betas, weight_decay=opt_conf.weight_decay,
                                                      eps=opt_conf.epsilon)

        # dataloader
        train_loader = DataLoader(train_data, batch_size=opt_conf.batch_size, shuffle=True,
                                  num_workers=opt_conf.num_workers)
        val_loader = DataLoader(val_data, batch_size=opt_conf.batch_size, shuffle=False,
                                num_workers=opt_conf.num_workers)

        # lr scheduler：根据 epoch × step 总步数
        num_training_steps_discrete = len(train_loader) * opt_conf.num_epochs_discrete
        num_training_steps_continuous = len(train_loader) * opt_conf.num_epochs_continuous

        self.lr_scheduler_discrete = get_scheduler(
            opt_conf.lr_scheduler, self.optimizer_discrete,
            num_warmup_steps=opt_conf.num_warmup_steps * opt_conf.gradient_accumulation_steps,
            num_training_steps=num_training_steps_discrete
        )
        self.lr_scheduler_continuous = get_scheduler(
            opt_conf.lr_scheduler, self.optimizer_continuous,
            num_warmup_steps=opt_conf.num_warmup_steps * opt_conf.gradient_accumulation_steps,
            num_training_steps=num_training_steps_continuous
        )



        # accelerator.prepare 包含 optimizer & scheduler
        (self.discrete_model, self.continuous_model,
         self.optimizer_discrete, self.optimizer_continuous,
         self.lr_scheduler_discrete, self.lr_scheduler_continuous,
         self.train_dataloader, self.val_dataloader) = accelerator.prepare(
            discrete_model, continuous_model,
            self.optimizer_discrete, self.optimizer_continuous,
            self.lr_scheduler_discrete, self.lr_scheduler_continuous,
            train_loader, val_loader
        )

        # 如果有 ckpt，连续阶段会用到
        self.discrete_ckpt_path = discrete_ckpt_path

        # 新增：用于记录离散阶段每个 epoch 的平均 loss
        self.discrete_epoch_losses = []

        LOG.info("Initialized two-stage training loop.")

    def train(self):
        LOG.info("=== Stage 1: Train discrete model ===")
        for epoch in range(self.opt_conf.num_epochs_discrete):
            self.train_discrete_epoch(epoch)
            if (epoch + 1) % 50 == 0:
                self.save_model(epoch, self.discrete_model, self.optimizer_discrete, prefix='discrete')

        # ===== 保存离散 loss 到 txt 并绘图 =====
        loss_save_dir = self.opt_conf.ckpt_dir
        os.makedirs(loss_save_dir, exist_ok=True)

        # 保存为 txt
        loss_txt_path = os.path.join(loss_save_dir, "discrete_train_loss.txt")
        with open(loss_txt_path, "w") as f:
            for i, loss_val in enumerate(self.discrete_epoch_losses):
                f.write(f"{i}\t{loss_val:.6f}\n")
        LOG.info(f"Saved discrete training losses to {loss_txt_path}")

        # 绘制并保存 PDF
        plt.figure(figsize=(8, 5))
        plt.plot(self.discrete_epoch_losses, label="Discrete Diffusion Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Average Cross-Entropy Loss")
        plt.title("Discrete Diffusion Model Training Loss Curve")
        plt.grid(True)
        plt.legend()
        loss_pdf_path = os.path.join(loss_save_dir, "discrete_train_loss.pdf")
        plt.savefig(loss_pdf_path, bbox_inches='tight')
        plt.close()
        LOG.info(f"Saved discrete loss curve to {loss_pdf_path}")




        # ===== 加载离散部分权重，冻结离散模型 =====
        if self.discrete_ckpt_path:
            LOG.info(f"Loading discrete model weights from {self.discrete_ckpt_path}")
            state = torch.load(self.discrete_ckpt_path, map_location=self.device)
            self.discrete_model.load_state_dict(state)
        self.discrete_model.eval()

        LOG.info("=== Stage 2: Train continuous model ===")
        for epoch in range(self.opt_conf.num_epochs_continuous):
            self.train_continuous_epoch(epoch)
            if (epoch + 1) % 50 == 0:
                self.save_model(epoch, self.continuous_model, self.optimizer_continuous, prefix='continuous')

    def train_discrete_epoch(self, epoch):
        self.discrete_model.train()
        total_loss = 0.0
        num_steps = 0
        progress_bar = tqdm(total=len(self.train_dataloader), desc=f"Discrete Epoch {epoch}", ascii=True)
        for step, batch in enumerate(self.train_dataloader):
            self.sample2dev(batch)
            bsz = batch['cat'].shape[0]
            t = torch.randint(0, self.discrete_diffusion.num_discrete_steps[0], (bsz,), device=self.device).long()
            noise = torch.randint(0, self.categories_num, batch['cat'].shape, device=self.device).long()
            _, noisy_batch = self.discrete_diffusion.add_noise_jointly(batch['cat'], {'cat': batch['cat']}, t, noise)

            with self.accelerator.accumulate(self.discrete_model):
                pred_logits = self.discrete_model(batch, noisy_batch, t)
                loss_cls = masked_cross_entropy(pred_logits, batch['cat'], batch['mask_cat'])
                loss_cls = loss_cls.mean()
                self.accelerator.backward(loss_cls)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.discrete_model.parameters(), 1.0)

                self.optimizer_discrete.step()
                self.lr_scheduler_discrete.step()
                self.optimizer_discrete.zero_grad()

            total_loss += loss_cls.item()
            num_steps += 1

            if step % self.log_interval == 0:
                LOG.info(f"[Discrete] Epoch {epoch} Step {step} Loss_cls={loss_cls.item():.4f}")
            progress_bar.update(1)
        progress_bar.close()

        avg_loss = total_loss / num_steps if num_steps > 0 else 0.0
        self.discrete_epoch_losses.append(avg_loss)




    def train_continuous_epoch(self, epoch):
        self.continuous_model.train()
        progress_bar = tqdm(total=len(self.train_dataloader), desc=f"Continuous Epoch {epoch}", ascii=True)
        for step, batch in enumerate(self.train_dataloader):
            self.sample2dev(batch)
            bsz = batch['box'].shape[0]
            t = torch.randint(0, 100, (bsz,), device=self.device).long()
            t_discrete = torch.randint(0, 1, (bsz,), device=self.device).long()

            # 用离散模型生成标签
            with torch.no_grad():
                noise = torch.randint(0, self.categories_num, batch['cat'].shape, device=self.device).long()
                _, noisy_batch = self.discrete_diffusion.add_noise_jointly(batch['cat'], {'cat': batch['cat']}, t_discrete, noise)
                pred_logits = self.discrete_model(batch, noisy_batch, t)
                pred_labels = pred_logits.argmax(dim=-1)

                # 50%真实 + 50%生成
                cond_labels = self.mix_labels(batch['cat'], pred_labels)


                # ---------- Apply classifier-free guidance masking ---------
                drop_prob = 0.1  # 10%
                mask = torch.rand_like(cond_labels.float()) < drop_prob
                cond_labels_cfg = cond_labels.clone()
                cond_labels_cfg[mask] = -1
                # -------------------------------------------------------------------

            # 加噪声
            noise_box = torch.randn(batch['box'].shape, device=self.device)
            noisy_boxes, _ = self.continuous_diffusion.add_noise_jointly(batch['box'], batch, t, noise_box)

            with self.accelerator.accumulate(self.continuous_model):
                pred_boxes = self.continuous_model(batch, noisy_boxes, t, cond_labels_cfg)
                # -------------------------------------------------------------------
                loss_box = masked_l2(batch['box_cond'], pred_boxes, batch['mask_box'])
                loss_box = loss_box.mean()

                self.accelerator.backward(loss_box)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.continuous_model.parameters(), 1.0) # 梯度裁剪，避免梯度爆炸
                self.optimizer_continuous.step()
                self.lr_scheduler_continuous.step()
                self.optimizer_continuous.zero_grad()

            if step % self.log_interval == 0:
                LOG.info(f"[Continuous] Epoch {epoch} Step {step} Loss_box={loss_box.item():.4f}")
            progress_bar.update(1)
        progress_bar.close()

    def save_model(self, epoch, model_to_save, optimizer, prefix):
        save_path = self.opt_conf.ckpt_dir / f"{prefix}_checkpoint-{epoch}/"
        os.makedirs(save_path, exist_ok=True)
        self.accelerator.save_state(save_path)
        torch.save(model_to_save.state_dict(), save_path / "model.safetensors")
        torch.save(model_to_save.state_dict(), save_path / "diffusion_pytorch_model.bin")
        torch.save(optimizer.state_dict(), save_path / "optimizer.bin")
        config_dict = self.config.to_dict()
        with open(save_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        LOG.info(f"Saved {prefix} model at {save_path}")

    def sample2dev(self, sample):
        # batch 转 device
        for k, v in sample.items():
            if isinstance(v, dict):
                for k1, v1 in v.items():
                    sample[k][k1] = v1.to(self.device)
            else:
                sample[k] = v.to(self.device)

    def mix_labels(self, real_labels, generated_labels):
        # 25%真实 + 75%生成 , <0.5 就表示每个位置有 50% 概率选真实标签。
        mask = torch.rand(real_labels.shape, device=real_labels.device) < 0.25
        return torch.where(mask, real_labels, generated_labels)

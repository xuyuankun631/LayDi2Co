from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from einops import rearrange
from labml_nn.sampling import Sampler
from torch.distributions import Categorical


class TemperatureSampler(Sampler):
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor):
        dist = Categorical(probs=logits / self.temperature)
        return dist.sample()


class DiscreteDiffusionScheduler:

    def __init__(self, alpha=0.1, beta=0.15, seq_max_length=16, device='cpu',
                 discrete_features_names: List[Tuple[str, int]] = None,
                 num_discrete_steps: List[int] = None,
                 num_cont_steps: int = 100):
        self.device = device
        self.num_cont_steps = num_cont_steps
        self.beta = beta
        self.alpha = alpha
        self.seq_max_length = seq_max_length
        self.discrete_features_names = discrete_features_names
        self.num_discrete_steps = num_discrete_steps
        self.cont2disc = {}
        self.transition_matrices = {}


        for (f_name, f_cat_num), f_steps in zip(discrete_features_names, num_discrete_steps):
            self.transition_matrices[f_name] = self.generate_transition_mat(f_cat_num, f_steps)


        self.sampler = TemperatureSampler(temperature=0.8)

    def add_noise_jointly(self, vec_cont, vec_cat: dict, timesteps: torch.IntTensor, noise):

        cat_res = {}
        for f_name, f_cat_num in self.discrete_features_names:
            t = timesteps
            prob_mat = [self.transition_matrices[f_name][t[i]][vec_cat[f_name][i]] for i in range(len(t))]

            prob_mat = torch.cat(prob_mat)
            cat_noise = torch.multinomial(prob_mat, 1, replacement=True)
            cat_noise = rearrange(cat_noise, '(d b) 1 -> d b', d=vec_cont.shape[0])
            cat_res[f_name] = cat_noise
        return vec_cont, cat_res



    def step_jointly(self, cont_output, cat_output: dict, timestep, sample, generator=None, return_dict=True):

        step_cat_res = {}
        for f_name, f_cat_num in self.discrete_features_names:

            t = timestep
            cls, _ = self.denoise_cat(cat_output[f_name], t,
                                      f_cat_num, self.transition_matrices[f_name])

            step_cat_res[f_name] = cls
        return cont_output, step_cat_res

    def generate_transition_mat(self, categories_num, num_discrete_steps):
        transition_mat = np.eye(categories_num) * (1 - self.alpha - self.beta) + self.alpha / categories_num
        transition_mat[:, -1] += self.beta
        transition_mat[-1, :] = 0
        transition_mat[-1, -1] = 1
        transition_mat_list = []
        curr_mat = transition_mat.copy()
        for _ in range(num_discrete_steps):
            transition_mat_list.append(torch.tensor(curr_mat).to(torch.float32).to(self.device))
            curr_mat = curr_mat @ transition_mat
        return transition_mat_list


    def denoise_cat(self, pred, t, cat_num, transition_mat_list):
        pred_prob = F.softmax(pred, dim=2)
        prob, cls = torch.max(pred_prob, dim=2)

        if isinstance(t, torch.Tensor):
            t = t.tolist()

        if isinstance(t, list) and len(set(t)) == 1:
            t = t[0]

        if isinstance(t, int):
            # 全 batch 使用一个 t
            if t > 1:
                m = torch.matmul(pred_prob.reshape((-1, cat_num)),
                                 transition_mat_list[t].float())
                m = m.reshape(pred_prob.shape)
                m[:, :, 0] = 0
                res = self.sampler(m)
            else:
                res = (cat_num - 1) * torch.ones_like(cls).to(torch.long)
                top = torch.topk(prob, prob.shape[1], dim=1)
                for ttt in range(prob.shape[0]):
                    res[ttt, top[1][ttt]] = cls[ttt, top[1][ttt]]
            return res, 0
        else:
            raise NotImplementedError("目前仅支持单一 t 时间步（int）用于推理/采样")






    @staticmethod
    def mapping_cont2disc(num_cont_steps, num_discrete_steps):
        block_size = num_cont_steps // num_discrete_steps
        cont2disc = {}
        for i in range(num_cont_steps):
            if i >= (num_discrete_steps - 1) * block_size:
                if num_cont_steps % num_discrete_steps != 0 and i >= num_discrete_steps * block_size:
                    cont2disc[i] = num_discrete_steps - 1
                else:
                    cont2disc[i] = i // block_size
            else:
                cont2disc[i] = i // block_size
        return cont2disc


class ContinuousDiffusionScheduler(DDPMScheduler):

    def __init__(self, num_train_timesteps=100, beta_schedule='squaredcos_cap_v2', device='cpu', **kwargs):
        super().__init__(num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule, **kwargs)
        self.device = device

    def add_noise_jointly(self, vec_cont: torch.FloatTensor, vec_cat, timesteps: torch.IntTensor, noise: torch.FloatTensor):

        noised_cont = super().add_noise(original_samples=vec_cont, timesteps=timesteps, noise=noise)

        return noised_cont, vec_cat

    def step_jointly(self, cont_output, cat_output, timestep, sample, generator=None, return_dict=True):
        box = super().step(model_output=cont_output, timestep=timestep.item(), sample=sample,
                           generator=generator, return_dict=return_dict)
        return box, cat_output

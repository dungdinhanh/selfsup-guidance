import os
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

import attr
import numpy as np
import torch
import torch.nn as nn
import yaml
from glide_text2im.tokenizer.simple_tokenizer import SimpleTokenizer
import torch.nn.functional as F
from .encoders import ImageEncoder, TextEncoder


@lru_cache()
def default_config_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


@attr.s
class CLIPModel:
    config: Dict[str, Any] = attr.ib()
    text_encoder: nn.Module = attr.ib()
    image_encoder: nn.Module = attr.ib()
    logit_scale: torch.Tensor = attr.ib()
    device: torch.device = attr.ib()
    tokenizer: SimpleTokenizer = attr.ib()

    def encode_prompts(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = []
        lens = []
        for prompt in prompts:
            sub_tokens, sub_len = self.tokenizer.padded_tokens_and_len(
                self.tokenizer.encode(prompt), self.text_encoder.max_text_len
            )
            tokens.append(sub_tokens)
            lens.append(sub_len)
        return (
            torch.tensor(tokens).to(dtype=torch.long, device=self.device),
            torch.tensor(lens).to(dtype=torch.long, device=self.device),
        )

    def text_embeddings(self, prompts: List[str]) -> torch.Tensor:
        tokens, lens = self.encode_prompts(prompts)
        z_t = self.text_encoder(tokens, lens)
        return z_t / (torch.linalg.norm(z_t, dim=-1, keepdim=True) + 1e-12)

    def image_embeddings(self, images: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z_i = self.image_encoder((images + 1) * 127.5, t)
        return z_i / (torch.linalg.norm(z_i, dim=-1, keepdim=True) + 1e-12)

    def cond_fn(self, prompts: List[str], grad_scale: float) -> Callable[..., torch.Tensor]:
        with torch.no_grad():
            z_t = self.text_embeddings(prompts)

        def cond_fn(x, t, grad_scale=grad_scale, **kwargs):
            with torch.enable_grad():
                x_var = x.detach().requires_grad_(True)
                z_i = self.image_embeddings(x_var, t)
                loss = torch.exp(self.logit_scale) * (z_t * z_i).sum()
                grad = torch.autograd.grad(loss, x_var)[0].detach()
            return grad * grad_scale
        return cond_fn

@attr.s
class CLIPModelRobust:
    config: Dict[str, Any] = attr.ib()
    text_encoder: nn.Module = attr.ib()
    image_encoder: nn.Module = attr.ib()
    logit_scale: torch.Tensor = attr.ib()
    device: torch.device = attr.ib()
    tokenizer: SimpleTokenizer = attr.ib()

    def encode_prompts(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = []
        lens = []
        for prompt in prompts:
            sub_tokens, sub_len = self.tokenizer.padded_tokens_and_len(
                self.tokenizer.encode(prompt), self.text_encoder.max_text_len
            )
            tokens.append(sub_tokens)
            lens.append(sub_len)
        return (
            torch.tensor(tokens).to(dtype=torch.long, device=self.device),
            torch.tensor(lens).to(dtype=torch.long, device=self.device),
        )

    def encode_prompts_others(self, prompts_others: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        list_tokens = []
        list_lens = []
        for prompts in prompts_others:
            tokens = []
            lens = []
            for prompt in prompts:
                sub_tokens, sub_len = self.tokenizer.padded_tokens_and_len(
                    self.tokenizer.encode(prompt), self.text_encoder.max_text_len
                )
                tokens.append(sub_tokens)
                lens.append(sub_len)
            list_tokens.append(tokens)
            list_lens.append(lens)
        return (
            torch.tensor(list_tokens).to(dtype=torch.long, device=self.device),
            torch.tensor(list_lens).to(dtype=torch.long, device=self.device),
        )

    def text_embeddings(self, prompts: List[str]) -> torch.Tensor:
        tokens, lens = self.encode_prompts(prompts)
        z_t = self.text_encoder(tokens, lens)
        return z_t / (torch.linalg.norm(z_t, dim=-1, keepdim=True) + 1e-12)

    def text_embeddings_others(self, prompts_others: List[List[str]]) -> torch.Tensor:
        n = len(prompts_others)
        z_t_others = []
        for i in range(n):
            tokens, lens = self.encode_prompts(prompts_others[i])
            z_t = self.text_encoder(tokens, lens)
            z_t_others.append(z_t)
        z_t_others = torch.stack(z_t_others)
        z_t_others_norm = z_t_others / (torch.linalg.norm(z_t_others, dim=-1, keepdim=True) + 1e-12)
        z_t_others_norm_sum = torch.sum(z_t_others_norm, dim=-2)
        return z_t_others_norm_sum

    def image_embeddings(self, images: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z_i = self.image_encoder((images + 1) * 127.5, t)
        return z_i / (torch.linalg.norm(z_i, dim=-1, keepdim=True) + 1e-12)

    def cond_fn(self, prompts: List[str], grad_scale: float, prompts_others: List[List[str]]) -> Callable[..., torch.Tensor]:
        with torch.no_grad():
            z_t = self.text_embeddings(prompts)
            z_t_others = self.text_embeddings_others(prompts_others)


        def cond_fn(x, t, grad_scale=grad_scale, osc=0.9, **kwargs):
            with torch.enable_grad():
                x_var = x.detach().requires_grad_(True)
                z_i = self.image_embeddings(x_var, t)
                z_t_centroid = z_t * osc + z_t_others * (1-osc)/4
                loss = torch.exp(self.logit_scale) * (z_t_centroid * z_i).sum()
                grad = torch.autograd.grad(loss, x_var)[0].detach()
            return grad * grad_scale

        return cond_fn

@attr.s
class CLIPModelContrastive:
    config: Dict[str, Any] = attr.ib()
    text_encoder: nn.Module = attr.ib()
    image_encoder: nn.Module = attr.ib()
    logit_scale: torch.Tensor = attr.ib()
    device: torch.device = attr.ib()
    tokenizer: SimpleTokenizer = attr.ib()

    def encode_prompts(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = []
        lens = []
        for prompt in prompts:
            sub_tokens, sub_len = self.tokenizer.padded_tokens_and_len(
                self.tokenizer.encode(prompt), self.text_encoder.max_text_len
            )
            tokens.append(sub_tokens)
            lens.append(sub_len)
        return (
            torch.tensor(tokens).to(dtype=torch.long, device=self.device),
            torch.tensor(lens).to(dtype=torch.long, device=self.device),
        )

    def text_embeddings(self, prompts: List[str]) -> torch.Tensor:
        tokens, lens = self.encode_prompts(prompts)
        z_t = self.text_encoder(tokens, lens)
        return z_t / (torch.linalg.norm(z_t, dim=-1, keepdim=True) + 1e-12)

    def image_embeddings(self, images: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z_i = self.image_encoder((images + 1) * 127.5, t)
        return z_i / (torch.linalg.norm(z_i, dim=-1, keepdim=True) + 1e-12)

    def cond_fn(self, prompts: List[str], grad_scale: float, z_t_ext, eps) -> Callable[..., torch.Tensor]:
        with torch.no_grad():
            z_t = self.text_embeddings(prompts)
            self.s = infodeg_calculation2(z_t, z_t_ext)
            # self.s = torch.cat([self.s_prim, self.s_others])
            self.eps = eps


        def cond_fn(x, t, grad_scale=grad_scale, **kwargs):
            with torch.enable_grad():
                x_var = x.detach().requires_grad_(True)
                z_i = self.image_embeddings(x_var, t)
                # print(self.s_prim)
                # print(torch.sum(self.s_others, dim=1))
                # z_t_comb = z_t * self.s_prim + torch.matmul(self.s_others, z_t_ext)
                # z_t_comb = torch.cat([z_t, z_t_ext])
                # print(z_t_ext.shape)
                # print(z_t_comb.shape)
                # exit(0)
                z_t_comb = z_t_ext
                logits = similarity_match(z_i, z_t_comb.detach())
                logits_t = similarity_match_instance2instance(z_i, z_t.detach())
                logits_cat = torch.cat([torch.unsqueeze(logits_t, dim=1), logits], dim=1)

                # loss = torch.exp(self.logit_scale) * (z_t_comb * z_i).sum()
                log_probs = F.log_softmax(logits_cat, dim = -1)
                # print(log_probs.shape)
                loss = (self.s * log_probs).sum(-1)
                grad = torch.autograd.grad(loss.sum(), x_var)[0].detach()
            self.augment_s(self.eps)
            return grad * grad_scale

        return cond_fn

    def augment_s(self, epsilon=0.94):
        # print(self.s.shape)
        # exit(0)
        delta_s = (1 - epsilon) * self.s[:, 1:]
        self.s[:, 1:]= self.s[:, 1:] - delta_s
        sum_delta = torch.sum(delta_s, dim = 1)
        self.s[:, 0] += sum_delta



@attr.s
class CLIPModelRobustExtCap:
    config: Dict[str, Any] = attr.ib()
    text_encoder: nn.Module = attr.ib()
    image_encoder: nn.Module = attr.ib()
    logit_scale: torch.Tensor = attr.ib()
    device: torch.device = attr.ib()
    tokenizer: SimpleTokenizer = attr.ib()

    def encode_prompts(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = []
        lens = []
        for prompt in prompts:
            sub_tokens, sub_len = self.tokenizer.padded_tokens_and_len(
                self.tokenizer.encode(prompt), self.text_encoder.max_text_len
            )
            tokens.append(sub_tokens)
            lens.append(sub_len)
        return (
            torch.tensor(tokens).to(dtype=torch.long, device=self.device),
            torch.tensor(lens).to(dtype=torch.long, device=self.device),
        )

    def text_embeddings(self, prompts: List[str]) -> torch.Tensor:
        tokens, lens = self.encode_prompts(prompts)
        z_t = self.text_encoder(tokens, lens)
        return z_t / (torch.linalg.norm(z_t, dim=-1, keepdim=True) + 1e-12)

    def image_embeddings(self, images: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z_i = self.image_encoder((images + 1) * 127.5, t)
        return z_i / (torch.linalg.norm(z_i, dim=-1, keepdim=True) + 1e-12)

    def cond_fn(self, prompts: List[str], grad_scale: float, z_t_ext, eps) -> Callable[..., torch.Tensor]:
        with torch.no_grad():
            z_t = self.text_embeddings(prompts)
            self.s_prim, self.s_others = infodeg_calculation(z_t, z_t_ext)
            self.eps = eps


        def cond_fn(x, t, grad_scale=grad_scale, **kwargs):

            with torch.enable_grad():
                x_var = x.detach().requires_grad_(True)
                z_i = self.image_embeddings(x_var, t)
                # print(self.s_prim)
                # print(torch.sum(self.s_others, dim=1))
                z_t_comb = z_t * self.s_prim + torch.matmul(self.s_others, z_t_ext)
                loss = torch.exp(self.logit_scale) * (z_t_comb * z_i).sum()
                grad = torch.autograd.grad(loss, x_var)[0].detach()
            self.augment_s(self.eps)
            return grad * grad_scale

        return cond_fn

    def augment_s(self, epsilon=0.94):
        delta_s = (1 - epsilon) * self.s_others
        self.s_others = self.s_others - delta_s
        sum_delta = torch.unsqueeze(torch.sum(delta_s, dim = 1), dim = 1)
        self.s_prim += sum_delta


def infodeg_calculation(z_t: torch.Tensor, z_t_ext: torch.Tensor):
    sim_matrix_wo_z_t = torch.matmul(z_t, torch.transpose(z_t_ext, 1, 0))
    matrix_ones = torch.ones((z_t.shape[0], 1), device=sim_matrix_wo_z_t.get_device())
    sim_matrix = torch.cat((matrix_ones, sim_matrix_wo_z_t), dim=1)
    sim_matrix_sum = torch.unsqueeze(torch.sum(sim_matrix, dim = 1), dim=1)
    sim_matrix = sim_matrix/sim_matrix_sum
    sim_prim = torch.unsqueeze(sim_matrix[:, 0], dim = 1)
    sim_others = sim_matrix[:, 1:]
    return sim_prim, sim_others

def infodeg_calculation2(z_t: torch.Tensor, z_t_ext: torch.Tensor):
    sim_matrix_wo_z_t = torch.matmul(z_t, torch.transpose(z_t_ext, 1, 0))
    matrix_ones = torch.ones((z_t.shape[0], 1), device=sim_matrix_wo_z_t.get_device())
    sim_matrix = torch.cat((matrix_ones, sim_matrix_wo_z_t), dim=1)
    sim_matrix_sum = torch.unsqueeze(torch.sum(sim_matrix, dim = 1), dim=1)
    sim_matrix = sim_matrix/sim_matrix_sum
    return sim_matrix


def create_clip_model(
    config_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    tokenizer: Optional[SimpleTokenizer] = None,
) -> CLIPModel:
    if config_path is None:
        config_path = default_config_path()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if tokenizer is None:
        tokenizer = SimpleTokenizer()

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    text_encoder = TextEncoder(
        n_bpe_vocab=config["n_vocab"],
        max_text_len=config["max_text_len"],
        n_embd=config["n_embd"],
        n_head=config["n_head_text"],
        n_xf_blocks=config["n_xf_blocks_text"],
        n_head_state=config["n_head_state_text"],
        device=device,
    )

    image_encoder = ImageEncoder(
        image_size=config["image_size"],
        patch_size=config["patch_size"],
        n_embd=config["n_embd"],
        n_head=config["n_head_image"],
        n_xf_blocks=config["n_xf_blocks_image"],
        n_head_state=config["n_head_state_image"],
        n_timestep=config["n_timesteps"],
        device=device,
    )

    logit_scale = torch.tensor(
        np.log(config["logit_scale"]),
        dtype=torch.float32,
        device=device,
        requires_grad=False,
    )

    return CLIPModel(
        config=config,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        logit_scale=logit_scale,
        device=device,
        tokenizer=tokenizer,
    )







def create_clip_model_robust(
    config_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    tokenizer: Optional[SimpleTokenizer] = None,
) -> CLIPModelRobust:
    if config_path is None:
        config_path = default_config_path()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if tokenizer is None:
        tokenizer = SimpleTokenizer()

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    text_encoder = TextEncoder(
        n_bpe_vocab=config["n_vocab"],
        max_text_len=config["max_text_len"],
        n_embd=config["n_embd"],
        n_head=config["n_head_text"],
        n_xf_blocks=config["n_xf_blocks_text"],
        n_head_state=config["n_head_state_text"],
        device=device,
    )

    image_encoder = ImageEncoder(
        image_size=config["image_size"],
        patch_size=config["patch_size"],
        n_embd=config["n_embd"],
        n_head=config["n_head_image"],
        n_xf_blocks=config["n_xf_blocks_image"],
        n_head_state=config["n_head_state_image"],
        n_timestep=config["n_timesteps"],
        device=device,
    )

    logit_scale = torch.tensor(
        np.log(config["logit_scale"]),
        dtype=torch.float32,
        device=device,
        requires_grad=False,
    )

    return CLIPModelRobust(
        config=config,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        logit_scale=logit_scale,
        device=device,
        tokenizer=tokenizer,
    )

def similarity_match(pred, target, mean=False):
    pred_norm = torch.nn.functional.normalize(pred, dim=1)
    target_norm = torch.nn.functional.normalize(target, dim=1)
    # cosine_sim = (pred_norm * target_norm).sum(dim=1)
    cosine_sim = torch.matmul(pred_norm, target_norm.T)
    loss = cosine_sim
    if mean:
        loss = loss.sum()
    return loss

def similarity_match_instance2instance(pred, target, mean=False):
    pred_norm = torch.nn.functional.normalize(pred, dim=1)
    target_norm = torch.nn.functional.normalize(target, dim=1)
    cosine_sim = (pred_norm * target_norm).sum(dim=1)
    # cosine_sim = torch.matmul(pred_norm, target_norm.T)
    loss = cosine_sim
    if mean:
        loss = loss.sum()
    return loss


def create_clip_model_robust_extcap(
    config_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    tokenizer: Optional[SimpleTokenizer] = None,
) -> CLIPModelRobustExtCap:
    if config_path is None:
        config_path = default_config_path()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if tokenizer is None:
        tokenizer = SimpleTokenizer()

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    text_encoder = TextEncoder(
        n_bpe_vocab=config["n_vocab"],
        max_text_len=config["max_text_len"],
        n_embd=config["n_embd"],
        n_head=config["n_head_text"],
        n_xf_blocks=config["n_xf_blocks_text"],
        n_head_state=config["n_head_state_text"],
        device=device,
    )

    image_encoder = ImageEncoder(
        image_size=config["image_size"],
        patch_size=config["patch_size"],
        n_embd=config["n_embd"],
        n_head=config["n_head_image"],
        n_xf_blocks=config["n_xf_blocks_image"],
        n_head_state=config["n_head_state_image"],
        n_timestep=config["n_timesteps"],
        device=device,
    )

    logit_scale = torch.tensor(
        np.log(config["logit_scale"]),
        dtype=torch.float32,
        device=device,
        requires_grad=False,
    )

    return CLIPModelRobustExtCap(
        config=config,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        logit_scale=logit_scale,
        device=device,
        tokenizer=tokenizer,
    )


def create_clip_model_contrastive(
    config_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    tokenizer: Optional[SimpleTokenizer] = None,
) -> CLIPModelContrastive:
    if config_path is None:
        config_path = default_config_path()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if tokenizer is None:
        tokenizer = SimpleTokenizer()

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    text_encoder = TextEncoder(
        n_bpe_vocab=config["n_vocab"],
        max_text_len=config["max_text_len"],
        n_embd=config["n_embd"],
        n_head=config["n_head_text"],
        n_xf_blocks=config["n_xf_blocks_text"],
        n_head_state=config["n_head_state_text"],
        device=device,
    )

    image_encoder = ImageEncoder(
        image_size=config["image_size"],
        patch_size=config["patch_size"],
        n_embd=config["n_embd"],
        n_head=config["n_head_image"],
        n_xf_blocks=config["n_xf_blocks_image"],
        n_head_state=config["n_head_state_image"],
        n_timestep=config["n_timesteps"],
        device=device,
    )

    logit_scale = torch.tensor(
        np.log(config["logit_scale"]),
        dtype=torch.float32,
        device=device,
        requires_grad=False,
    )

    return CLIPModelContrastive(
        config=config,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        logit_scale=logit_scale,
        device=device,
        tokenizer=tokenizer,
    )

def center_crop_arr(images, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    y_size = images.shape[2]
    x_size = images.shape[3]
    crop_y = (y_size - image_size) // 2
    crop_x = (x_size - image_size) // 2
    return images[:, :, crop_y : crop_y + image_size, crop_x : crop_x + image_size]

def custom_normalize(images, mean, std):
    # print(images.shape)
    # Check if the input tensor has the same number of channels as the mean and std
    if images.size(1) != len(mean) or images.size(1) != len(std):
        raise ValueError("The number of channels in the input tensor must match the length of mean and std.")
    images = images.to(torch.float)
    # Normalize the tensor
    for c in range(images.size(1)):
        images[:, c, :, :] = (images[:, c, :, :] - mean[c]) / std[c]

    return images



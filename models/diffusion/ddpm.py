# models/ddpm.py
import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_beta_scheduler(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """线性beta调度"""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_scheduler(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """余弦beta调度"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def sigmoid_beta_scheduler(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """Sigmoid beta调度"""
    betas = torch.linspace(-6, 6, timesteps)
    betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    return betas


def get_beta_scheduler(schedule_type: str, timesteps: int, **kwargs) -> torch.Tensor:
    """获取beta调度"""
    if schedule_type == "linear":
        return linear_beta_scheduler(timesteps, **kwargs)
    elif schedule_type == "cosine":
        return cosine_beta_scheduler(timesteps)
    elif schedule_type == "sigmoid":
        return sigmoid_beta_scheduler(timesteps, **kwargs)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule_type}")


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    """
    高斯扩散模型
    """
    def __init__(self, denoiser: nn.Module, **kwargs):
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = kwargs["timesteps"]
        self.prediction_type = kwargs["prediction_type"]

        betas = get_beta_scheduler(kwargs["beta_scheduler"], self.timesteps, beta_start=kwargs["beta_start"], beta_end=kwargs["beta_end"])
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance', torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None: noise = torch.randn_like(x_0)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )
    
    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def predict_eps_from_x0(self, x_t: torch.Tensor, t: torch.Tensor, x_0: torch.Tensor) -> torch.Tensor:
        return (
            (x_t - extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_0) /
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        )
        
    def get_v(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_alphas_cumprod, t, x_0.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * x_0
        )
    
    def predict_x0_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def model_predictions(self, x_t: torch.Tensor, t: torch.Tensor, local_cond: torch.Tensor, global_cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            local_cond: Additive原型条件
            global_cond: AdaGN条件
        """
        # UNet forward call
        model_output = self.denoiser(x_t, t, local_cond, global_cond)
        
        if self.prediction_type == "eps":
            pred_noise = model_output
            pred_x0 = self.predict_x0_from_eps(x_t, t, pred_noise)
        elif self.prediction_type == "v":
            v = model_output
            pred_x0 = self.predict_x0_from_v(x_t, t, v)
            pred_noise = self.predict_eps_from_x0(x_t, t, pred_x0)
        elif self.prediction_type == "x":
            pred_x0 = model_output
            pred_noise = self.predict_eps_from_x0(x_t, t, pred_x0)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        return pred_noise, pred_x0
    
    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, local_cond: torch.Tensor, global_cond: torch.Tensor):
        pred_noise, pred_x0 = self.model_predictions(x_t, t, local_cond, global_cond)
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * pred_x0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract(self.posterior_log_variance, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance, pred_x0

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, local_cond: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        b = x_t.shape[0]
        model_mean, _, model_log_variance, _ = self.p_mean_variance(x_t, t, local_cond, global_cond)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(b, *([1] * (len(x_t.shape) - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape: Tuple[int, ...], local_cond: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        device = self.betas.device
        b = shape[0]
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch, local_cond, global_cond)
        return x

    @torch.no_grad()
    def ddim_sample(self, shape: Tuple[int, ...], local_cond: torch.Tensor, global_cond: torch.Tensor, ddim_steps: int = 50, eta: float = 0.0) -> torch.Tensor:
        device = self.betas.device
        b = shape[0]
        times = torch.linspace(-1, self.timesteps - 1, steps=ddim_steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        x = torch.randn(shape, device=device)
        
        for time, time_prev in time_pairs:
            t = torch.full((b,), time, device=device, dtype=torch.long)
            pred_noise, pred_x0 = self.model_predictions(x, t, local_cond, global_cond)
            
            if time_prev < 0:
                x = pred_x0
                continue
            
            alpha = self.alphas_cumprod[time]
            alpha_prev = self.alphas_cumprod[time_prev]
            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))
            c = torch.sqrt(1 - alpha_prev - sigma ** 2)
            noise = torch.randn_like(x)
            x = torch.sqrt(alpha_prev) * pred_x0 + c * pred_noise + sigma * noise
        return x

    def compute_loss(self, x_0: torch.Tensor, t: torch.Tensor, local_cond: torch.Tensor, global_cond: torch.Tensor) -> Dict[str, torch.Tensor]:
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        
        # 传入双重条件
        model_output = self.denoiser(x_t, t, local_cond, global_cond)
        
        if self.prediction_type == "eps":
            target = noise
        elif self.prediction_type == "x":
            target = x_0
        elif self.prediction_type == "v":
            target = self.get_v(x_0, noise, t)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
            
        loss = F.mse_loss(model_output, target)
        return {'loss': loss}


    def forward(self, x_0: torch.Tensor, proto: torch.Tensor, cond: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算训练损失
        """
        # 随机采样时间步
        t = torch.randint(0, self.timesteps, (x_0.shape[0],), device=x_0.device).long()
        return self.compute_loss(x_0, t, proto, cond)

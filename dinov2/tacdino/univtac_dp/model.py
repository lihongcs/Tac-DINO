"""Action diffusion head conditioned on vision, tactile, and joint state."""
import torch
from torch import nn
from diffusers import DDPMScheduler, DDIMScheduler
from ._vendor.conditional_unet1d import ConditionalUnet1D


class FrozenFeatureDP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Four fixed slots: head, wrist, left tactile, right tactile.
        # The wrist slot is zero for tasks that use a single RGB camera.
        width = cfg["observation_steps"] * (4 * cfg["feature_dim"] + cfg["action_dim"])
        self.fusion = nn.Sequential(nn.Linear(width, cfg["condition_dim"]), nn.Mish(),
                                    nn.Linear(cfg["condition_dim"], cfg["condition_dim"]))
        self.unet = ConditionalUnet1D(input_dim=cfg["action_dim"],
            global_cond_dim=cfg["condition_dim"], down_dims=cfg["unet_down_dims"],
            diffusion_step_embed_dim=128, cond_predict_scale=True)
        options = dict(num_train_timesteps=cfg["diffusion_train_steps"],
                       beta_schedule=cfg["noise_schedule"],
                       prediction_type=cfg["prediction_type"], clip_sample=False)
        self.noise_scheduler = DDPMScheduler(**options)
        self.inference_scheduler = DDIMScheduler(**options)

    def condition(self, batch):
        features = batch["features"]
        x = torch.cat([features.flatten(1), batch["state"].flatten(1)], dim=1)
        return self.fusion(x)

    def loss(self, batch, generator):
        action = batch["action"]
        noise = torch.randn(action.shape, device=action.device, generator=generator)
        t = torch.randint(self.cfg["diffusion_train_steps"], (len(action),),
                          device=action.device, generator=generator)
        noisy = self.noise_scheduler.add_noise(action, noise, t)
        predicted = self.unet(noisy, t, global_cond=self.condition(batch))
        error = (predicted - noise).square() * batch["valid"].unsqueeze(-1)
        count = batch["valid"].sum() * self.cfg["action_dim"]
        return error.sum() / count.clamp_min(1)

    @torch.no_grad()
    def predict(self, batch, generator):
        cond = self.condition(batch)
        sample = torch.randn((len(cond), self.cfg["prediction_horizon"], self.cfg["action_dim"]),
                              device=cond.device, generator=generator)
        self.inference_scheduler.set_timesteps(self.cfg["diffusion_inference_steps"], device=cond.device)
        for t in self.inference_scheduler.timesteps:
            predicted = self.unet(sample, t, global_cond=cond)
            sample = self.inference_scheduler.step(predicted, t, sample, eta=0.0).prev_sample
        return sample

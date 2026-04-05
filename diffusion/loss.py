import torch
from diffusion.scheduler import NoiseScheduler

scheduler = NoiseScheduler()

def diffusion_loss(model, x0, t):
    noise = torch.randn_like(x0)
    xt = scheduler.add_noise(x0, noise, t)

    pred = model(xt, t)

    # 🔥 match shapes for DiT (token output)
    if pred.dim() == 3:
        pred = pred.mean(dim=1).view_as(x0)

    return ((noise - pred) ** 2).mean()

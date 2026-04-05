import torch

class NoiseScheduler:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps)

        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, noise, t):
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1).to(x0.device)
        return (
            torch.sqrt(alpha_bar_t) * x0 +
            torch.sqrt(1 - alpha_bar_t) * noise
        )

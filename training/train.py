import torch
import wandb
from models.unet import SimpleUNet
from models.dit import SimpleDiT
from diffusion.loss import diffusion_loss

def train(config):
    device = "cuda"

    #  init wandb
    if config["logging"]["use_wandb"]:
        wandb.init(project=config["logging"]["project"], config=config)

    #  model selection
    if config["model"] == "unet":
        model = SimpleUNet().to(device)
    else:
        model = SimpleDiT().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

    for epoch in range(config["training"]["epochs"]):
        for step in range(100):  # mock loop
            x = torch.randn(config["training"]["batch_size"], 3, 32, 32).to(device)
            t = torch.randint(0, 1000, (x.size(0),), device=device)

            loss = diffusion_loss(model, x, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log metrics
            if config["logging"]["use_wandb"]:
                wandb.log({
                    "loss": loss.item(),
                    "epoch": epoch
                })

        print(f"Epoch {epoch}: loss={loss.item():.4f}")

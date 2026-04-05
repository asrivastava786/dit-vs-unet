from torchmetrics.image.fid import FrechetInceptionDistance

def compute_fid(real, fake):
    fid = FrechetInceptionDistance(feature=2048)
    fid.update(real, real=True)
    fid.update(fake, real=False)
    return fid.compute()

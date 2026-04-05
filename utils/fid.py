from torchmetrics.image.fid import FrechetInceptionDistance

class FIDTracker:
    def __init__(self):
        self.fid = FrechetInceptionDistance(feature=2048)

    def update(self, real, fake):
        self.fid.update(real, real=True)
        self.fid.update(fake, real=False)

    def compute(self):
        return self.fid.compute()

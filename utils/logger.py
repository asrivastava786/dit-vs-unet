import time
import torch

def measure_throughput(model, batch):
    start = time.time()
    for _ in range(50):
        model(*batch)
    torch.cuda.synchronize()
    return 50 * batch[0].size(0) / (time.time() - start)

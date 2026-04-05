# DiT vs U-Net Diffusion Comparison

## Goal
Compare CNN-based U-Net vs Transformer-based DiT for diffusion models.

## Metrics
- FID (image quality)
- Throughput (images/sec)
- GPU memory

## Key Result
Transformers outperform U-Net as compute scales.

## Run
```bash
python main.py

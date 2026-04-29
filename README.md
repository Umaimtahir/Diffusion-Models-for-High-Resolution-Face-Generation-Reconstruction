# Diffusion-Models-for-High-Resolution-Face-Generation-Reconstruction


End-to-end implementation of a Denoising Diffusion Probabilistic Model (DDPM) from scratch using base PyTorch layers.
No pretrained diffusion pipelines were used.

Highlights
5-stage DDPM pipeline:
Data preprocessing + integrity checks
Forward diffusion (noising)
U-Net denoiser with residual blocks
Timestep conditioning via sinusoidal embeddings
Mixed-precision training + EMA + checkpointing
Face generation from pure Gaussian noise
Target image reconstruction through reverse diffusion
Quantitative evaluation with PSNR and SSIM
Gradio app for interactive sampling
Architecture
U-Net Channel Path
3 -> 64 -> 128 -> 256 -> 128 -> 64 -> 3

Diffusion Setup
Timesteps: T = 300
Beta schedule: linear from 1e-4 to 0.02
Objective: noise prediction (\epsilon-prediction) with MSE loss
Training
Optimizer: AdamW
Learning rate: 2e-4
AMP + GradScaler
Gradient clipping: 1.0
EMA model tracking
Cosine LR schedule
Project Structure
diffusion-model-gen-a4.ipynb — complete notebook implementation
ddpm_outputs/ — checkpoints and generated outputs (runtime-generated)
gradio section in notebook — app deployment cell
Dataset
Recommended dataset for this project:

FFHQ (high-quality faces):
https://www.kaggle.com/datasets/greatgamedota/ffhq-face-data-set
How to Run (Kaggle)
Create a new Kaggle notebook with GPU (T4 / dual T4 if available).
Add FFHQ dataset as input.
Run notebook cells in order:
setup + config
data scan + verify
model definition
training
sampling / reconstruction / metrics
gradio app
For best sampling quality, use:
best checkpoint (ddpm_best.pt)
EMA-enabled sampling
Outputs
The notebook produces:

Forward diffusion visualization (multiple noising levels)
Reverse diffusion visualization (intermediate denoising steps)
At least 5 generated images from random noise
Reconstruction panel: Target vs Noisy vs Reconstructed
PSNR and SSIM metrics
Loss vs Epoch training curve
Notes
Generation quality depends heavily on:
checkpoint quality,
EMA usage,
and sampling configuration (stochastic vs deterministic).
Reconstruction and generation should be evaluated separately.
Future Improvements
DDIM / accelerated sampling
256x256 training
Conditional diffusion (class/style conditioning)
Attention blocks at more resolutions

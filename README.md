# Diffusion-Models-for-High-Resolution-Face-Generation-Reconstruction
🧠 Denoising Diffusion Probabilistic Model (DDPM) — From Scratch (PyTorch)

End-to-end implementation of a Denoising Diffusion Probabilistic Model (DDPM) built entirely from scratch using base PyTorch layers.
No pretrained diffusion pipelines or external frameworks were used.

🚀 Highlights
✅ Complete 5-stage DDPM pipeline
✅ Data preprocessing with integrity checks
✅ Forward diffusion (progressive noising)
✅ U-Net denoiser with residual blocks
✅ Timestep conditioning via sinusoidal embeddings
✅ Mixed precision training (AMP + GradScaler)
✅ EMA (Exponential Moving Average) for stable sampling
✅ Checkpointing system
✅ Face generation from pure Gaussian noise
✅ Image reconstruction via reverse diffusion
✅ Quantitative evaluation using PSNR and SSIM
✅ Interactive Gradio app for sampling
🏗️ Architecture
U-Net Channel Flow
3 → 64 → 128 → 256 → 128 → 64 → 3
Encoder-decoder structure with skip connections
Residual blocks for stable training
Time embeddings injected at multiple levels
⚙️ Diffusion Setup
Timesteps (T): 300
Beta Schedule: Linear (1e-4 → 0.02)
Training Objective:
Noise prediction (ε-prediction)
Loss: Mean Squared Error (MSE)
🏋️ Training Details
Optimizer: AdamW
Learning Rate: 2e-4
Scheduler: Cosine LR decay
Mixed Precision: AMP + GradScaler
Gradient Clipping: 1.0
EMA: Enabled for better sampling stability
📂 Project Structure
.
├── diffusion-model-gen-a4.ipynb   # Full implementation notebook
├── ddpm_outputs/                 # Generated outputs + checkpoints
│   ├── ddpm_best.pt
│   └── samples/
└── README.md
📊 Dataset

Recommended dataset:

👉 FFHQ (High-Quality Human Faces)
https://www.kaggle.com/datasets/greatgamedota/ffhq-face-data-set

▶️ How to Run (Kaggle)
Create a new Kaggle Notebook with GPU (T4 or dual T4 recommended)
Add the FFHQ dataset as input
Run notebook cells in order:
Setup & configuration
Data scanning & verification
Model definition
Training
Sampling / reconstruction / metrics
Gradio app
🎨 Outputs

The notebook generates:

📉 Forward diffusion visualization (noise progression)
🔄 Reverse diffusion visualization (denoising steps)
🖼️ At least 5 generated images from random noise
🧪 Reconstruction panel:
Original vs Noisy vs Reconstructed
📏 Metrics:
PSNR (Peak Signal-to-Noise Ratio)
SSIM (Structural Similarity Index)
📈 Training curve (Loss vs Epoch)
🧪 Sampling Tips

For best results:

Use best checkpoint (ddpm_best.pt)
Enable EMA model for sampling
Experiment with:
Stochastic vs deterministic sampling


⚠️ Notes
Generation quality depends heavily on:
Checkpoint quality
EMA usage
Sampling configuration
Always evaluate:
Generation quality and Reconstruction quality separately
🔮 Future Improvements
🚀 DDIM / accelerated sampling
🖼️ Higher resolution training (256×256)
🎯 Conditional diffusion (class/style conditioning)
🧠 Attention blocks at multiple resolutions
📌 Summary

This project demonstrates a fully custom diffusion model pipeline, covering everything from theory to deployment, making it a strong foundation for research or advanced generative modeling work.

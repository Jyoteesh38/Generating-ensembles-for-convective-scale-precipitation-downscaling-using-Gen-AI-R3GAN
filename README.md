# R3GAN: Generating Ensembles for Convective-Scale Precipitation Downscaling Using Generative AI

![Training framework](docs/GAN_training.png)

This repository implements **R3GAN**, a distributed generative adversarial network (GAN) for the prediction of convective-scale precipitation (4.4 km horizontal resolution) using ERA5 atmospheric variables as input and Australian regional reanalysis (BARRA-C2) precipitation as target.

---

## 📌 Project Highlights
- 📈 **Trained on massive datasets**: Over **359,000 training samples (each training sample has 71 images)** (1980–2020) and **17,520 test samples** (2021–2022) used to ensure robust learning and generalization.
- 🎯 High-resolution precipitation (4.4 km) ensemble generation using coarse ERA5 reanalysis inputs
- 🌍 Spatial conditioning with static variables (e.g., orography, land-sea mask)
- 🧠 Residual U-Net generator with multi-head attention
- 🧪 Relativistic GAN loss + zero-centered gradient penalties for stable training
- 🚀 Distributed training with Horovod + TensorFlow on multi-GPU systems (80 NVIDIA V100 GPUs)
- 🧬 **R3GAN ensemble framework**: Combines **noise seeds**, **variance**, and **model epoch sampling** to enhance diversity in high-resolution precipitation generation.
- 🔁 **Multi-source ensemble generation**: Improves **calibration**, **diversity**, and **uncertainty quantification** for convective precipitation events.

---

## 🧠 Model Summary

### Generator (R3GAN):
- Residual U-Net with skip connections
- Multi-head self-attention at bottleneck
- Input: ERA5 patch, static variables, Gaussian noise
- Output: High-resolution precipitation ensemble member

### Discriminator:
- Convolutional classifier
- Relativistic comparison of real vs. generated patches and dual-sided zero-centered gradient penalties

- Losses include:
  - Mean Square Error (MSE)
  - Relativistic GAN Loss
  - Zero-centered Gradient Penalties (R1/R2)

### 📦 Data

This project uses high-volume climate and observation datasets in `.zarr` format:

- **`ERA5_ML_data.zarr`** — ERA5 atmospheric reanalysis variables from 1980–2022 (coarse resolution)
- **`B2C_Pr_data_2k25.zarr`** — BARRA-C2 target high-resolution precipitation (4.4 km) data
- **`B_C2_static.zarr`** — Static predictors including land-sea mask and orography

> 📁 These datasets are not included in this repository due to their large size.
>  
> 📬 For access or data instructions, please contact the author.
>
### 🚀 Training

This model is designed for large-scale distributed training using Horovod.

To train using **80 GPUs**, launch with:

```bash
horovodrun -np 80 -H localhost:80 python R3GAN/train_r3gan.py
```
Ensure that the system has proper NCCL support, GPU settings, and TF_DISABLE_NVTX_RANGES / TF_CPP_MIN_LOG_LEVEL environment variables configured (as handled in utils.py).

train_r3gan.py is the main training script for the R3GAN model. This script trains a Relativistic GAN model (R3GAN) for convective-scale precipitation downscaling using Horovod-based multi-GPU distributed training. It handles data loading, preprocessing, model building, and training loops for the generator and discriminator, with validation and checkpointing.

models.py defines the architecture of the generator and discriminator networks for the R3GAN model using U-Net blocks, attention layers, and convolutional modules. It includes reusable components such as residual blocks, padding layers, and attention modules tailored for image-like spatiotemporal inputs.

losses.py implements custom loss functions for the GAN model, including relativistic adversarial loss and zero-centered gradient penalties for stability. Includes both discriminator and generator loss functions designed to mitigate mode collapse and improve convergence.

utils.py provides utility functions for Horovod setup, GPU configuration, data shuffling, splitting, and loading in a parallelized training environment. Helps ensure consistent data handling and efficient batching across distributed training workers.

Inference_metrics.py implements evaluation metrics for ensemble precipitation prediction, including normalized rank histograms, ROC curves, and frequency-based power spectral density (PSD) analysis.

model_predictions.py generates ensemble and deterministic predictions using trained R3GAN and UNET models on ERA5 inputs and BARRA-C2 static fields. Includes support for multi-model, multi-seed ensemble generation using Dask for parallel computation, and denormalization of outputs for post-processing.

loss_curve.py parses training history and visualizes generator and discriminator losses over epochs. Helps monitor model convergence and identify optimal training checkpoints.

Inference_plots.py performs a comprehensive evaluation of model outputs, including statistical distributions, spatial plots, PSD, ROC, CRPS, and Brier scores. Generates comparison plots across ERA5, BARRA-C2, UNET, and R3GAN predictions for ensemble and deterministic forecasts.

## 🔍 Inference

Run the script [`scripts/Inference_plots.py`](scripts/Inference_plots.py) to generate **high-resolution precipitation ensemble predictions** using trained R3GAN generator models.

### This script demonstrates:

- 🔄 **Loading trained generator models** from saved `.h5` checkpoints
- 📥 **Preprocessing and loading ERA5 reanalysis inputs** and static variables from `.zarr` format
- 🎲 **Injecting random Gaussian noise seeds** to simulate ensemble diversity
- 🎯 **Generating downscaled precipitation fields** at convective resolution (4.4 km)
- 🖼️ **Saving and visualizing output fields** for selected timesteps and ensemble members
- 📊 **Calculating ensemble statistics** across generated members

> ✅ Designed to support batch-wise inference over large datasets and multiple noise seeds for uncertainty quantification.



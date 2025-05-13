# R3GAN: Generating Ensembles for Convective-Scale Precipitation Downscaling Using Generative Machine Learning

![Generator Architecture](docs/Arch_R3GAN8_combined_plot.png)
![Training framework](docs/GAN_training.png)

This repository implements **R3GAN**, a distributed generative adversarial network (GAN) for downscaling of convective-scale precipitation using global ERA5 reanalysis and Australian regional reanalysis data.

---

## 📌 Project Highlights
- 📈 **Trained on massive datasets**: Over **359,000 training samples** (1980–2020) and **17,520 test samples** (2021–2022) used to ensure robust learning and generalization.
- 🎯 High-resolution precipitation ensemble generation using coarse ERA5 reanalysis inputs
- 🌍 Spatial conditioning with static variables (e.g., orography, land-sea mask)
- 🧠 Residual U-Net generator with multi-head attention
- 🧪 Relativistic GAN loss + zero-centered gradient penalties for stable training
- 🚀 Distributed training with Horovod + TensorFlow on multi-GPU systems (80 GPUs)
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

### 📦 Data

This project uses high-volume climate and observation datasets in `.zarr` format:

- **`ERA5_ML_data.zarr`** — ERA5 climate reanalysis variables from 1980–2022 (coarse resolution)
- **`B2C_Pr_data_2k25.zarr`** — BARRA-C2 target high-resolution precipitation data
- **`B_C2_static.zarr`** — Static predictors including land-sea mask and orography

> 📁 These datasets are not included in this repository due to their large size.
>  
> 📬 For access or data instructions, please contact the author.


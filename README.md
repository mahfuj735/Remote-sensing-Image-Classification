---
title: LiteRS-Mamba Remote Sensing Land Cover Classification
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: mit
short_description: Real-time 7-class satellite land cover classification across Bangladesh deltaic ecosystems.
---

# 🛰️ LiteRS-Mamba: Real-Time Multi-Platform Remote Sensing Land Cover Intelligence for Bangladesh

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Opset%2017-005ced?logo=onnx&logoColor=white)](https://onnx.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Dataset: MLC-BD](https://img.shields.io/badge/Dataset-MLC--BD%20(10%2C665%20Tiles)-blue)](https://www.kaggle.com/)

> **Master's / Academic Thesis Research Repository**  
> **Author:** Md. Mahfujul Karim Sheikh (Student ID: `CSE11220120735`)  
> **Project:** Multi-Platform Remote Sensing Land Cover Classification in Bangladesh: Benchmarking Deep Vision Paradigms, Proposing LiteRS-Mamba, and Test-Time Spectral Radiometric Adaptation.

---

## 🌟 Executive Summary

Remote sensing land-use and land-cover (LULC) classification in the dynamic deltaic terrain of Bangladesh is severely impaired by **cross-sensor radiometric discrepancies** across web-mapping satellite providers (ArcGIS Map, Bing Maps, Google Maps). Conventional deep learning models suffer catastrophic cross-platform domain collapse when evaluated across sensors.

This repository provides **LiteRS-Mamba**, an ultra-compact **$2.05\text{ M}$ parameter** architecture engineered specifically for low-power edge UAVs, survey drones, and satellite edge computing:
1. **Multi-Scale Stem (MS-Stem):** Dual $3\times3$ and $5\times5$ convolutional tokenizer resolving fine-grained riparian boundaries and parcel edges.
2. **4-Way Omnidirectional Selective State Space Engine (O4-SSM):** Scans 2D visual tokens along horizontal ($x$), vertical ($y$), and reverse pathways in parallel, with learnable softmax directional fusion.
3. **Test-Time Radiometric Adaptation (TTA-RHM):** Zero-shot cumulative distribution matching recovering $+45.65\%$ accuracy on raw Google Maps imagery without retraining.
4. **Token-Level SSM Grad-CAM:** Explainable AI framework offering physical topological verification and forensic failure diagnosis.
5. **Universal ONNX & INT8 Edge Deployment:** Serialized graph verified to $3.34 \times 10^{-6}$ mathematical parity, quantized to **$8.58\text{ MB}$** ($23.23\%$ storage reduction) with only $0.29\%$ accuracy delta ($97.43\% \rightarrow 97.14\%$) and $99.71\%$ decision agreement.

---

## 🗺️ The 7 Canonical Bangladesh Land Cover Classes

| Class Icon | Class Name | Bengali Meaning | Key Ecological & Physical Characteristics |
| :---: | :--- | :--- | :--- |
| 🌾 | **`Agricultural_Land`** | ফসলি জমি / কৃষি ভূমি | Alluvial crop fields, seasonal boro/aman paddy cultivation, silt floodplains. |
| 🏖️ | **`Beaches_Coastline`** | সমুদ্র সৈকত ও উপকূলরেখা | Sandy shores, intertidal mudflats in Cox's Bazar, Kuakata & Sundarbans delta. |
| 🏙️ | **`Built_up_Areas`** | শহরাঞ্চল ও পাকা স্থাপনা | High-density urban dwellings, impervious rooftops, asphalt roads, transport corridors. |
| 🌿 | **`Grasslands_Open_Land`** | ঘাসজমি ও চরাঞ্চল | Vegetated riverine chars, grazing ground, and riparian bars in Jamuna/Padma basin. |
| ⛰️ | **`Hills_Mountains`** | পাহাড় ও টিলা অঞ্চল | Tertiary folding ranges, tea gardens, tropical forest slopes in CHT & Sylhet. |
| 🌲 | **`Trees_Forests`** | গাছপালা ও বনাঞ্চল | Dense mangrove canopies of Sundarbans, Sal forests of Madhupur, social plantations. |
| 🌊 | **`Water_Bodies`** | নদী, খাল ও জলাশয় | Major braided rivers (Padma, Jamuna, Meghna), haors, beels, and coastal estuaries. |

---

## 📊 Comprehensive Experimental Benchmark Summary

### 1. In-Domain Classification vs. Deep Vision Baselines (ArcGIS Map)

| Model Architecture | Paradigm | Parameters (M) | In-Domain OA (%) | Macro-F1 (%) | Storage Footprint (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ResNet-50** | Classical CNN | 25.56 | 94.94% | 94.88% | 97.80 MB |
| **ConvNeXt-Tiny** | Modern ConvNet | 28.60 | 96.69% | 96.65% | 109.50 MB |
| **Swin-Transformer-Tiny** | Hierarchical ViT | 28.30 | 97.23% | 97.20% | 112.40 MB |
| **Vision-Mamba-Tiny (Vim)** | 1D Visual SSM | 5.50 | 96.22% | 96.18% | 22.10 MB |
| 🚀 **LiteRS-Mamba (Proposed FP32)** | **Omnidirectional SSM** | **2.05** | **97.19%** | **97.18%** | **8.25 MB** |
| ⚡ **LiteRS-Mamba (Quantized INT8)** | **Edge Quantized SSM** | **2.05** | **97.14%** | **97.10%** | **8.58 MB** |

> **Efficiency Takeaway:** LiteRS-Mamba matches the accuracy of Swin-Transformer ($97.19\%$ vs $97.23\%$) while requiring **$13.8\times$ fewer parameters** and **$13.1\times$ less storage**!

---

## 💻 Hardware Benchmarking & Pareto Frontier

```
   High Accuracy
        ^
 98% +  |           ★ LiteRS-Mamba (97.19%, 8.25 MB) [Pareto Optimal]
        |               ● Swin-T (97.23%, 112 MB)
 96% +  |       ● ConvNeXt-T (96.69%, 109 MB)
        |       ● Vision-Mamba (96.22%, 22 MB)
 94% +  |   ● ResNet-50 (94.94%, 98 MB)
        +---------------------------------------------------->
       0 MB           40 MB          80 MB         120 MB
                         Storage Footprint
```

* **CPU Throughput (ONNX Runtime):** $297.73\text{ ms}$ ($3.4\text{ FPS}$, a **$1.47\times$ acceleration** over native PyTorch CPU).
* **GPU Throughput (CUDA T4):** $155.90\text{ ms}$ ($6.4\text{ FPS}$).
* **Quantization Fidelity Audit ($N=350$ scenes):** Accuracy delta is merely **$0.29\%$** ($97.43\% \rightarrow 97.14\%$) with **$99.71\%$ decision agreement**.

---

## 🚀 Quickstart: Running the Web App Locally

### 1. Clone the Repository
```bash
git clone https://github.com/mahfuj735/Remote-sensing-Image-Classification.git
cd Remote-sensing-Image-Classification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Interactive Dashboard
```bash
python app.py
```
Open your browser at `http://localhost:7860`. You can drag-and-drop satellite scenes, view instant predictions, toggle TTA radiometric alignment, and inspect Grad-CAM attention heatmaps!

---

## 🌐 Deploying to Hugging Face Spaces (in 2 Minutes)

This repository is structured to run directly on Hugging Face Spaces without any modifications:

1. Create a free account at [huggingface.co](https://huggingface.co/).
2. Click **New Space** $\rightarrow$ Set Space Name (e.g. `liters-mamba-bd`).
3. Select **SDK: Gradio** and **Hardware: CPU Basic (Free)**.
4. Push this repository to Hugging Face:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/liters-mamba-bd
   git push space main
   ```
5. Your live web application will be active at `https://huggingface.co/spaces/YOUR_USERNAME/liters-mamba-bd`!

---

## 📦 Repository Structure

```
Remote-sensing-Image-Classification/
├── app.py                     # Main interactive Gradio application
├── requirements.txt           # Runtime dependencies
├── README.md                  # Hugging Face + GitHub documentation
├── checkpoints/               # Trained model weights (< 12 MB each)
│   ├── best_liters_mamba_ArcGIS_Map.pth  # Primary champion model (8.25 MB)
│   ├── best_liters_mamba_Bing_Maps.pth   # Bing Maps model
│   ├── best_liters_mamba_Google_Maps.pth # Google Maps model
│   ├── liters_mamba_fp32.onnx            # Serialized ONNX Opset 17
│   └── liters_mamba_int8.onnx            # Post-training INT8 quantized
├── sample_images/             # Test gallery tiles across all 7 LULC classes
│   ├── 01_Agricultural_Land.png
│   ├── 02_Beaches_Coastline.png
│   ├── 03_Built_up_Areas.png
│   ├── 04_Grasslands_Open_Land.png
│   ├── 05_Hills_Mountains.png
│   ├── 06_Trees_Forests.png
│   ├── 07_Water_Bodies.png
│   ├── 08_Google_Agricultural_Land_CrossPlatform.png
│   └── 09_Google_Water_Bodies_CrossPlatform.png
└── assets/                    # Figures, architecture diagrams, and charts
    ├── LiteRS_Mamba_Exact_Architecture.svg
    ├── fig_edge_pareto_efficiency_frontier.png
    ├── fig_edge_latency_fps_comparison.png
    └── fig_tta_visual_gallery_7classes.png
```

---

## 📜 Citation & Attribution

If you use this model or code in your research, please cite:

```bibtex
@mastersthesis{sheikh2026litersmamba,
  author       = {Md. Mahfujul Karim Sheikh},
  title        = {Multi-Platform Remote Sensing Land Cover Classification in Bangladesh: Benchmarking Deep Vision Paradigms, Proposing LiteRS-Mamba, and Test-Time Spectral Radiometric Adaptation},
  school       = {Department of Computer Science and Engineering},
  year         = {2026},
  note         = {ID: CSE11220120735}
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).

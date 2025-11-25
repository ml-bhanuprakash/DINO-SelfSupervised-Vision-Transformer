# DINO Self-Supervised Vision Learning — Engineering Implementation (Single Notebook)

This repository showcases a compact yet production-grade implementation of a **DINO-style self-supervised visual representation system** using **Vision Transformers (ViT)**.  
Everything — data pipeline, augmentation engine, transformer backbone, distillation logic, and attention visualizations — is implemented inside a single, clean IPython notebook.

---

## 🔹 Project Highlights (What This Implementation Demonstrates)

### **✔ Fully Custom DINO Data Pipeline (ImageNet-64)**
- Custom dataset loader for ImageNet-64 batches  
- Efficient batching & preprocessing  
- Deterministic seeding for reproducibility

### **✔ Multi-Crop Augmentation Engine (Global + Local Crops)**
- 2 global crops + multiple local crops  
- Color jitter, grayscale, Gaussian Blur, Solarization  
- Preprocessing consistent with DINOv2 methodology  
- Built for high-quality semantic feature learning

### **✔ Vision Transformer + DINO Projection Head**
- Supports Tiny / Small / Base ViT variants  
- Student network = trainable  
- Teacher network = EMA-updated (momentum 0.996)  
- DINO projection head for embedding alignment

### **✔ DINO Loss (Full Implementation)**
- Temperature scaling  
- Output centering (running mean)  
- Cross-view cross-entropy alignment  
- Teacher temperature warm-up scheduling  
- No shortcuts — full paper-accurate implementation

### **✔ Training Loop (Industry-Style Logic)**
- Multi-crop forward passes  
- Gradient clipping  
- Last-layer freezing option  
- Periodic checkpoint saving  
- EMA teacher updates executed every step

### **✔ Explainability: Attention Map Visualizations**
- Extracts last-layer self-attention weights  
- Creates high-resolution heatmaps and overlays  
- Works for both trained and pretrained DINO models  
- Supports patch-level interpretability

---

## 🔹 Why This Project Is Valuable
This notebook demonstrates the practical engineering required to:

- Build a **self-supervised vision model** without labels  
- Train **transformer-based feature extractors** from scratch  
- Implement modern techniques like **EMA distillation**, **centering**, and **temperature scaling**  
- Produce interpretability artifacts (attention maps) used in model evaluations  
- Create a **reproducible mini-framework** inside a single notebook — ideal for rapid experimentation and research workflows  

---

## 🔹 Tech Stack
- Python, PyTorch, Torchvision  
- Vision Transformers (ViT)  
- NumPy, PIL, Matplotlib  
- OpenCV & scikit-image (for attention overlays)  
- einops (tensor manipulation)  

---

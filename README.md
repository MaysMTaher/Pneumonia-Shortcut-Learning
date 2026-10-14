# Anatomical Leakage and Shortcut Probing in Lightweight Chest Radiography Models

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](#) 
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

Official repository for the paper: **"Anatomical Leakage and Shortcut Probing in Lightweight Chest Radiography Models: A Pediatric-to-Adult Cross-Domain Evaluation"**.

---

## 📌 Abstract
Deep neural networks for pneumonia diagnosis on chest radiographs (CXR) frequently attain near-perfect validation metrics ($\text{AUC} > 0.99$). However, under pediatric-to-adult domain shift, these metrics can be highly misleading. In this study, we systematically probe shortcut reliance across three input modes: **Full Image**, **Lung-Only**, and **Background-Only**. 

Our findings reveal a severe **Background Superiority Paradox**: models trained exclusively on background pixels generalize well externally ($\text{AUC} = 0.9245 - 0.9767$), while models constrained to anatomical lungs suffer a major performance collapse ($\text{AUC} \approx 0.7160 - 0.7367$). To fix this, we introduce **Random Background Masking (RBM 30%)**, which improves external generalization to **0.9633**.

---

## 📊 Key Experimental Results

| Model Architecture | Input Probing Mode | Internal AUC (Pediatric) | External AUC (Adult) | External BAR (%) |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet18** | Full Image | 0.9953 [0.9920–0.9981] | 0.9550 [0.9506–0.9592] | 55.5% |
| **ResNet18** | Lung-Only | 0.9928 [0.9885–0.9962] | **0.7367** [0.7219–0.7521] | — |
| **ResNet18** | Background-Only | 0.9960 [0.9931–0.9985] | **0.9245** [0.9161–0.9330] | — |
| **ResNet18 (Mitigated)** | **RBM 30% Proposed** | **0.9962** | **0.9633** (+0.83%) | — |

---

## 🏗️ System Architecture

![System Architecture](figures/Figure1_Proposed_System_Architecture.png)

---

## Quick Start & Installation

```bash
# Clone repository
git clone [https://github.com/YOUR_USERNAME/Pneumonia-Shortcut-Learning.git](https://github.com/YOUR_USERNAME/Pneumonia-Shortcut-Learning.git)
cd Pneumonia-Shortcut-Learning

# Install dependencies
pip install -r requirements.txt

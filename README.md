<div align="center">

# 💵 Automated Indian Currency Note Classification System
### A Comparative Study of Deep Convolutional Neural Networks (Unit IV Syllabus)

[![PyTorch](https://img.shields.io/badge/Framework-PyTorch%202.0-ee4c2c.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Deployment-Gradio%20Live-orange?style=for-the-badge&logo=gradio)](https://9d30c4418ea63f9be5.gradio.live/)
[![Accuracy](https://img.shields.io/badge/Champion%20Accuracy-97.77%25-brightgreen?style=for-the-badge)](https://github.com/)
[![Precision](https://img.shields.io/badge/Champion%20Precision-97.93%25-blue?style=for-the-badge)](https://github.com/)

<br>

### 🚀 **[Click Here to Test the Live Web App Demo](https://9d30c4418ea63f9be5.gradio.live/)**
*Upload any Indian currency note photo to get real-time denomination detection & confidence scores.*

</div>

---

## 📌 1. Project Overview & Motivation
Automated banknote recognition is essential for automated banking kiosks, Cash Deposit Machines (CDMs), smart currency sorting, and assistive tools for visually impaired individuals. 

This project performs an **empirical comparative benchmark** across **5 foundational CNN architectures** from the university deep learning curriculum (**Unit IV**):
1. **LeNet-5** (Classic baseline trained from scratch)
2. **AlexNet** (ReLU, Dropout, Large Conv Kernels)
3. **Oxford VGG16** (Deep uniform $3 \times 3$ convolutions) — 🏆 **Champion Model**
4. **Google Inception (GoogLeNet)** (Multi-scale parallel inception modules) — 🥈 **Runner-Up**
5. **ResNet-50** (Residual learning with skip connections)

---

## 🏆 2. Benchmark Comparison & Results

All models were evaluated on the **exact same unseen test set of 358 real banknote images** across 8 classes:

| # | Model from Syllabus | Architecture Style | Parameters | Test Accuracy | **Test Precision** | Test Recall | Test F1-Score | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **LeNet-5** | Shallow Conv-Pool (Scratch) | 25.8M | 71.79% | 73.04% | 71.79% | 71.71% | Baseline |
| **2** | **AlexNet** | First Deep CNN (ReLU + Dropout) | 57.0M | 94.13% | 94.63% | 94.13% | 94.18% | Strong |
| **3** | **Oxford VGG16** | **Deep Uniform $3\times3$ Convs** | **134.2M** | **97.77%** | **97.93%** | **97.77%** | **97.78%** | 🏆 **Champion** |
| **4** | **Google Inception** | Multi-Scale Parallel Modules | 5.6M | 96.09% | 96.20% | 96.09% | 96.11% | 🥈 **Runner-Up** |
| **5** | **ResNet-50** | Residual Skip Connections | 23.5M | 60.00% | 82.10% | 60.00% | 62.42% | Under-converged (5 ep) |

### 🔍 Key Scientific Insights:
- **Why Oxford VGG16 Won (97.93% Precision):** Its stack of small $3 \times 3$ filters preserves subtle spatial micro-features (intricate geometric borders, watermark threads, and micro-lettering) without aggressive downsampling.
- **Why Google Inception is the Efficiency Leader:** Achieved **96.20% precision** using only **5.6 million parameters** (~24 times lighter than VGG16) via multi-scale parallel receptive fields.
- **Why LeNet-5 Lagged (71.79% Accuracy):** Designed in 1998 for low-resolution 32x32 grayscale digits; lacks the parameter capacity to handle complex RGB banknote security patterns.

---

## 📊 3. Dataset Breakdown
* **Dataset Name:** Indian Currency Note Images Dataset (2020)
* **Total Samples:** **3,566 Images** across 8 balanced classes
* **Input Resolution:** $224 \times 224 \times 3$ (RGB)

| Class Label | Denomination | Total Images | Percentage |
| :---: | :---: | :---: | :---: |
| `10` | ₹10 Note | 448 | 12.56% |
| `20` | ₹20 Note | 450 | 12.62% |
| `50` | ₹50 Note | 450 | 12.62% |
| `100` | ₹100 Note | 450 | 12.62% |
| `200` | ₹200 Note | 500 | 14.02% |
| `500` | ₹500 Note | 437 | 12.25% |
| `2000` | ₹2000 Note | 438 | 12.28% |
| `Background` | Non-Currency Background | 393 | 11.02% |
| **TOTAL** | **All Categories** | **3,566** | **100.0%** |

### Data Split:
- **Training Set (80%):** 2,852 images (Augmented with rotations, horizontal flips, color jitter)
- **Validation Set (10%):** 356 images (Used for early validation & parameter monitoring)
- **Test Set (10%):** 358 images (Completely held-out unseen test samples)

---

## 🎓 4. University Curriculum Mapping

This implementation covers core topics across all four units of the neural network curriculum:

- **Unit I (Introduction to Neural Networks):**
  - Loss Function: **Categorical Cross-Entropy Loss** ($\mathcal{L} = -\sum y_i \log(\hat{y}_i)$)
  - Output Activation: **Softmax function** for multi-class probability distribution
  - Optimization: **Backpropagation** with gradient descent
- **Unit II (Optimization Techniques):**
  - Optimizer: **Adam (Adaptive Moment Estimation)** with bias correction and momentum
  - Learning Rate Scheduling: Controlled fine-tuning ($\alpha = 10^{-4}$)
- **Unit III (Deep Algorithms & Regularization):**
  - **Dropout Regularization ($p = 0.4$):** Integrated into dense classifier heads to prevent co-adaptation
  - **Batch Normalization:** Stabilizes gradient propagation
- **Unit IV (Convolutional Neural Networks):**
  - Feature extraction through **Input, Convolution, ReLU, Pooling, and Dense layers**
  - Benchmarking of **LeNet-5, AlexNet, Oxford VGG16, Google Inception, and ResNet-50**
  - **Transfer Learning:** Repurposing pre-trained ImageNet representations for specialized banknote features

---

## 🛠️ 5. Project Structure

```text
├── README.md                          # Comprehensive project documentation
├── app.py                             # Gradio web application deployment script
├── requirements.txt                   # Environment dependencies
├── .gitignore                         # Excludes checkpoints and heavy assets
└── Currency_Note_Classification.ipynb # Complete end-to-end Google Colab notebook
```

---

## 💻 6. How to Run Locally

### Prerequisites
Make sure you have Python 3.9+ installed.

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR_USERNAME>/Indian-Currency-Classification.git
cd Indian-Currency-Classification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Web Application
```bash
python app.py
```
Open your browser at `http://127.0.0.1:7860` to use the classifier locally, or use the generated public link to share.

---

## 🌐 7. Live Web Deployment

The application is deployed using **Gradio**:
- **Live URL:** [https://9d30c4418ea63f9be5.gradio.live/](https://9d30c4418ea63f9be5.gradio.live/)
- Supports drag-and-drop image upload with real-time confidence bar graphs across the top 3 predicted denominations.

---

## 📜 8. References
1. Mane, V. (2020). *Indian Currency Note Images Dataset 2020*. Kaggle.
2. Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition* (VGG). arXiv:1409.1556.
3. Szegedy, C., et al. (2015). *Going Deeper with Convolutions* (GoogLeNet / Inception). CVPR 2015.
4. He, K., et al. (2016). *Deep Residual Learning for Image Recognition* (ResNet). CVPR 2016.

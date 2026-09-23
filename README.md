# Automated Indian Currency Classification using Deep CNN Architectures

An end-to-end deep learning project comparing 5 Convolutional Neural Network (CNN) architectures from university curriculum (Unit IV) for automated recognition and classification of Indian currency banknotes.

---

## 🏆 Project Highlights & Benchmark Results
All 5 syllabus models were trained and benchmarked on the exact same dataset of **3,566 images** across 8 classes (`₹10`, `₹20`, `₹50`, `₹100`, `₹200`, `₹500`, `₹2000`, and `Background`):

| Model Name | Architecture Style | Parameters | Test Accuracy | **Test Precision** | Test F1-Score | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. LeNet-5** | Shallow Conv-Pool (Scratch) | 25.8M | 71.79% | 73.04% | 71.71% | Baseline |
| **2. AlexNet** | First Deep CNN (ReLU + Dropout) | 57.0M | 94.13% | 94.63% | 94.18% | Strong |
| **3. Oxford VGG16** | **Deep Uniform 3×3 Convolutions** | **134.2M** | **97.77%** | **97.93%** | **97.78%** | 🏆 **Champion (97.9% Precision)** |
| **4. Google Inception** | Multi-Scale Parallel Modules | 5.6M | 96.09% | 96.20% | 96.11% | 🥈 **Runner-Up (Lightweight)** |
| **5. ResNet-50** | Residual Bottleneck (Skip Connections)| 23.5M | 60.00% | 82.10% | 62.42% | Under-converged (5 epochs) |

---

## 📚 Syllabus Alignment
- **Unit I (Neural Networks):** Categorical Cross-Entropy Loss, Softmax, Backpropagation.
- **Unit II (Optimization):** Adam Optimizer with momentum and adaptive learning rate.
- **Unit III (Regularization):** Dropout (0.4) and Batch Normalization to prevent overfitting.
- **Unit IV (CNNs):** LeNet, AlexNet, Oxford VGG16, Google Inception (GoogLeNet), and ResNet-50.

---

## 💻 Tech Stack
- **Framework:** PyTorch & Torchvision
- **Evaluation:** Scikit-learn, Pandas, Matplotlib, Seaborn
- **Deployment:** Gradio Web Application
- **Data Source:** Kaggle Indian Currency Note Images Dataset (2020)

---

## 🚀 Quick Start & Deployment

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Gradio Web App
```bash
python app.py
```
Open the generated local URL or shareable link in your browser, upload any note image, and get instant prediction and confidence scores.

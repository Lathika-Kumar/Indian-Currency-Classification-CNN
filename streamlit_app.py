import os
import pickle
import torch
from torchvision import transforms
from PIL import Image
import streamlit as st
import pandas as pd

# Set Page Configuration
st.set_page_config(
    page_title="Indian Currency Classifier",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f3f4f6;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #2563eb;
        margin-bottom: 1rem;
    }
    .badge-champion {
        background-color: #10b981;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

# Denomination Motif & Security Information
DENOMINATION_INFO = {
    "₹10": {
        "color": "Chocolate Brown",
        "motif": "Sun Temple, Konark (Odisha)",
        "dimension": "63 mm × 123 mm"
    },
    "₹20": {
        "color": "Greenish Yellow",
        "motif": "Ellora Caves (Maharashtra)",
        "dimension": "63 mm × 129 mm"
    },
    "₹50": {
        "color": "Fluorescent Blue",
        "motif": "Hampi with Chariot (Karnataka)",
        "dimension": "66 mm × 135 mm"
    },
    "₹100": {
        "color": "Lavender",
        "motif": "Rani Ki Vav (Gujarat)",
        "dimension": "66 mm × 142 mm"
    },
    "₹200": {
        "color": "Bright Yellow",
        "motif": "Sanchi Stupa (Madhya Pradesh)",
        "dimension": "66 mm × 146 mm"
    },
    "₹500": {
        "color": "Stone Grey",
        "motif": "Red Fort with Indian Flag (Delhi)",
        "dimension": "66 mm × 150 mm"
    },
    "₹2000": {
        "color": "Magenta",
        "motif": "Mangalyaan (Mars Orbiter Mission)",
        "dimension": "66 mm × 166 mm"
    },
    "Background": {
        "color": "N/A",
        "motif": "Non-currency object or plain background",
        "dimension": "N/A"
    }
}

# Device Selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model release download URL for Streamlit Cloud
MODEL_RELEASE_URL = "https://github.com/Lathika-Kumar/Indian-Currency-Classification-CNN/releases/download/v1.0.0/currency_classifier.pkl"

# Load Model with Streamlit Resource Caching
@st.cache_resource(show_spinner=False)
def load_classifier():
    model_path = os.path.join(os.path.dirname(__file__), "currency_classifier.pkl")
    
    # If model is not found locally (e.g. on Streamlit Cloud), auto-download it from GitHub Releases
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
        download_container = st.empty()
        download_container.info("⏳ Downloading trained model weights (537 MB) from GitHub Releases... This only runs once (~15-20s).")
        progress_bar = st.progress(0, text="Connecting to model host...")
        
        def dl_progress(block_num, block_size, total_size):
            if total_size > 0:
                fraction = min(1.0, (block_num * block_size) / total_size)
                progress_bar.progress(fraction, text=f"Downloading Oxford VGG16: {int(fraction * 100)}% ({int(block_num * block_size / (1024*1024))} MB / {int(total_size / (1024*1024))} MB)")
        
        import urllib.request
        try:
            urllib.request.urlretrieve(MODEL_RELEASE_URL, model_path, reporthook=dl_progress)
            progress_bar.empty()
            download_container.success("✅ Model weights downloaded and cached successfully!")
        except Exception as e:
            progress_bar.empty()
            download_container.error(
                f"⚠️ Model file '{model_path}' is missing on the server and could not be auto-downloaded from:\n"
                f"{MODEL_RELEASE_URL}\n\n"
                f"Error details: {e}\n\n"
                f"👉 Please attach 'currency_classifier.pkl' to your GitHub Release under tag 'v1.0.0'."
            )
            return None, None
    
    with st.spinner("Initializing Oxford VGG16 neural network in memory..."):
        with open(model_path, "rb") as f:
            package = pickle.load(f)
            
        model = package['model']
        class_names = package['class_names']
        
        model.to(device)
        model.eval()
        return model, class_names

model, class_names = load_classifier()

# Image Preprocessing Pipeline (PyTorch VGG Standard)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/banknotes.png", width=72)
    st.markdown("### 🎓 Academic Project Details")
    st.markdown("**Course:** 23ADR405 - Deep Learning")
    st.markdown("**Curriculum:** Unit IV (CNNs)")
    st.markdown("**Institution:** Karpagam College of Engineering")
    st.markdown("**Champion Model:** Oxford VGG16 <span class='badge-champion'>97.93% Precision</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Syllabus Benchmark Comparison")
    benchmark_df = pd.DataFrame({
        "Model": ["LeNet-5", "AlexNet", "VGG16", "Inception", "ResNet-50"],
        "Precision": ["73.04%", "94.63%", "97.93% 🏆", "96.20%", "82.10%"],
        "Params": ["25.8M", "57.0M", "134.2M", "5.6M", "23.5M"]
    })
    st.dataframe(benchmark_df, hide_index=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ System Info")
    st.info(f"**Execution Device:** `{device}`\n\n**Total Classes:** 8 denominations")

# Main Header
st.markdown("<div class='main-header'>💵 Automated Indian Currency Note Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Deep Learning Vision System for Banknote Identification & Denomination Recognition</div>", unsafe_allow_html=True)

# Layout: Two Columns (Upload & Preview | Predictions & Details)
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📤 Upload Currency Image")
    uploaded_file = st.file_uploader(
        "Choose a banknote photo (JPG, PNG, JPEG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload any clear photo of an Indian currency note."
    )
    
    # Camera capture option
    enable_camera = st.checkbox("📸 Use Web Camera Instead", value=False)
    camera_file = None
    if enable_camera:
        camera_file = st.camera_input("Capture banknote using camera")

    active_file = camera_file if enable_camera and camera_file else uploaded_file

    if active_file:
        pil_image = Image.open(active_file).convert("RGB")
        st.image(pil_image, caption="Uploaded Currency Note", use_container_width=True)

with col_right:
    st.markdown("### 🎯 Classification Results")
    
    if active_file and model is not None:
        with st.spinner("Analyzing micro-features and denominations..."):
            tensor = preprocess(pil_image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(tensor)
                probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            
            top_idx = int(probabilities.argmax())
            top_class = class_names[top_idx]
            top_conf = float(probabilities[top_idx]) * 100.0
            
            # Top Prediction Card
            st.markdown(f"""
                <div class='metric-card'>
                    <h4 style='margin:0; color:#1f2937;'>Predicted Denomination:</h4>
                    <h1 style='margin:5px 0; color:#2563eb; font-size:2.8rem;'>{top_class}</h1>
                    <p style='margin:0; font-size:1.1rem; color:#059669; font-weight:600;'>
                        Model Confidence: {top_conf:.2f}%
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Denomination Motif Details
            if top_class in DENOMINATION_INFO and top_class != "Background":
                info = DENOMINATION_INFO[top_class]
                st.markdown("#### 🏛️ Security & Motif Details (RBI Specification)")
                det_col1, det_col2 = st.columns(2)
                with det_col1:
                    st.write(f"🎨 **Base Color:** {info['color']}")
                    st.write(f"📏 **Dimensions:** {info['dimension']}")
                with det_col2:
                    st.write(f"🏛️ **Reverse Motif:** {info['motif']}")
            
            st.markdown("---")
            st.markdown("#### 📈 Denomination Probability Distribution")
            
            prob_dict = {class_names[i]: float(probabilities[i]) * 100.0 for i in range(len(class_names))}
            prob_df = pd.DataFrame(list(prob_dict.items()), columns=["Denomination", "Probability (%)"])
            prob_df = prob_df.sort_values(by="Probability (%)", ascending=False).reset_index(drop=True)
            
            # Render Progress Bars for Top 4 classes
            for idx, row in prob_df.iterrows():
                den = row["Denomination"]
                p = row["Probability (%)"]
                col_bar_lbl, col_bar = st.columns([1, 4])
                with col_bar_lbl:
                    st.markdown(f"**{den}**")
                with col_bar:
                    st.progress(min(max(p / 100.0, 0.0), 1.0), text=f"{p:.2f}%")
                    
    else:
        st.info("👈 Upload an Indian currency note photo from the left panel to see real-time classification.")
        
        # Display sample guide
        st.markdown("#### Supported Denominations:")
        st.markdown("- ₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000")
        st.markdown("- Automatic rejection of non-currency backgrounds")

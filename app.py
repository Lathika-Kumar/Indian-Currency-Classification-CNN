import os
import pickle
import torch
from torchvision import transforms
from PIL import Image
import gradio as gr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Starting application on: {device}")

pkl_path = "currency_classifier.pkl"

if not os.path.exists(pkl_path):
    raise FileNotFoundError(
        f"'{pkl_path}' not found in the current directory!\n"
        f"Please download 'currency_classifier.pkl' from Google Colab and place it inside:\n"
        f"D:\\Projects\\Indian_Currency_Classification"
    )

print(f"Loading '{pkl_path}'...")
with open(pkl_path, "rb") as f:
    package = pickle.load(f)

model = package['model']
class_names = package['class_names']

model.to(device)
model.eval()
print(f"Model successfully loaded! Classes: {class_names}")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def classify_uploaded_image(input_image):
    if input_image is None:
        return {"Please select an image": 0.0}
    
    pil_img = Image.fromarray(input_image).convert('RGB')
    tensor = preprocess(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    return {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}

if __name__ == "__main__":
    demo = gr.Interface(
        fn=classify_uploaded_image,
        inputs=gr.Image(
            sources=['upload'],
            type="numpy",
            label="Upload Currency Note Image"
        ),
        outputs=gr.Label(num_top_classes=3, label="Predicted Denomination & Confidence"),
        title="💵 Automated Indian Currency Note Classifier",
        description="Upload a photo of an Indian currency note to predict its denomination.",
        live=True
    )
    demo.launch(share=True)

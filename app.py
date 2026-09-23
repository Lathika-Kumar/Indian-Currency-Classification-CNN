import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['₹10', '₹100', '₹20', '₹200', '₹2000', '₹50', '₹500', 'Background']
num_classes = len(class_names)

def load_vgg16_model():
    model = models.vgg16(weights=None)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )
    weights_path = "best_vgg16_currency_model.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

classifier_model = load_vgg16_model()

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
        outputs = classifier_model(tensor)
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
        title="Automated Indian Currency Note Classifier",
        description="Upload a photo of an Indian currency note to predict its denomination.",
        live=True
    )
    demo.launch(share=True)

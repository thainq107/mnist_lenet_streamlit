import os
import torch
import gdown
import uuid
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms

@st.cache_resource
def load_model(model_path, num_classes=10):
    lenet_model = LeNetClassifier(num_classes)
    lenet_model.load_state_dict(torch.load(model_path, weights_only=True))
    lenet_model.eval()
    return lenet_model
model = load_model('lenet_model.pt')

def inference(img_path, model):
    image = Image.open(img_path)
    w, h = image.size
    if w != h:
        crop = transforms.CenterCrop(min(w, h))
        image = crop(image)
        wnew, hnew = image.size
    img_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])
    img_new = img_transform(image)
    img_new = img_new.expand(1, 1, 28, 28)
    with torch.no_grad():
        predictions = model(img_new)
    preds = nn.Softmax(dim=1)(predictions)
    p_max, yhat = torch.max(preds.data, 1)
    return p_max.item(), yhat.item()

def main():
  st.title('Digit Recognition')
  st.title('Model: LeNet. Dataset: MNIST')
  uploaded_img = st.file_uploader('Input Image', type=['jpg', 'jpeg', 'png'])
  example_button = st.button('Run example')
  st.divider()
  
  if example_button:
    uploaded_img_path = 'demo_8.png'
  else:
    if uploaded_img is not None:
      uploaded_img_path = uploaded_img
  p, label = inference(uploaded_img_path, model)
  st.image(uploaded_img_path)
  st.success(f"The uploaded image is of the digit {label} with {p:.2f} % probability.") 

if __name__ == '__main__':
     main() 

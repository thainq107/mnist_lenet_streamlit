import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms

class LeNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding='same')
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.avgpool1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.avgpool2(outputs)
        outputs = F.relu(outputs)
        outputs = self.flatten(outputs)
        outputs = self.fc_1(outputs)
        outputs = self.fc_2(outputs)
        outputs = self.fc_3(outputs)
        return outputs

@st.cache_resource
def load_model(model_path, num_classes=10):
    lenet_model = LeNetClassifier(num_classes)
    lenet_model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
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
    # uploaded_img = st.file_uploader('Input Image', type=['jpg', 'jpeg', 'png'])
    # print(uploaded_img)
    # example_button = st.button('Run example')
    # st.divider()
    
    # if example_button:
    #     uploaded_img_path = 'demo_8.png'
    # # else:
    # #     if uploaded_img is not None:
    # #           uploaded_img_path = uploaded_img

    file = st.file_uploader("Please upload an image of a digit", type=["jpg", "png"])
    if file is not None:
        image = Image.open(file)
    st.write(image.size)
    # if file is not None:
    #     p, label = inference(file, model)
    # st.image(file)
    # st.success(f"The uploaded image is of the digit {label} with {p:.2f} % probability.") 

if __name__ == '__main__':
     main() 

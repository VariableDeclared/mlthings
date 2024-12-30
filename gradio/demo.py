import torch
from minio import Minio
import torchvision.transforms as transforms
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Define model layers
        self.model_layers = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(16*97*172, 120),
            nn.ReLU(),
            # Adding hidden layers
            nn.Linear(120, 120),
            nn.Linear(120, 120),
            nn.Linear(120, 120),
                        
            nn.Linear(120, 2)
        )
        
    def forward(self, x):
        out = self.model_layers(x)
        return out

client = Minio(
    "s3.orangebox.cloud:9000",
    access_key="t9ZqEVxN1GbrtTBltMcI",
    secret_key="KazwAJHSaFYZXnomowQ4ICIk6V26k3NQIlRzpKML",
    secure=False,
)

client.fget_object('industrial-cnn-models','model.pth','./model.pth')
model = CNN().to(device)
model.load_state_dict(torch.load('model.pth', weights_only=True))
model.eval()

import requests
from PIL import Image, ImageOps
from torchvision import transforms

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

def predict(inp):
  inp = inp.crop((50, 60, 750, 460))  #Size: 700x400
  inp = ImageOps.grayscale(inp) 
  transform = transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize((0.5), (0.5))])
  inp = transform(inp).unsqueeze(0)
  with torch.no_grad():
    inp = inp.to(device)
    # inp = inp.reshape(400,700, cmap="gray")
    prediction = model(inp).argmax(1)
    if prediction == 0:
      return "abnormal"
    else:
      return "normal"
  return prediction

import gradio as gr

gr.Interface(fn=predict,
             inputs=gr.Image(type="pil"),
             outputs=gr.Label("State")).launch()
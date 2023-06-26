import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

from PIL import Image

class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialTransformer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.unsqueeze(0)
        return x

class TemporalTransformer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalTransformer, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.bn = nn.BatchNorm3d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

class VideoTransformer(nn.Module):
    def __init__(self, in_channels, spatial_out_channels, temporal_out_channels, num_classes):
        super(VideoTransformer, self).__init__()
        self.spatial_transformer = SpatialTransformer(in_channels, spatial_out_channels)
        self.temporal_transformer = TemporalTransformer(spatial_out_channels, temporal_out_channels)
        self.fc = nn.Linear(spatial_out_channels, num_classes)

    def forward(self, x):
        x = self.spatial_transformer(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.temporal_transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x


model = VideoTransformer(in_channels=3, spatial_out_channels=1, temporal_out_channels=1, num_classes=1)
model.eval()

frame_count = 0
cam = cv2.VideoCapture('hdvideo.mp4')

while frame_count < 2:
    ret, frame = cam.read()

    if ret:
        frame_count += 1
        
        # Convert image to RGB form.
        image_rgb = Image.fromarray(frame).convert('RGB')

        # Resize the image.
        resize = transforms.Resize([224, 224])
        image_rgb = resize(image_rgb)

        # Transform the image into a tensor.
        to_tensor = transforms.ToTensor()
        tensor = to_tensor(image_rgb)

        # Add a new dimension of size 1 to tensor. (Align tensor dimension to pass to the resnet model.)
        tensor = tensor.unsqueeze(0)

        input_data = torch.tensor(tensor)
        output = model(input_data)
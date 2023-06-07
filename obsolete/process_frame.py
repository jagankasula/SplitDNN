import cv2
import datetime
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

model = None
model_left = None
model_right = None


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


def init():
    global model, model_left, model_right

    model = models.resnet50(pretrained=True)
    # model.fc = nn.Linear(512, 28, 28) # Adjust the output size to match model_left
    model.eval()
    
    # summary(model, input_size=(3, 224, 224), device='cpu')

    model_left, model_right = split_model(model, split_point=21)
    
    print("Model Left")
    print(model_left)
    # summary(model_left, input_size=(3, 224, 224), device='cpu')
    
    print("Model Right")
    print(model_right)
    summary(model_right, input_size=(512, 28, 28), device='cpu')

    model_left.eval()
    model_right.eval()


def split_model(model, split_point=20):
    net1 = []
    for module in model.children():
        if isinstance(module, nn.Sequential):
            for m in module.children():
                net1.append(m)
        else:
            net1.append(module)

    net1 = nn.Sequential(*net1)
    left = list(net1.children())[:split_point]
    model_left = nn.Sequential(*left)
    model_right = list(net1.children())[split_point:]

    # Adjust the shape of model_right input
    model_right = nn.Sequential(*model_right)

    # Remove the 'flatten' module from model_right
    model_right = model_right[:-1]

    return model_left, model_right


def process_frame(frame):
    img = Image.fromarray(frame).convert('RGB')

    resize = transforms.Resize([224, 224])
    img = resize(img)

    to_tensor = transforms.ToTensor()
    tensor = to_tensor(img)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to('cpu')

    result_left = model_left(tensor)
    result_left = result_left.view(result_left.size(0), -1)
    result = model_right(result_left)

    return result


def process(input_path):
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    video_length = total_frames / fps
    print(f"No of frames: {total_frames}")
    print(f"Video length: {video_length:.2f} seconds")

    startTime = datetime.datetime.now()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = process_frame(frame)
        print(result)

    endTime = datetime.datetime.now()
    print(f"Total processing time: {endTime - startTime}")


if __name__ == '__main__':
    init()
    process('hdvideo.mp4')

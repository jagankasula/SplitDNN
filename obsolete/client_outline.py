import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import datetime
import numpy as np

# Load the ResNet-50 model
model = models.resnet50(pretrained=True)
model.eval()

utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

def producer_video_left(img_rbg):

    count = 0
    print(img_rbg)
    img = Image.fromarray(img_rbg).convert('RGB')
    count+=1
    resize = transforms.Resize([224, 224])
    img = resize(img)

    to_tensor = transforms.ToTensor()
    tensor = to_tensor(img)
    tensor = tensor.unsqueeze(0)

    output = model(tensor)
    output = output.cpu()
    
    return output

def consumer(response_body, count, startTime):
    result_right_np = np.asarray(response_body)
    result_right = torch.Tensor(result_right_np)
    with torch.no_grad():
        output = torch.nn.functional.softmax(result_right, dim=1)
            
    #print(f"Final Result: {result_right}")
    results = utils.pick_n_best(predictions=output, n=1)
    #print(f"Final Result: {results}")
    print(f"Final Result for Image number {count}: {results}")
    endTime = datetime.datetime.now()
    print(f"Total processing time: {endTime - startTime}")


def main_runner():
    
    time_start = datetime.datetime.now()
    count = 0

    print('Timestamp Start: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

    cam = cv2.VideoCapture('hdvideo.mp4')

    startTime = datetime.datetime.now()

    while True:
        count += 1
        ret, img_rbg = cam.read()   
        if ret: 
            output = producer_video_left(img_rbg)
            output = output.detach().numpy()
            consumer(output, count, startTime)
        else:
            time_finish = datetime.datetime.now()
            print('Timestamp Finish: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
            break

    print(f'Timestamp difference: {time_finish - time_start}')
    cam.release()
    cv2.destroyAllWindows()


if __name__=='__main__':
    main_runner()
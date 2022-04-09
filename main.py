from os.path import isfile, join
from os import listdir
import os
import torch
from PIL import Image
import cv2

#1008 756
#yolov5s

# def path_reader(pathway):
#     paths = [f for f in listdir(pathway) if isfile(join(pathway, f))]
#     out_paths = []
#     for ps in range(0, len(paths)):
#         out_paths.append(paths[ps])
#     return out_paths

def get_boundaries():
    img_batch = []
    imgLeft = cv2.imread("leftTest.jpg")
    imgRight = cv2.imread("rightTest.jpg")
    if imgLeft.shape[0] < imgLeft.shape[1]:
        img_size = int(imgLeft.shape[0]/2)
    else:
        img_size = int(imgLeft.shape[1]/2)
    imgLeft = cv2.resize(imgLeft, (img_size, img_size))
    imgRight = cv2.resize(imgRight, (img_size, img_size))
    cv2.imwrite("left.jpg", imgLeft)
    cv2.imwrite("right.jpg", imgRight)
    imgLeft = Image.open("left.jpg")
    imgRight = Image.open("right.jpg")
    img_batch.append(imgLeft)
    img_batch.append(imgRight)
    return img_batch, img_size

def get_strawbs():
    image_outputs, size_outputs = get_boundaries()
    results = model(image_outputs, size=size_outputs)
    results.save()

model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')  # local repo
model.conf = 0.8
if os.path.exists("runs/detect/exp/left.jpg"):
    os.remove("runs/detect/exp/left.jpg")
if os.path.exists("runs/detect/exp/right.jpg"):
    os.remove("runs/detect/exp/right.jpg")
if os.path.exists("runs/detect/exp/"):
    os.rmdir("runs/detect/exp/")
get_strawbs()






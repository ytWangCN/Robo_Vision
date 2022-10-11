import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
from math import ceil
import argparse
import copy
from PIL import Image
from torchvision import transforms, datasets
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from LeNet5 import LeNet5

# parameter of this program
INPUT_IMAGE_SIZE = 100
INPUT_PTH_MODEL_PATH = '../model/Vision_NumDetect.pth'
OUTPUT_ONNX_PATH = '../model/NumDetect.onnx'

# the function about making the ".pth" model trained by pytorch convert to the "ONNX" model
def convert_model_to_ONNX(input_img_size, input_pth_model, output_ONNX):
    dummy_input = torch.randn(1, 1, input_img_size, input_img_size)
    model = LeNet5(5)

    state_dict = torch.load(input_pth_model, map_location='cpu').state_dict()

    # load the model
    model.load_state_dict(state_dict)

    input_names = ["input_image"]
    output_names = ["output_classification"]

    model.eval()

    torch.onnx.export(model, dummy_input, output_ONNX, verbose=True, input_names=input_names,
                      output_names=output_names)

if __name__ == '__main__':
    # call the function named "convert_model_to_ONNX".
    convert_model_to_ONNX(INPUT_IMAGE_SIZE, INPUT_PTH_MODEL_PATH, OUTPUT_ONNX_PATH)

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from model import Image_Encoder_CNN, Image_Encoder_LSTM
from torch.autograd import Variable
from torchvision import transforms


def main():

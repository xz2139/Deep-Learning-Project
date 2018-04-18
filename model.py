import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import models
from torch.autograd import Variable

class Image_Encoder_CNN(nn.Module):

    def __init__(self, input_size):
        super(Image_Encoder_CNN, self).__init__()
        vgg11 = models.vgg11(pretrained=True)
        vgg11.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),)
        self.vgg11 = vgg11
        # print(self.vgg11)
        # print(vgg11)
        self.linear = nn.Linear(vgg11.classifier[0].out_features, input_size)
        self.bn = nn.BatchNorm1d(input_size, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, image):
        imfeatures = self.vgg11(image)
        imfeatures = Variable(imfeatures.data)
        imfeatures = imfeatures.view(imfeatures.size(0), -1)
        imfeatures = self.bn(self.linear(imfeatures))
        return imfeatures


class Image_Encoder_LSTM(nn.Module):

    def __init__(self, embedding_size, hidden_size, num_layers, use_cuda):
        super(Image_Encoder_LSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_cuda = use_cuda
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.init_weights()

    def init_hidden(self, imfeatures):
        hidden = (Variable(torch.zeros(self.num_layers.n_direction, imfeatures.size(0), self.hidden_size)), Variable(torch.zeros(self.n_layers.n_direction, imfeatures.size(0), self.hidden_size)))
        return (hidden[0].cuda(), hidden[1].cuda()) if use_cuda else hidden

    def init_weights(self):
        self.lstm.weight_hh_l0 = nn.init.xavier_uniform(self.lstm.weight_hh_l0)
        self.lstm.weight_ih_l0 = nn.init.xavier_uniform(self.lstm.weight_ih_l0)

    def forward(self, imfeatures):
        hidden, _ = self.lstm(imfeatures)
        output = hidden[-1]
        return output


from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import models


class Image_Encoder_CNN(nn.Module):

    def __init__(self, input_size):
        super(Image_Encoder, self).__init__()
        vgg11 = models.vgg11(pretained=True)
        modules = list(vgg11.children())[:-1]
        self.vgg11 = nn.Sequential(*modules)
        self.linear = nn.Linear(vgg11.fc.in_features, input_size)
        self.bn = nn.BatchNorm1d(input_size, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        self.lstm.weight_hh_l0 = nn.init.xavier_uniform(self.lstm.weight_hh_l0)
        self.lstm.weight_ih_l0 = nn.init.xavier_uniform(self.lstm.weight_ih_l0)

    def forward_vgg(self, image):
        imfeatures = self.vgg11(image)
        imfeatures = Variable(imfeatures.data)
        imfeatures = imfeatures.view(imfeatures.size(0), -1)
        imfeatures = self.bn(self.linear(imfeatures))
        return imfeatures


class Image_Encoder_LSTM(nn.Module):

    def __init__(self, embedding_size, hidden_size, num_layers, use_cuda)
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.use_cuda = use_cuda

    self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
    self.init_weights()

    def init_hidden(self, imfeatures):
        hidden = (Variable(torch.zeros(self.num_layers.n_direction, imfeatures.size(0), self.hidden_size)), Variable(torch.zeros(self.n_layers.n_direction, imfeatures.size(0), self.hidden_size)))
        return (hidden[0].cuda(), hidden[1].cuda()) if use_cuda else hidden

    def init_weight(self):
        self.lstm.weight_hh_l0 = nn.init.xavier_uniform(self.lstm.weight_hh_l0)
        self.lstm.weight_ih_l0 = nn.init.xavier_uniform(self.lstm.weight_ih_l0)

    def forwar_lstm(self, imfeatures):
        hidden, _ = self.lstm(imfeatures)
        output = hidden[-1]
        return output

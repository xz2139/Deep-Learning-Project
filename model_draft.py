import nltk
import pickle
from collections import Counter
import pandas as pd
import glob
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import shutil
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import torchvision
from torchvision import models


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(data='files_sentence.csv', threshold=4):
    """Build a simple vocabulary wrapper."""
    sentences=pd.read_csv('files_sentence.csv')
    mydict = dict(zip(sentences.id, sentences.sentence))
    ids=list(mydict)
    counter = Counter()
    for i, id in enumerate(ids):
        caption = str(mydict[id])
  #      tokens = nltk.tokenize.word_tokenize(caption.lower())
        tokens = caption.lower().split()

        counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized the captions." %(i, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main():
    vocab = build_vocab('train_sentence.csv',
                        threshold=4)
    vocab_path = 'data/vocab_try.pkl'
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: %d" %len(vocab))
    print("Saved the vocabulary wrapper to '%s'" %vocab_path)

    
    
sentences=pd.read_csv('train_sentence.csv')
sentences['file']=None

def files(string):
    (folder,name)=string.split("_")
    return glob.glob('mouths/'+str(string)+'/*.png')

sentences['file']=sentences['id'].apply(files)
sentences[sentences.file.apply(lambda x: len(x))!=0].reset_index().to_pickle('sentence_nonempty.pkl')

with open('vocab_try.pkl', 'rb') as f:
    vocab = pickle.load(f)

with open('sentence_nonempty.pkl', 'rb') as f:
    table = pickle.load(f)


class Dataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, table, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        with open(table, 'rb') as f:
            self.table = pickle.load(f)
        self.ids = list(self.table.id)
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        table = self.table
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = table[table.id==ann_id]['sentence'].item()
        path = table[table.id==ann_id]['file'].item()
        images=[]
        for i in range(151):
            try:
                p=path[i]
                image = Image.open(p).convert('RGB')
                image=np.transpose(image, (2, 0, 1))
#                 if self.transform is not None:
#                     image = self.transform(image)
                image=torch.from_numpy(np.array(image)).float()
                transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                image=transform(image)
                images.append(image)
            except IndexError:
                images.append(torch.from_numpy(np.zeros((3,224,224))).float())                

        seq_img=torch.stack(images,0)
#         seq_img=torch.from_numpy(seq_img).float()

        # Convert caption (string) to word ids.
 #       tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        tokens = (str(caption).lower()).split()
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return seq_img, target

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
#     print(len(images))
#     print(images[0].shape)
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loader(table, vocab,transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    mvlrs = Dataset(table=table,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=mvlrs, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader

loader=get_loader('sentence_nonempty.pkl', vocab, None, 64,shuffle=True, num_workers=0) 

import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import models
from torch.autograd import Variable

class Image_Encoder_CNN(nn.Module):

    def __init__(self, output_size):
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
        self.linear = nn.Linear(vgg11.classifier[0].out_features, output_size)
        self.bn = nn.BatchNorm1d(output_size, momentum=0.01)
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
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        #print('lstm',embed_size, hidden_size, num_layers)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    def sample(self, features, length, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(length):      
 #           print(sampled_ids)# maximum sampling length
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
    #             print(predicted)
    #             print(sampled_ids)
    #             print('breaks-----')
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                         # (batch_size, 1, embed_size)
        sampled_ids = torch.cat(sampled_ids)                  # (batch_size, 20)
        return sampled_ids.squeeze()
    
def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)



cnn_encoder=Image_Encoder_CNN(output_size=1000)
lstm_encoder=Image_Encoder_LSTM(embedding_size=1000, hidden_size=512, num_layers=1, use_cuda=False)
decoder=DecoderRNN(embed_size=512, hidden_size=256, vocab_size=len(vocab), num_layers=1)
criterion = nn.CrossEntropyLoss()
params = list(cnn_encoder.linear.parameters())+list(cnn_encoder.bn.parameters()) + list(lstm_encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=0.001)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
resume=None

best_loss=100
for epoch in range(20):
        for i, (images, captions, lengths) in enumerate(loader):
            images = to_var(images, volatile=True)
            cnn_encoder.zero_grad()
            lstm_encoder.zero_grad()
            l=[]
            for single in torch.unbind(images,1):
                features = cnn_encoder(single)  
                l.append(features)
            
            ll=torch.stack(l,0)
            lstm_features = lstm_encoder(ll)
            outputs = decoder(lstm_features, Variable(captions), lengths)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            loss = criterion(outputs, Variable(targets))
            loss.backward()
            optimizer.step()
            if loss.data[0]<best_loss:
                best_loss=loss.data[0]
                is_best=True
            else:
                is_best=False
                
            # Print log info
            if i % 10 == 0:
                print('Epoch: %d,Step:%d, Loss: %.4f, Perplexity: %5.4f'
                      %(epoch,i, loss.data[0], np.exp(loss.data[0]))) 
                with open('Progress.txt', 'a') as f:
                    print('Epoch: %d,Step:%d, Loss: %.4f, Perplexity: %5.4f'
                      %(epoch,i, loss.data[0], np.exp(loss.data[0])), file=f)  
                        # Save the models
                    print('sample: ',list(decoder.sample(lstm_features[0].unsqueeze(0),lengths[0]).data), file=f)
                    print('caption: ',list(captions[0]), file=f)                
                        
            if i % 50 == 0:

 #               torch.save(cnn_encoder.state_dict(),'cnn_encoder-%d-%d.pkl' %(epoch+1, i+1))
 #               torch.save(lstm_encoder.state_dict(),'lstm_encoder-%d-%d.pkl' %(epoch+1, i+1))                
 #               torch.save(decoder.state_dict(),'decoder-%d-%d.pkl' %(epoch+1, i+1))
                
                save_checkpoint({'epoch': epoch + 1,
                            'step':i+1,
                            'cnn_encoder': cnn_encoder.state_dict(),
                            'lstm_encoder': lstm_encoder.state_dict(),
                            'decoder': decoder.state_dict(),   
                            'best_loss':best_loss,
                            'optimizer' : optimizer.state_dict()}, is_best)
            if resume!=None:
                if os.path.isfile(resume):
                    print("=> loading checkpoint '{}'".format(resume))
                    checkpoint = torch.load(resume)
                    epoch = checkpoint['epoch']
                    best_loss = checkpoint['best_loss']
                    cnn_encoder.load_state_dict(checkpoint['cnn_encoder'])
                    lstm_encoder.load_state_dict(checkpoint['lstm_encoder'])                    
                    decoder.load_state_dict(checkpoint['decoder'])                    
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
                else:
                    print("=> no checkpoint found at '{}'".format(resume))
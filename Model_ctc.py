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
import torch.nn.functional as F
import torchvision
from torchvision import models
from warpctc_pytorch import CTCLoss
USE_CUDA=True


# In[162]:

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
    vocab_path = 'vocab_try.pkl'
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

d=sentences.file.apply(lambda x: len(x)).max()


# In[4]:

class Dataset(data.Dataset):
    def __init__(self, table, vocab, transform=None):
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
        for i in range(d):
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
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    images = torch.stack(images, 0)
    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loader(table, vocab,transform, batch_size, shuffle, num_workers):
    mvlrs = Dataset(table=table,vocab=vocab,transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=mvlrs,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,collate_fn=collate_fn)
    return data_loader

batch_size=32
loader=get_loader('sentence_nonempty.pkl', vocab, None, batch_size,shuffle=True, num_workers=0) 


# In[122]:

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
    def __init__(self, embedding_size, hidden_size, num_layers, use_cuda,drop=0.2):
        super(Image_Encoder_LSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dropout=nn.Dropout(drop)
        self.num_layers = num_layers
        self.use_cuda = use_cuda
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.init_weights()
    def init_hidden(self, imfeatures):
        hidden = (Variable(torch.zeros(self.num_layers.n_direction, imfeatures.size(0), self.hidden_size)),Variable(torch.zeros(self.n_layers.n_direction, imfeatures.size(0), self.hidden_size)))
        return (hidden[0].cuda(), hidden[1].cuda()) if use_cuda else hidden
    def init_weights(self):
        self.lstm.weight_hh_l0 = nn.init.xavier_uniform(self.lstm.weight_hh_l0)
        self.lstm.weight_ih_l0 = nn.init.xavier_uniform(self.lstm.weight_ih_l0)
    def forward(self, imfeatures):
        imfeatures=self.dropout(imfeatures)
        hidden,_= self.lstm(imfeatures)
        output = hidden[-1]
        return output
    


# In[151]:

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size      
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))
    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        # Create variable to store attention energies
        attn_energies = to_var(torch.zeros(this_batch_size, max_len)) # B x S
#         if USE_CUDA:
#             attn_energies = attn_energies.cuda()
        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b].squeeze(0), encoder_outputs[i, b])
        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies).unsqueeze(1)
    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        elif self.method == 'general':
           # a=rnn_output[:,0]
           # b=ll[1, 0]
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy      
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.dot(energy)
            return energy
        
        


# In[153]:

class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.2):
        super(AttnDecoderRNN, self).__init__()
        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
#         self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.lstm=nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device='cuda')
    def forward(self, input_seq, last_hidden, encoder_outputs):
        
        # Note: we run this one step at a time
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size) # S=1 x B x N
        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.lstm(embedded, last_hidden)
        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x S=1 x N
        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))
        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)
        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights

    


# In[154]:

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)



# In[155]:

cnn_encoder=Image_Encoder_CNN(output_size=1000)
lstm_encoder=Image_Encoder_LSTM(embedding_size=1000, hidden_size=1000, num_layers=1, use_cuda=True)
decoder = AttnDecoderRNN('general', 1000, len(vocab), n_layers=1)
criterion = nn.CrossEntropyLoss()
params = list(cnn_encoder.linear.parameters())+list(cnn_encoder.bn.parameters()) +list(lstm_encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=0.0001)
cnn_encoder.cuda()
lstm_encoder.cuda()
decoder.cuda()


# In[10]:

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
resume=None


# In[156]:

best_loss=100
ctc_loss = CTCLoss()
for epoch in range(100):
    epoch_loss=0
    for i, (images, captions, lengths) in enumerate(loader):
        images=images
        captions=captions
        lengths=lengths  
        actual_size=captions.shape[0]
        images = to_var(images, volatile=True)
        cnn_encoder.zero_grad()
        lstm_encoder.zero_grad()
        decoder.zero_grad()
        l=[]
        for single in torch.unbind(images,1):
            features = cnn_encoder(single)  
            l.append(features)

        ll=torch.stack(l,0)
        lstm_features = lstm_encoder(ll)
        decoder_hidden=(lstm_features.unsqueeze(0).contiguous(),to_var(torch.zeros(1, actual_size, 1000)).contiguous())
        #encoder_hidden=to_var(torch.rand(2,batch_size,1000))
        #decoder_hidden = encoder_hidden[:decoder.n_layers*2] # Use last (forward) hidden state from encoder
        decoder_input = to_var(torch.LongTensor([1] * actual_size))
        outputs = to_var(torch.zeros(max(lengths),actual_size, decoder.output_size))
        for t in range(max(lengths)):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, ll)
            outputs[t] = decoder_output
            decoder_input = to_var(captions[:,t])
        # expected shape of seqLength x batchSize x alphabet_size
        probs = Variable(outputs.data, requires_grad=True).contiguous()
        labels = Variable(captions.view(captions.numel()))
        label_sizes = Variable(torch.IntTensor([max(lengths)]*actual_size))
        probs_sizes = Variable(torch.IntTensor([max(lengths)]*actual_size))
        # tells autograd to compute gradients for probs
        loss = ctc_loss(probs.cpu(), labels.int(), probs_sizes, label_sizes)
        epoch_loss+=loss.data[0]
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
            with open('Progress2.txt', 'a') as f:
                print('Epoch: %d,Step:%d, Loss: %.4f, Perplexity: %5.4f'
                  %(epoch,i, loss.data[0], np.exp(loss.data[0])), file=f)  
                    # Save the models
                print('prediction: ',list(outputs[:,0].max(1)[1].data), file=f)
                print('target: ',list(captions[0]), file=f)                

        if i % 50 == 0:

#               torch.save(cnn_encoder.state_dict(),'cnn_encoder-%d-%d.pkl' %(epoch+1, i+1))
#               torch.save(lstm_encoder.state_dict(),'lstm_encoder-%d-%d.pkl' %(epoch+1, i+1))                
#               torch.save(decoder.state_dict(),'decoder-%d-%d.pkl' %(epoch+1, i+1))


#             save_checkpoint({'epoch': epoch + 1,
#                         'step':i+1,
#                         'cnn_encoder': cnn_encoder.state_dict(),
#                         'decoder': decoder.state_dict(),   
#                         'best_loss':best_loss,
#                         'optimizer' : optimizer.state_dict()}, is_best)
            pass

        if resume!=None:
            if os.path.isfile(resume):
                print("=> loading checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume)
                epoch = checkpoint['epoch']
                best_loss = checkpoint['best_loss']
                cnn_encoder.load_state_dict(checkpoint['cnn_encoder'])                   
                decoder.load_state_dict(checkpoint['decoder'])                    
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(resume))

    with open('Progress2.txt', 'a') as f:
        print('Epoch: %d training_loss:%.4f'%(epoch, epoch_loss), file=f)

    sentences2=pd.read_csv('test_sentence.csv').head(1000)
    sentences2['file']=None

    def files(string):
        (folder,name)=string.split("_")
        return glob.glob('/scratch/xz2139/testmouths/'+str(string)+'/*.png')

    sentences2['file']=sentences2['id'].apply(files)

    sentences2[sentences2.file.apply(lambda x: len(x))!=0].reset_index().to_pickle('sentence2_nonempty.pkl')

    with open('sentence2_nonempty.pkl', 'rb') as f:
        table2 = pickle.load(f)

    batch_size=32
    loader2=get_loader('sentence2_nonempty.pkl', vocab, None, batch_size,shuffle=True, num_workers=0) 
    test_loss=0
    for i, (images, captions, lengths) in enumerate(loader2):
        actual_size=captions.shape[0]
        images = to_var(images, volatile=True)
        l=[]
        for single in torch.unbind(images,1):
            features = cnn_encoder(single)  
            l.append(features)
        ll=torch.stack(l,0)
        lstm_features = lstm_encoder(ll)
        decoder_hidden=(lstm_features.unsqueeze(0).contiguous(),to_var(torch.zeros(1, actual_size, 1000)).contiguous())
        #encoder_hidden=to_var(torch.rand(2,batch_size,1000))
        #decoder_hidden = encoder_hidden[:decoder.n_layers*2] # Use last (forward) hidden state from encoder
        decoder_input = to_var(torch.LongTensor([1] * actual_size))
        outputs = to_var(torch.zeros(max(lengths),actual_size, decoder.output_size))
        for t in range(max(lengths)):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, ll)
            outputs[t] = decoder_output
            decoder_input = to_var(captions[:,t])
        probs = Variable(outputs.data, requires_grad=True).contiguous()
        labels = Variable(captions.view(captions.numel()))
        label_sizes = Variable(torch.IntTensor([max(lengths)]*actual_size))
        probs_sizes = Variable(torch.IntTensor([max(lengths)]*actual_size))
        # tells autograd to compute gradients for probs
        lossctc = ctc_loss(probs.cpu(), labels.int(), probs_sizes, label_sizes)
        epoch_loss+=lossctc.data[0]
        test_loss+=lossctc.data[0]
    print(test_loss)

    with open('Progress2.txt', 'a') as f:
        print('Epoch: %d Testing_loss:%.4f'%(epoch, test_loss), file=f)
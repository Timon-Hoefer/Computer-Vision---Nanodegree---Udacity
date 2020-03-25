import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        #define the embedding for the words in the captions
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # define the lstm which takes input of size embed:size and outputs hidden_size objects
        self.lstm = nn.LSTM(embed_size, hidden_size,num_layers,batch_first=True)
        
        # we want as output vectors of length vocab_size
        self.fc = nn.Linear(hidden_size, vocab_size)
        

        
       
        
        
        
    def forward(self, features, captions):
        #delete <end> in every caption, since it is not an input to the LSTM 
        captions = captions[:,:-1]
        
        # create embedded word vectors for each word in a caption
        embed = self.embedding(captions)
        
        # add to all the captions the feature vector as input to the LSTM
        inp = torch.cat((features.unsqueeze(1), embed), 1)
        
        
        #run the LSTM
        out, __ = self.lstm(inp)
        
        #put it through the final linear layer
        captions = self.fc(out)
        return captions

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        outputs = []
        embeddings = inputs
        for ii in range(max_len):
            lstm_out, states = self.lstm(embeddings, states)
            out = lstm_out.squeeze(1)
            out = self.fc(out)
            _, prediction = out.max(1)
            outputs.append(prediction.item())
                
            if prediction == 1:
                break
               
            embeddings = self.embedding(prediction).unsqueeze(1)
        return outputs 
# TODO add shebang to allow only python3 execution
# hw1 driver script
# Author: Azmyin Md. Kamal

"""
TODO answer to Task 3
"""

"""
#* Important Tutorials:
* https://www.youtube.com/watch?v=AvKSPZ7oyVg&ab_channel=PatrickLoeber
* https://cnvrg.io/pytorch-lstm/#:~:text=Long%20Short%20Term%20Memory%20(LSTMs,term%20dependencies%2C%20and%20vanishing%20gradients.
* https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
* Choice of loss function https://machinelearningmastery.com/loss-functions-in-pytorch-models/
"""

"""
* Important implementation details
- max_len = 200, this is a design choice
- number of LSTM layers - 3, this is a design choice
- hidden_size -- needs to be found out by trial and error, starting with 10
"""

# Imports
from model_base import CriticBase, ComposerBase # Import templates
from midi2seq import process_midi_seq, seq2piano, random_piano, piano2seq, segment
import numpy as np # Numpy tools
import pandas as pd # Pandas Dataframe tools
from math import ceil
import sys

import glob # Tools to resolve filepaths
import os.path # Import platform agnostic (Windows, Linux) path resolution tools
from pathlib import Path

import torch # Requried to create tensors to store all numerical values
import torch.nn as nn # Required for weight and bias tensors
import torch.nn.functional as F # Required for the activation functions
from torch.utils.data import TensorDataset, DataLoader, Dataset # How do we use them?
from sklearn.preprocessing import MinMaxScaler # Normalize note values between 0 - 1 # This might be redundant
from sklearn.model_selection import train_test_split
from google_drive_downloader import GoogleDriveDownloader as gdd
import os

# Debugging functions

# ---------------------------------------------------------
def debug_quit():
    print(f"DEBUG Quit")
    sys.exit(0)
# ---------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# print(f"Using device: {device}")

root = './' # Location of the root directory

class CriticLSTM(nn.Module):
    def __init__(self, num_classes = 2, input_size=1, hidden_size=51, num_layers=3): # Initialize with some default values
        super(CriticLSTM,self).__init__() # https://stackoverflow.com/questions/43080583/attributeerror-cannot-assign-module-before-module-init-call
        
        #* LSTM Unit
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.num_layers = num_layers # Number of recurrent layers, we are using 3
        self.input_size = input_size # Input tensor size, for our case its 201
        self.hidden_size = hidden_size # Number of states in the hidden layer
        
        #* Fully connected layer
        # TODO this may be modified to become one class classifier problem
        if (num_classes <2 or num_classes > 2):
            print(f"Critic is a binary classifier, only two classes are allowed!")
            self.num_classes = 2

        self.num_classes = num_classes # 1 -- good music,0 -- bad music

        # Define a 2 or 3 layer LSTM unit
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout = 0.8) 
        
        # Define the output fully connected layer
        self.FC = nn.Linear(self.hidden_size, self.num_classes)
    
    # This method takes data through one forward pass 
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.LSTM(x, (h0, c0))
        out = self.FC(out[:, -1, :])
        
        #! Needs sigmoid here
        
        #out = torch.sigmoid(self.FC(out[:, -1, :])) # Now the output is between 0 and 1
        return out

# Task 1: Critic LSTM
# Input tensor --> LSTM Unit --> Fully connected layer --> 0 or 1
# Most code adopted from https://saturncloud.io/blog/how-to-use-lstm-in-pytorch-for-classification/
class Critic(CriticBase):
    def __init__(self, load_trained = False, conv_val = 0.001): # Initialize with some default values
        super(Critic,self).__init__() # https://stackoverflow.com/questions/43080583/attributeerror-cannot-assign-module-before-module-init-call
        
        # Define model
        self.model = CriticLSTM(num_classes = 2, input_size=1, hidden_size=51, num_layers=3) # With all default values
        self.model = self.model.to(device)
        
        #* Construct model from scratch
        # Define classwide parameters
        self.convergence_threshold = conv_val  
        self.load_trained = load_trained 
        
        url = '1RxQz6Wcl6xzuEhhB7em9L6z9nc4v5QTi'
        
        # Predefine some hyper parameters for training
        self.lr = 0.001
        self.loss_fn = nn.CrossEntropyLoss() # Chosen based on https://machinelearningmastery.com/loss-functions-in-pytorch-models/
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Load model if load_trained is True, else construct the model
        if (load_trained):
            
            #* Load a pretrained model
            """
            If load_trained is True, load a trained model from a file.
            Should include code to download the file from Google drive if necessary.
            else, construct the model
            """
            #* As suggested by Teslim from the class
            gdd.download_file_from_google_drive(file_id=url,
                                    dest_path='./az_critic_final.pth',
                                    unzip=True)

            # gdown.download(url, output, quiet=False)
            self.model.load_state_dict(torch.load("./az_critic_final.pth"))
        
        print(self.model)
    
    # WORKS DO NOT DELETE
    # Inherited from the BaseModel class
    # Called by fit method
    #* Will be called multiple times by the fit method
    def train(self, x):
        
        """
        Train the model on ONE batch of data
        :param x: train data: for Critic , x will be a tuple of two tensors (data, label)\
        x is essentially one batch of data
        :return: (mean) loss of the model on the batch
        """
        loss = None
        
        #in_seq , in_label = x['sequence'].to(device) , x['label'].to(device) # Without this causes an error
        #! in_seq may be a row vector style, we need to reshape it
        in_seq , in_label = x
        
        dims = in_seq.size()
        bz, seq_len = dims[0],dims[1]
        chx = 1 # Default

        desired_shape = (bz, seq_len, chx)
        label_2d = torch.randn(bz, 2) # To convert discrete 1 and 0 to [1,0] and [0,1] encodings

        # Reshape if necessary
        if in_seq.shape != torch.Size(desired_shape):
            in_seq = in_seq.view(bz,seq_len,chx)
        
        # According to professor, 
        # Label is a tensor of n values (0 or 1) where n is the batch size. 
        # We are converting them into one-hot encoding of 2 bits i.e 2 classes
        for idx, lbs in enumerate(in_label):
            label_2d[idx] = torch.tensor([1, 0]).float() if lbs.item() else torch.tensor([0, 1]).float()
        
        in_seq = in_seq.to(device)
        label_2d = label_2d.to(device)
        output = self.model(in_seq)
        loss = self.loss_fn(output, label_2d)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(f"Loss for this batch: -- {loss.item()}")

        return loss
    
    #! TODO prepare when Task 2 is done
    def score(self, x):
        '''
        Compute the score of a music sequence
        :param x: a music sequence
        :return: the score between 0 and 1 that reflects the quality of the music: the closer to 1, the better
        '''
        # Pseudocode
        # Run the model
        # Run sigmoid function
        # Report from the two classes, report the class with highest score?
        
        self.model.train(False)
        with torch.no_grad():
            output = self.model(x.to(device))
            predicted_index = torch.argmax(output, dim=1)
            predicted_index ^= 1 # index 0 is good and index 1 is bad 
        return predicted_index

# ------------------------------------------------ EOF Critic -------------------------------------------------------

class ComposerLSTM(nn.Module):
    def __init__(self, seq_len = 50, hidden_size=64, num_layers=2, dense_size = 128, emb_dim = 10, vocab_size = 382):
        super(ComposerLSTM, self).__init__()
        # Based on https://www.youtube.com/watch?v=euwN5DHfLEo&t=111s&ab_channel=mildlyoverfitted
        #* To make a prediction, at least a sequence of 50 piano notes must be passed in
        #self.input_size = input_size # Input size to LSTM for one timestep's data
        self.num_layers = num_layers # Number of recurrent layers, we are using 3
        self.hidden_size = hidden_size # Number of states in the hidden layer
        self.vocab_size = vocab_size # Final output would be class that chooses one note from the available unique notes pool
        self.emb_dim = emb_dim # Rule of thumb, a number between 7 ~ 19, the embedding size should be between the square root and the cube root of the number of categories
        self.max_norm = 2 # Experimental
        self.seq_len = seq_len # Number of notes in each sequence
        self.dense_dim = dense_size # Number of neurons that immediately follows the LSTM network
        #* I used a rule of thum, self.dense_dim will have 4 times the number of hidden states in LSTM

        #* Embedding will encode each integer into a continous variable.
        self.embedding = nn.Embedding(num_embeddings = self.vocab_size, 
                                      embedding_dim = self.emb_dim)
        
        #* LSTM layer acts on the embeddings, not the category data themselves
        self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, 
                            batch_first=True, dropout = 0.3) 
        
        # Add batch normalization here
        self.bn = nn.BatchNorm1d(hidden_size)  # Batch normalization layer

        #* Take LSTM's output to a fully connected layer
        #* Check basics.ipynb to find out why these two layers are defined like this
        self.linear_1 = nn.Linear(self.seq_len, self.dense_dim)
        self.linear_2 = nn.Linear(self.dense_dim, self.vocab_size) # Outputs logits over all notes in the vocabulary

        # Softmax layer
        self.softmax = nn.Softmax(dim = 1)

    # This method takes data through one forward pass 
    def forward(self, x):
        """
        x - torch tensor, ([batch_size, seq_len, 1]) 3D tensor where each note is a row in the sample, torch.int64
        returns prob [batch_size, vocab_size]: Tensor containing probaiblity for all the classes
        """
        # Define work variables
        out = None

        # # Initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0),  self.hidden_size).to(device)
        
        emb = self.embedding(x) # [batch_size, seq_len, emb_dim]
        oo,(_,_) = self.lstm(emb, (h0,c0))

        oo = self.bn(oo.permute(0, 2, 1)).permute(0, 2, 1) # Batch normalize
        oo_mean = oo.mean(dim = 2) # [batch_size, seq_len]
        
        fc_out = self.linear_1(oo_mean) # (seq_len, dense_dim)
        # TODO test with leaky relu
        # https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
        fc_out = F.relu(fc_out) # Relu activation
        logits = self.linear_2(fc_out) # logits (dense_dim, vocab_size)
        
        probs = logits # https://discuss.pytorch.org/t/cross-entropy-loss-is-not-decreasing/43814/2
        # Rescale logits to values between 0 to 1  
        # probs = F.softmax(logits, dim=1) # probs [batch_size, vocab_size]
        
        #  In language modeling hidden states are used to define the probability of 
        #  the next word, p(wt+1|w1,...,wt) =softmax(Wht+b).
        #  print(f"probs shape: {probs.shape}") # ([328]) an row vector of 382 values each containing the probaiblity of the predicted note

        return probs


class Composer(ComposerBase):
    #* Need to get rid of num_notes
    def __init__(self, load_trained = False, seq_len = 50,  conv_val = 0.001): # Initialize with some default values
        super(Composer, self).__init__()

        # Preprocessing, build a library of all unique notes
        self.vocab_dim = 128*2 + 100 + int(ceil(126/5)) # Number of unique notes in the vocabulary
        self.notes = [x for x in range(self.vocab_dim)] # Each integer now represents one unique class, Numpy

        # Define model
        self.model = ComposerLSTM(seq_len=seq_len)
        self.model = self.model.to(device)
        
        # Set a convergence threshold for loss
        self.convergence_threshold = conv_val
        self.load_trained = load_trained
        
        #* LSTM Unit
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        
        # Predefine some hyper parameters for training
        self.lr = 0.01
        self.loss_fn = nn.CrossEntropyLoss() # Predicting categorical distribution over piano notes
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Load model if load_trained is True, else construct the model
        if (load_trained):
            
            file_id = "1EMLxoMZ5CaX49yckSwdfE7AL7vhvVVnN"
           
            #* Load a pretrained model
            """
            If load_trained is True, load a trained model from a file.
            Should include code to download the file from Google drive if necessary.
            else, construct the model
            """
            #* As suggested by Teslim from the class
            gdd.download_file_from_google_drive(file_id=file_id, dest_path='./az_cps.pth',unzip=True)
            self.model.load_state_dict(torch.load("./az_cps.pth"))

        print(self.model)
    
    # # Inherited from the BaseModel class
    # #* Will be called multiple times by the fit method
    def train(self,x):
        """
        x - single torch tensor containing rows of sequences when training Composer
        """
        # Define work variables
        loss = None
        
        # Convert batched tensor to correct shape
        dims = x.size() 
        _, seq_len = dims[0],dims[1]

        # Make sure x is on cuda
        x = x.to(device)
        
        #* Divide into training sequence and test note
        x_train = x[:, :(seq_len - 1)].to(device)
        y_test = x[:, -1] #[(batch_size)] # Array of notes equaling the batch_size
        
        # TODO self.in_action encoding needs to happen here on the fly
        probs = self.model(x_train) # Predict, detach from graph, [batch_size, 382]
        loss = self.loss_fn(probs, y_test) # https://saturncloud.io/blog/how-to-calculate-crossentropy-from-probabilities-in-pytorch/
        print(f"Loss in this bach: {loss.item()}")

        # Do backpropagation, Adam optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
    
    def compose(self, n = 200):
        
        '''
        Generate a music sequence
        :param n: length of the sequence to be generated
        :return: the generated sequence
        '''
        # Pull vocabulary from model instance
        # notes_vocab = self.notes

        print(f"Number of notes in Composer vocabulary: {len(self.notes)}")

        if n % 50 == 0:
            print(f"Sequence length is completely divisible by 50")
            quos = n//50 # Floor division
        else:
            print(f"Sequence length not completely divisible by 50")
            quos = n//50 # Floor division
            n = quos * 50 # This number is now divisible by 50

        midi_notes = [] # Master list that will hold all the notes in the sequence

        # Initialize working tensor
        midi_in = torch.zeros(50, dtype=torch.long).to(device) # ([50])

        # Choose the starting note
        first_notes_lss = [256,257,258,259,260,261,262,263,264] # Found from data analysis
        first_note = np.random.choice(first_notes_lss)

        # Push this note final event list
        midi_notes.append(first_note)

        # Push first note into the work tensor
        midi_in[0] = torch.tensor(first_note, dtype=torch.long)

        #* Our model was trained for a sequence of valid note sequence
        #* Just passing [256, 0,0, ....... 0] yeilds a nonsense sequence

        #* Version 1: Use only random piano notes

        rand_seq= piano2seq(random_piano(n=49)) #! Generates a lot more piano notes than specified
        rand_seq = rand_seq[:49]
        # Add this part to the main event list

        ## We can choose to push only a subset or all of it
        for nn in rand_seq:
            midi_notes.append(nn)

        ## Update the work tensor
        midi_in[1:] = torch.from_numpy(rand_seq)

        #* Version 2 start with a good piano sequence from the database
        #* Objectively workse
        # midi_load = process_midi_seq(datadir=root, n=1, maxlen=49)
        # # Update main list
        # for nn in midi_load[0][:49]:
        #     midi_notes.append(nn)
        
        # # Update work tensor
        # midi_in[1:] = torch.from_numpy(midi_load[0][:49])

        # How many notes remains?
        n = n - 50

        # Set model to evaluation mode, we only want to infer
        self.model.eval() 
        
        # Get new notes
        for i in range(n): # Count it now n - 1
            midi_in = midi_in.to(device) # Put work tensor to GPU
            with torch.no_grad():
                in_seq = midi_in.clone().view(1,-1) # Copy without referece, convert to a batched input
                probs = self.model(in_seq) # Inferece, output logits
            
            midi_in = midi_in.cpu() # Pull midi_in from CPU for processing
            probs = self.model.softmax(probs) # Apply softmax to scale all logits between 0 and 1.
            probs = probs.detach().cpu() # Detach from computation graph move tensor to CPU
            next_note_pred = self.notes[torch.argmax(probs).item()] # Scalar, piano note
            # print(f"next_note: {next_note_pred}")

            # Push this note to master list
            midi_notes.append(next_note_pred)

            # Update work tensor and shift event sequence by one note forward
            next_note_tensor= torch.tensor(next_note_pred ,dtype=torch.long)
            midi_in = torch.cat((midi_in, next_note_tensor.unsqueeze(-1)),dim = 0)
            midi_in = midi_in[1:] # Shift by one position forward
        
        return np.array(midi_notes) # Numpy 1D array

    
    # ------------------------------------------------ EOF Composer -------------------------------------------------------

# ------------------------------------------------------------ EOF --------------------------------------------------
#! # WORKS DO NOT DELETE
    # # Inherited from the BaseModel class
    # # Called by fit method
    # #* Will be called multiple times by the fit method
    # def train(self, x):
        
    #     """
    #     Train the model on ONE batch of data
    #     :param x: train data: for Critic , x will be a tuple of two tensors (data, label)\
    #     x is essentially one batch of data
    #     :return: (mean) loss of the model on the batch
    #     """
    #     in_seq , in_label = x['sequence'].to(device) , x['label'].to(device) # Without this causes an error
    #     output = self.model(in_seq)
    #     loss = self.loss_fn(output, in_label)

    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    #     return loss



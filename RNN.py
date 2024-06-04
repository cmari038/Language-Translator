import torch
import torch.nn as nn
import pandas as pd
import numpy as np

data = pd.read_csv("data.csv")

if torch.cuda.is_available("cuda"):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    

class RNN(nn.Module):
    def __init__(self, input, hidden_dimension, num_layers, output):
        self.hidden_dimension = hidden_dimension
        self.num_layers = num_layers
        self.rnn = nn.RNN(input, hidden_dimension, num_layers, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_dimension, output)
    
    def forward(self, x):
        h = torch.zeros(self.num_layers, x.size(0), self.hidden_dimension)
        out, hn = self.rnn(x, h)
        out = self.fc(out[:, -1, :]) 
        return out
        
    
    


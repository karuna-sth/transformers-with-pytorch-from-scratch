import torch
import math
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init()
        self.d_model = d_model
        self.vocabsize = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) *math.sqrt(self.d_model)
     
     

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
         
        # matrix of shape (seq_len, d_model)
        pe = torch.zero_(seq_len, d_model)

        # vector of shape (seq_len)
        position = torch.arange(0, seq_len, device=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # sin to even postion and cos to odd
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.sin(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        #save tensor in file
        self.register_buffer('pe', pe)

    def forward(self, x):
        x += (self.pe[:, :x.shape[1]]).requires_grad_(False)
        return self.dropout(x)
    


# Add & Norm
class LayerNormalization(nn.Module):
    def __init__(self, eplison: float = 10**-6) -> None:
        super().__init__()
        self.eplison = eplison
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x-mean) / (std+self.eplison) + self.bias
    


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout:float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        # (batch, seq_len, d_model) ---> (batch, seq_len, d_ff ) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    


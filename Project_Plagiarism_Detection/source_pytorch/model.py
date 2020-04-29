# torch imports
import torch.nn.functional as F
import torch.nn as nn


## TODO: Complete this classifier
class BinaryClassifier(nn.Module):
    """
    Define a neural network that performs binary classification.
    The network should accept your number of features as input, and produce 
    a single sigmoid value, that can be rounded to a label: 0 or 1, as output.
    
    Notes on training:
    To train a binary classifier in PyTorch, use BCELoss.
    BCELoss is binary cross entropy loss, documentation: https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss
    """

    ## TODO: Define the init function, the input params are required (for loading code in train.py to work)
    def __init__(self, input_features, hidden_dim, output_dim):
        """s
        Initialize the model by setting up linear layers.
        Use the input parameters to help define the layers of your model.
        :param input_features: the number of input features in your training/test data
        :param hidden_dim: helps define the number of nodes in the hidden layer(s)
        :param output_dim: the number of outputs you want to produce
        """
        super(BinaryClassifier, self).__init__()
        
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = 0.3

        # define any initial layers, here
        
        self.norm_feat = nn.BatchNorm1d(self.input_features)
        
        self.norm_feat_to_hidden = nn.Linear(self.input_features, self.hidden_dim)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
        self.hidden_to_logits = nn.Linear(self.hidden_dim, self.output_dim)
        
    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        """
        Perform a forward pass of our model on input features, x.
        :param x: A batch of input features of size (batch_size, input_features)
        :return: A single, sigmoid-activated value as output
        """
        
        # define the feedforward behavior
        
        n = self.norm_feat(x)
        h = self.norm_feat_to_hidden(n)
        relu = F.relu(h)
        d = self.dropout(relu)
        l = self.hidden_to_logits(d)
        s = F.sigmoid(l)
        
        return s
    
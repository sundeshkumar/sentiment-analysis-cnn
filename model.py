import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextCNN, self).__init__()
        
        # Embedding Layer:specifies 300-dimensional vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Convolutional Layers:  uses filter windows (h) of 3, 4, 5
        # with 100 feature maps for each window size.
        self.conv3 = nn.Conv2d(1, 100, (3, embed_dim))
        self.conv4 = nn.Conv2d(1, 100, (4, embed_dim))
        self.conv5 = nn.Conv2d(1, 100, (5, embed_dim))
        
        # Dropout Layer: specifies a dropout rate (p) of 0.5
        self.dropout = nn.Dropout(0.5) # to prevent overfitting
        
        # Fully Connected Layer: 
        # Inputs = 100 filters * 3 window sizes = 300 features
        self.fc = nn.Linear(300, num_classes)

    def forward(self, x):
        # Input x shape: [Batch_Size, Sequence_Length]
        x = self.embedding(x)
        
        # Add channel dimension (similar to image channels for Conv2d)
        # Shape becomes: [Batch_Size, 1, Sequence_Length, Embedding_Dim]
        x = x.unsqueeze(1) 

        # Convolution + ReLU + Max-over-time Pooling
        
        # Filter window size 3
        x3 = F.relu(self.conv3(x)).squeeze(3)
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)
        
        # Filter window size 4
        x4 = F.relu(self.conv4(x)).squeeze(3)
        x4 = F.max_pool1d(x4, x4.size(2)).squeeze(2)
        
        # Filter window size 5
        x5 = F.relu(self.conv5(x)).squeeze(3)
        x5 = F.max_pool1d(x5, x5.size(2)).squeeze(2)

        # Concatenate features from all filter maps
        out = torch.cat((x3, x4, x5), 1)
        
        # Apply Dropout and Final Classification
        out = self.dropout(out)
        logits = self.fc(out)
        
        return logits
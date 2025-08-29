import torch.nn as nn

class TextClassificationNN(nn.Module):
    def __init__(self, input_tensor=500, num_classes=2):

        super(TextClassificationNN, self).__init__()
        self.input_dim = input_tensor 
        self.num_classes = num_classes
        
        self.layers = nn.Sequential(
            nn.Linear(input_tensor , 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.layers(x)
    
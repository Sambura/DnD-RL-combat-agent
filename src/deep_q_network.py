from torch import nn

class DnDEvalModel(nn.Module):
    def __init__(self, in_layers, out_layers):
        super(DnDEvalModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_layers, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 16, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, out_layers, 3, padding=1),
        )

    def forward(self, x):
        return self.layers(x)

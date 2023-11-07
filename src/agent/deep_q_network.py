from torch import nn

class DnDEvalModel(nn.Module):
    def __init__(self, in_layers: int, out_layers: int):
        super(DnDEvalModel, self).__init__()
        relu_slope = 0.003
        self.layers = nn.Sequential(
            nn.Conv2d(in_layers, 16, 3, padding=1),
            nn.LeakyReLU(relu_slope),
            nn.Conv2d(16, 64, 5, padding=2),
            nn.LeakyReLU(relu_slope),
            nn.Conv2d(64, 16, 5, padding=2),
            nn.LeakyReLU(relu_slope),
            nn.Conv2d(16, out_layers, 3, padding=1),
        )

    def forward(self, x):
        return self.layers(x)

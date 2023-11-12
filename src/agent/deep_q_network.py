from torch import nn

class DnDEvalModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        relu_slope = 0.003
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.LeakyReLU(relu_slope),
            nn.Conv2d(16, 64, 5, padding=2),
            nn.LeakyReLU(relu_slope),
            nn.Conv2d(64, 16, 5, padding=2),
            nn.LeakyReLU(relu_slope),
            nn.Conv2d(16, out_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.layers(x)

class DnDEvalModelRelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 16, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.layers(x)

class ResidualBlock(nn.Module):
    def __init__(self, num_channels: int=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            nn.BatchNorm2d(num_channels)
        )
    
    def forward(self, x):
        y = self.layers(x)
        return nn.functional.relu(y + x)
    
class DnDEvalModelRT(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.Conv2d(256, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, kernel_size=15, padding=7)
        )

    def forward(self, x):
        return self.layers(x)
    
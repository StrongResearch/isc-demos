from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, inch, ouch, krnl, strd, padd, drop):
        super(ConvBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=inch, out_channels=ouch, kernel_size=krnl, stride=strd, padding=padd),
            nn.Dropout(p=drop),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.model(x)

class ConvNet(nn.Module):
    def __init__(self, dropout):
        super(ConvNet, self).__init__()
        self.model = nn.Sequential(
            # (B, C, H, W) = (B, 1, 28, 28) -> (B, 50, 24, 24)
            ConvBlock(inch=1, ouch=50, krnl=3, strd=1, padd=0, drop=dropout),
            # (B, C, H, W) = (B, 50, 24, 24) -> (B, 100, 8, 8)
            ConvBlock(inch=50, ouch=100, krnl=5, strd=2, padd=0, drop=dropout),
            # (B, C, H, W) = (B, 100, 8, 8) -> (B, 500, 1, 1)
            ConvBlock(inch=100, ouch=500, krnl=4, strd=2, padd=0, drop=dropout),
            nn.Flatten(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(100, 10)
        )
    def forward(self, x):
        return self.model(x)

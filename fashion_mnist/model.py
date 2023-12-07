from torch import nn

class ConvNet(nn.Module):
    def __init__(self, dropout):
        super(ConvNet, self).__init__()
        self.model = nn.Sequential(
            # (B, C, H, W) = (B, 1, 28, 28) -> (B, 10, 26, 26), Feature map 784 -> 6,760
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=0),
            nn.Dropout(p=dropout),
            # (B, C, H, W) = (B, 10, 26, 26) -> (B, 10, 24, 24) Feature map 6,760 -> 5,760
            nn.MaxPool2d(kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # (B, C, H, W) = (B, 10, 24, 24) -> (B, 30, 10, 10) Feature map 5,760 -> 3,000
            nn.Conv2d(in_channels=10, out_channels=30, kernel_size=5, stride=2, padding=0),
            nn.Dropout(p=dropout),
            # (B, C, H, W) = (B, 30, 10, 10) -> (B, 30, 8, 8) Feature map 3,000 -> 1,920
            nn.MaxPool2d(kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # (B, C, H, W) = (B, 30, 8, 8) -> (B, 50, 3, 3) Feature map 1,920 -> 450
            nn.Conv2d(in_channels=30, out_channels=50, kernel_size=4, stride=2, padding=0),
            nn.Dropout(p=dropout),
            # (B, C, H, W) = (B, 50, 3, 3) -> (B, 50, 1, 1) Feature map 450 -> 50
            nn.MaxPool2d(kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        logits = self.model(x)
        return logits
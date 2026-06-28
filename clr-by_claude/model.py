import torch
import torch.nn as nn

def conv_block(in_ch, out_ch):
    """Two conv layers + ReLU activation. The basic repeating unit of a U-Net."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )

class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # --- Encoder: shrink the image, learn broad patterns ---
        self.enc1 = conv_block(3, 16)       # input: 3 color channels (RGB)
        self.pool1 = nn.MaxPool2d(2)        # halves the image size (128 -> 64)

        self.enc2 = conv_block(16, 32)
        self.pool2 = nn.MaxPool2d(2)        # 64 -> 32

        self.bottleneck = conv_block(32, 64)  # smallest, most compressed representation

        # --- Decoder: grow back up, fill in detail ---
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 32 -> 64
        self.dec2 = conv_block(64, 32)  # 64 in-channels because we concat the skip connection

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # 64 -> 128
        self.dec1 = conv_block(32, 16)

        self.out_conv = nn.Conv2d(16, 3, kernel_size=1)  # back to 3 color channels

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)  # skip connection: bring back detail from encoder
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        return torch.sigmoid(out)  # squashes output to [0,1], matching normalized image values
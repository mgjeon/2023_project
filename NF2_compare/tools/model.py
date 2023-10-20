import torch
from torch import nn

class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.remap = nn.Conv3d(in_channels=3, out_channels=8, kernel_size=2, stride=(2, 1, 1), padding=(0, 0, 128))
        self.remapT = nn.ConvTranspose3d(in_channels=8, out_channels=3, kernel_size=(3, 2, 1), stride=(2, 1, 1), padding=(0, 0, 103))

        self.encoder = nn.Sequential(                                                              
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=2, stride=1, padding='same'), 
            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=2), # (16, 128, 128, 128)
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding='same'),
            nn.LeakyReLU(0.1), 
            nn.MaxPool3d(kernel_size=2), # (32, 64, 64, 64)
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding='same'),
            nn.LeakyReLU(0.1), 
            nn.MaxPool3d(kernel_size=2), # (64, 32, 32, 32)    
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding='same'),
            nn.LeakyReLU(0.1), 
            nn.MaxPool3d(kernel_size=2), # (128, 16, 16, 16)         
            nn.Conv3d(in_channels=128, out_channels=1, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU(0.1), 
            nn.MaxPool3d(kernel_size=2), # (1, 8, 8, 8)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels=1, out_channels=128, kernel_size=2, stride=2, padding=0),     # (128, 16, 16, 16)
            nn.LeakyReLU(0.1),
            nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0),    # (64, 32, 32, 32)
            nn.LeakyReLU(0.1),
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=0),     # (32, 64, 64, 64)
            nn.LeakyReLU(0.1),
            nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0),     # (16, 128, 128, 128)
            nn.LeakyReLU(0.1),
            nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=2, stride=2, padding=0),      # (8, 256, 256, 256)
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        x = self.remap(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.remapT(x)
        return x
    

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.encoder1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
        )

        self.encoder3 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
        )

        self.encoder4 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(2, 2, 1), padding=(1, 1, 0)),
            nn.ReLU(),
            nn.Conv3d(256, 512, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
        )

        self.upconv3 = nn.ConvTranspose3d(512, 512, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=(1, 1, 0))
        self.decoder3 = nn.Sequential(
            nn.Conv3d(256+512, 256, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
        )
        
        self.upconv2 = nn.ConvTranspose3d(256, 256, kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.decoder2 = nn.Sequential(
            nn.Conv3d(128+256, 128, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
        )

        self.upconv1 = nn.ConvTranspose3d(128, 128, kernel_size=(3, 3, 1), stride=(2, 2, 1))
        self.decoder1 = nn.Sequential(
            nn.Conv3d(64+128, 64, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
        )

        self.final = nn.Conv3d(64, 50, kernel_size=(3, 3, 1), padding=(1, 1, 0))

    def forward(self, x):
        x1 = self.encoder1(x)
        x = self.maxpool(x1)
        x2 = self.encoder2(x)
        x = self.maxpool(x2)
        x3 = self.encoder3(x)
        x = self.maxpool(x3)
        x = self.encoder4(x)
        x = self.upconv3(x)
        x = torch.concatenate([x, x3], 1)
        x = self.decoder3(x)
        x = self.upconv2(x)
        x = torch.concatenate([x, x2], 1)
        x = self.decoder2(x)
        x = self.upconv1(x)
        x = torch.concatenate([x, x1], 1)
        x = self.decoder1(x)

        x = self.final(x)

        return x
    

class zUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.first = nn.Conv3d(1, 50, kernel_size=(3, 3, 1), padding=(1, 1, 0))

        self.encoder1 = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
        )

        self.encoder3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
        )

        self.encoder4 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=(2, 2, 1), padding=(1, 1, 0)),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
        )

        self.upconv3 = nn.ConvTranspose3d(256, 256, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=(1, 1, 0))
        self.decoder3 = nn.Sequential(
            nn.Conv3d(128+256, 128, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
        )
        
        self.upconv2 = nn.ConvTranspose3d(128, 128, kernel_size=(2, 2, 3), stride=(2, 2, 2))
        self.decoder2 = nn.Sequential(
            nn.Conv3d(64+128, 64, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
        )

        self.upconv1 = nn.ConvTranspose3d(64, 64, kernel_size=(3, 3, 2), stride=(2, 2, 2))
        self.decoder1 = nn.Sequential(
            nn.Conv3d(32+64, 32, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
        )

        self.final = nn.Conv3d(32, 3, kernel_size=(3, 3, 1), padding=(1, 1, 0))

    def forward(self, x):
        x = torch.permute(x, (0, 4, 2, 3, 1))
        x = self.first(x)
        x = torch.permute(x, (0, 4, 2, 3, 1))
        x1 = self.encoder1(x)
        x = self.maxpool2(x1)
        x2 = self.encoder2(x)
        x = self.maxpool2(x2)
        x3 = self.encoder3(x)
        x = self.maxpool(x3)
        x = self.encoder4(x)
        x = self.upconv3(x)
        x = torch.concatenate([x, x3], 1)
        x = self.decoder3(x)
        x = self.upconv2(x)
        x = torch.concatenate([x, x2], 1)
        x = self.decoder2(x)
        x = self.upconv1(x)
        x = torch.concatenate([x, x1], 1)
        x = self.decoder1(x)

        x = self.final(x)

        return x
    

class small_zUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.first = nn.Conv3d(1, 50, kernel_size=(3, 3, 1), padding=(1, 1, 0))

        n = 8

        self.encoder1 = nn.Sequential(
            nn.Conv3d(3, n, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
            nn.Conv3d(n, 2*n, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv3d(2*n, 2*n, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
            nn.Conv3d(2*n, 4*n, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
        )

        self.encoder3 = nn.Sequential(
            nn.Conv3d(4*n, 4*n, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
            nn.Conv3d(4*n, 8*n, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
        )

        self.encoder4 = nn.Sequential(
            nn.Conv3d(8*n, 8*n, kernel_size=(2, 2, 1), padding=(1, 1, 0)),
            nn.ReLU(),
            nn.Conv3d(8*n, 16*n, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
        )

        self.upconv3 = nn.ConvTranspose3d(16*n, 16*n, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=(1, 1, 0))
        self.decoder3 = nn.Sequential(
            nn.Conv3d(8*n+16*n, 8*n, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
            nn.Conv3d(8*n, 8*n, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
        )
        
        self.upconv2 = nn.ConvTranspose3d(8*n, 8*n, kernel_size=(2, 2, 3), stride=(2, 2, 2))
        self.decoder2 = nn.Sequential(
            nn.Conv3d(4*n+8*n, 4*n, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
            nn.Conv3d(4*n, 4*n, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
        )

        self.upconv1 = nn.ConvTranspose3d(32, 32, kernel_size=(3, 3, 2), stride=(2, 2, 2))
        self.decoder1 = nn.Sequential(
            nn.Conv3d(2*n+4*n, 2*n, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
            nn.Conv3d(2*n, 2*n, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(),
        )

        self.final = nn.Conv3d(2*n, 3, kernel_size=(3, 3, 1), padding=(1, 1, 0))

    def forward(self, x):
        x = torch.permute(x, (0, 4, 2, 3, 1))
        x = self.first(x)
        x = torch.permute(x, (0, 4, 2, 3, 1))
        x1 = self.encoder1(x)
        x = self.maxpool2(x1)
        x2 = self.encoder2(x)
        x = self.maxpool2(x2)
        x3 = self.encoder3(x)
        x = self.maxpool(x3)
        x = self.encoder4(x)
        x = self.upconv3(x)
        x = torch.concatenate([x, x3], 1)
        x = self.decoder3(x)
        x = self.upconv2(x)
        x = torch.concatenate([x, x2], 1)
        x = self.decoder2(x)
        x = self.upconv1(x)
        x = torch.concatenate([x, x1], 1)
        x = self.decoder1(x)

        x = self.final(x)

        return x
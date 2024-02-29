import torch.nn as nn
import lightning.pytorch as pl
import torch
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.2),

            nn.ReLU(nn.BatchNorm2d(mid_channels)),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.2),

            nn.ReLU(nn.BatchNorm2d(out_channels)),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class UNet2(pl.LightningModule):
    def __init__(self, n_channels,n_out_channels):
        super(UNet2, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = 0.0001
        self.weight_decay = 0.0001
        self.n_channels = n_channels
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024))
        #self.down5 = (Down(1024, 2048))
        #self.down6 = Down(2048,4096)
        #self.up1 = Up(4096,2048)
        #self.up2 = (Up(2048,1024))
        self.up3 = (Up(1024, 512))
        self.up4 = (Up(512, 256))
        self.up5 = (Up(256, 128))
        self.up6 = (Up(128, 64))
        self.outc = (OutConv(64, n_out_channels))
        self.loss = torch.nn.MSELoss()
      #  self.precision = torchmetrics.Precision(task='multiclass',average='macro',num_classes=n_classes)
      #  self.recall = torchmetrics.Recall(task='multiclass',average='macro',num_classes=n_classes)
      #  self.confmat = ConfusionMatrix(task='multiclass',num_classes=n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        #x6 = self.down5(x5)
        #x7 = self.down6(x6)
        #x = self.up1(x7, x6)
        #x = self.up2(x6, x5)
        x = self.up3(x5, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        logits = self.outc(x)
        return logits

    def training_step(self, batch, batch_idx):
        x = batch['inputs']
        y = batch['targets']        
        logits = self.forward(x)
        loss = self.loss(logits, y)
        #acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('training loss', loss, on_epoch=True,on_step=True,prog_bar=True)
        #self.log('train accuracy',acc,on_epoch=True,on_step=False)
        # self.log('train balanced acc',balanced_accuracy_score(y,logits.argmax(dim=1),self.n_classes),on_epoch=True)
        # self.log('train precision',self.precision(logits,y),on_step=False,on_epoch=True)
        # self.log('train recall',self.recall(logits,y),on_step=False,on_epoch=True)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        x = batch['inputs']
        y = batch['targets']
        logits = self.forward(x)
        loss = self.loss(logits, y)
        #acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('validation loss', loss, on_step=True,on_epoch=True,prog_bar=True)
       # self.log('validation accuracy',acc,on_step=False,on_epoch=True)
       # self.log('valid balanced acc',balanced_accuracy_score(y,logits.argmax(dim=1),self.confmat),on_epoch=True)
        # self.log('validation recall',self.recall(logits,y),on_step=False,on_epoch=True)
        return {'loss': loss}
    
#    def test_step(self, batch, batch_idx):
#        x, y,time = batch
#        logits = self.forward(x)
#        loss = self.loss(logits, y)
       # acc = (logits.argmax(dim=1) == y).float().mean()
       # self.log('test accuracy',acc,on_epoch=True,on_step=False,)
       # self.log('test loss', loss, on_epoch=True, on_step=False,logger=True)
#        return {'test loss': loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        predictions = self.forward(batch['inputs'])
        y = batch['targets']
        return predictions,y

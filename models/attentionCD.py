import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.utils import model_zoo
from torchvision import models


class SegNetEnc(nn.Module):

    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()

        layers = [
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ]
        layers += [
                      nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
                      nn.BatchNorm2d(in_channels // 2),
                      nn.ReLU(inplace=True),
                  ] * num_layers
        layers += [
            nn.Conv2d(in_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class SegNetEnc11(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor, num_layers):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear'),
        ]
        layers += [
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class SegNetEnc2(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor, num_layers):
        super().__init__()

        layers = [
            nn.Upsample(scale_factor=scale_factor, mode='bilinear'),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class SegNetEnc3(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor, num_layers):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class feature_extractor(nn.Module):
    def __init__(self,backbone='vgg16',pretrained=True,freeze_backbone=False):
        super(feature_extractor,self).__init__()

        vgg = models.vgg16(pretrained)
        features = list(vgg.features.children())
        self.dec1 = nn.Sequential(*features[:5])#160
        self.dec2 = nn.Sequential(*features[5:10])#80
        self.dec3 = nn.Sequential(*features[10:17])#40
        self.dec4 = nn.Sequential(*features[17:24])#20
        self.dec5 = nn.Sequential(*features[24:])#10

        self.enc5 = SegNetEnc(512, 512, 1)#20
        self.enc4 = SegNetEnc(1024, 256, 1)#40
        self.enc3 = SegNetEnc(512, 128, 1)#80
        self.enc2 = SegNetEnc(256, 128, 1)#160
        # self.enc1 = SegNetEnc(192, 64, 1)  # 160
        self.enc1 = nn.Sequential(
            #nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(192, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )#160
        # self.enc1 = SegNetEnc3(192, 64,1,1) #320
        # self.enc5c = SegNetEnc2(512, 256, 2, 1)#40
        # self.enc4c = SegNetEnc2(256, 128, 2, 1)#80
        # self.enc3c = SegNetEnc2(128, 64, 2, 1)#160
        # self.enc2c = SegNetEnc3(128,64, 1,1)#320
        self.enc5xy = SegNetEnc(512,256,1)
        self.enc4xy = SegNetEnc(512,256,1)
        self.enc3xy = SegNetEnc(384,128,1)
        self.enc2xy = SegNetEnc2(256,128,1, 1)
        self.enc1xy = SegNetEnc2(192,128, 1, 1)#40

        self.hfc = SegNetEnc3(512, 128, 2, 1)#40   384->128
        self.lfc = SegNetEnc3(384, 128, 2, 1)#40   192->128
        self.p5 = nn.Conv2d(256, 2, 3, padding=1)#320
        self.p4 = nn.Conv2d(256,2,3,padding=1)
        self.p3 = nn.Conv2d(128,2,3, padding=1)
        self.p2 = nn.Conv2d(128,2,3, padding=1)
        self.p1 = nn.Conv2d(128, 2, 3, padding=1)

        #initialize_weights(self.enc5, self.enc4, self.enc3,
        #                   self.enc2, self.enc1, self.enc5c,self.enc4c, self.enc3c, self.enc2c)
        #self.lf_af = SegNetEnc2(64, 64, 2, 1)#320
        #self.lf_prediction = nn.Conv2d(64, 2, 3, padding=1)#320

        #self.hf_af = SegNetEnc2(192, 64, 2, 1)#320
        #self.hf_prediction = nn.Conv2d(64, 2, 3, padding=1)

    def forward(self, x,y):
        x_f1 = self.dec1(x)
        x_f2 = self.dec2(x_f1)
        x_f3 = self.dec3(x_f2)
        x_f4 = self.dec4(x_f3)
        x_f5 = self.dec5(x_f4)

        x_enc5 = self.enc5(x_f5)

        x_enc4 = self.enc4(torch.cat([x_f4, x_enc5], 1))

        x_enc3 = self.enc3(torch.cat([x_f3, x_enc4], 1))

        x_enc2 = self.enc2(torch.cat([x_f2, x_enc3], 1))
        x_enc1 = self.enc1(torch.cat([x_f1, x_enc2], 1))
        # x_enc5 = self.enc5c(x_enc5)#40
        # x_enc4 = self.enc4c(x_enc4)#80
        # x_enc3 = self.enc3c(x_enc3)#160
        # x_enc2 = self.enc2c(x_enc2)#320

        y_f1 = self.dec1(y)
        y_f2 = self.dec2(y_f1)
        y_f3 = self.dec3(y_f2)
        y_f4 = self.dec4(y_f3)
        y_f5 = self.dec5(y_f4)

        y_enc5 = self.enc5(y_f5)  #

        y_enc4 = self.enc4(torch.cat([y_f4, y_enc5], 1))  #

        y_enc3 = self.enc3(torch.cat([y_f3, y_enc4], 1))  #

        y_enc2 = self.enc2(torch.cat([y_f2, y_enc3], 1))  #
        y_enc1 = self.enc1(torch.cat([y_f1, y_enc2], 1))  #

        enc5xy = self.enc5xy(abs(x_enc5-y_enc5))
        enc4xy = self.enc4xy(torch.cat([abs(x_enc4-y_enc4), enc5xy],1))
        enc3xy = self.enc3xy(torch.cat([abs(x_enc3-y_enc3), enc4xy],1))
        enc2xy = self.enc2xy(torch.cat([abs(x_enc2-y_enc2), enc3xy],1))
        enc1xy = self.enc1xy(torch.cat([abs(x_enc1 - y_enc1), enc2xy], 1))

        p5 = self.p5(enc5xy)
        p4 = self.p4(enc4xy)
        p3 = self.p3(enc3xy)
        p2 = self.p2(enc2xy)
        p1 = self.p1(enc1xy)

        hf = torch.cat(
            [F.upsample_bilinear(enc5xy, enc4xy.size()[2:]), enc4xy],
            1)  # 160
        hf = self.hfc(hf)
        # print(x_hf.size())#torch.Size([1, 192, 40, 40])
        lf = torch.cat([F.upsample_bilinear(enc3xy, enc2xy.size()[2:]), enc2xy, enc1xy], 1)#160
        lf = self.lfc(lf)



        return p1,p2,p3,p4,p5, hf, lf

class cross_AD_enhancement(nn.Module):
    def __init__(self):
        super(cross_AD_enhancement,self).__init__()
        self.ADHF2ADLF = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),  # 128->128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.ADLF2ADHF = nn.Sequential(
            nn.Conv2d(128,128,3,padding = 1), #128->128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.adhl = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1),#256->64
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))

        #self.adlfrf = nn.Sequential(nn.Conv2d(576, 64, 3, padding=1),
         ##                         nn.BatchNorm2d(64),
           #                       nn.ReLU(inplace=True))

        self.adhl_prediction = nn.Conv2d(64, 2, 3, padding=1)#64

        self.hf_prediction = nn.Conv2d(128,2,3,padding=1)#128
        self.lf_prediction = nn.Conv2d(128,2,3,padding=1)

        #initialize_weights(self.enc5, self.enc4, self.enc3, self.enc2, self.enc1, self.hf_af, self.hf_prediction,
        #                   self.hf_xy, self.lfe, self.l_xy_hf_af, self.edge, self.attention, self.final_prediction)

    def forward(self,hft,lft):

        hf_prediction = self.hf_prediction(hft)
        lf_prediction = self.lf_prediction(lft)
        #print(ad_hf.size())
        #print(ad_lf.size())

        adhl = self.adhl(torch.cat([F.upsample_bilinear(hft, lft.size()[2:]), lft], 1))

        adhl_prediction = self.adhl_prediction(adhl)

        ad_hf2lf = self.ADHF2ADLF(F.upsample_bilinear(hft, lft.size()[2:]))
        ad_lf2hf = self.ADLF2ADHF(F.upsample_bilinear(lft, hft.size()[2:]))

        hft1 = hft.mul(ad_lf2hf)
        lft1 = lft.mul(ad_hf2lf)

        return hf_prediction,lf_prediction, adhl_prediction, hft1,lft1


class attentionCD(nn.Module):
    def __init__(self, num_classes, in_channels=3, backbone='vgg16', pretrained=True, use_aux=True, freeze_bn=False,
                 freeze_backbone=False):
        super(attentionCD, self).__init__()

        self.feature_extractor = feature_extractor()
        self.cross_AD_enhancement = cross_AD_enhancement()
        self.final_prediction = nn.Conv2d(16,2,3,padding = 1)

        #self.l_xy_hf_af = SegNetEnc3(192,64,1,1)
        #self.edge =nn.Conv2d(64,num_classes,3,padding=1)
        #initialize_weights( self.cross_AD_enhancement, self.hf_prediction, self.lf_prediction,
        #                   self.adhl, self.adhl_prediction, self.final_prediction)
        #if freeze_bn: self.freeze_bn()
        #if freeze_backbone:
         #   set_trainable([self.f1, self.f2, self.f3, self.f4, self.f5], False)

    def forward(self, x,y):
        [p1,p2,p3,p4,p5, hft, lft] = self.feature_extractor(x,y)
        # recurrent
        hf_prediction = []
        lf_prediction= []
        adhl_prediction = []

        for i in range(3):
            [hf_p,lf_p,ad_p, hft,lft] = self.cross_AD_enhancement(hft,lft)
            hf_prediction.append(F.upsample_bilinear(hf_p,x.size()[2:]))
            lf_prediction.append(F.upsample_bilinear(lf_p,x.size()[2:]))
            adhl_prediction.append(F.upsample_bilinear(ad_p,x.size()[2:]))


        p1 = F.upsample_bilinear(p1, x.size()[2:])
        p2 = F.upsample_bilinear(p2, x.size()[2:])
        p3 = F.upsample_bilinear(p3, x.size()[2:])
        p4 = F.upsample_bilinear(p4, x.size()[2:])
        p5 = F.upsample_bilinear(p5, x.size()[2:])
        hf_p = F.upsample_bilinear(hf_p, x.size()[2:])
        lf_p = F.upsample_bilinear(lf_p, x.size()[2:])
        ad_p = F.upsample_bilinear(ad_p, x.size()[2:])

        final_prediction = self.final_prediction(torch.cat([p1,p2,p3,p4,p5,hf_p,lf_p,ad_p],1))


        return hf_prediction, lf_prediction, adhl_prediction, p1,p2,p3,p4,p5,final_prediction

    #def get_backbone_params(self):
    #    return chain(self.initial.parameters(), self.layer1.parameters(), self.layer2.parameters(),
    #                 self.layer3.parameters(), self.layer4.parameters())

   # def get_decoder_params(self):
   #     return chain(self.master_branch.parameters(), self.auxiliary_branch.parameters())
    def get_backbone_params(self):
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18,wide_resnet50_2
from torchsummary import summary
from collections import OrderedDict
import pickle

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        #identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = a_w * a_h

        return out





class Decoder (nn.Module):
    def __init__(self, base_width, out_channels=1):
        super(Decoder , self).__init__()

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(256),
                                 nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(
            nn.Conv2d(256+256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.db1_shor_cut = nn.Sequential(  nn.Conv2d(256+256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),)



        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(256, 128, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(
            nn.Conv2d(128+128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.db2_shor_cut = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True), )
        #self.inception = InceptionB(192)

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(128, 64, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True))
        self.db3 = nn.Sequential(
            nn.Conv2d(64+64,64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.db3_shor_cut = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(inplace=True), )

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(64, 32, kernel_size=3, padding=1),
                                 nn.BatchNorm2d( 32),
                                 nn.ReLU(inplace=True))

        self.db4 = nn.Sequential(
            nn.Conv2d(32+64, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.db4_shor_cut = nn.Sequential(nn.Conv2d(96, 48, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(48),
                                          nn.ReLU(inplace=True), )

        self.up5 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(48, 48, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(48),
                                 nn.ReLU(inplace=True))

        self.se = SE(in_chnls=32,ratio=4)


        self.db5 = nn.Sequential(
            nn.Conv2d(48, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.db5_shor_cut = nn.Sequential(nn.Conv2d(48, 24, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(24),
                                          nn.ReLU(inplace=True), )

        self.res_bn_relu = nn.Sequential(nn.BatchNorm2d(24),
                                         nn.ReLU(inplace=True), )
        self.final_out = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 2, kernel_size=3, padding=1),
            #nn.Sigmoid(),

        )


        self.Init()


    def Init(self):
        mo_list = [self.up1,self.up2,self.up3,self.up4,self.up5, self.db1,self.db2,self.db3,self.db4,self.db5,self.final_out,self.se,self.res_bn_relu,
                   self.db1_shor_cut,self.db2_shor_cut,self.db3_shor_cut,self.db4_shor_cut,self.db5_shor_cut]
        for m in mo_list:
        #for m in self.block_down1.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, forward_out,aggregate1,aggregate2,aggregate3,bn_out_128x128,x):
        up1 = self.up1(forward_out)
        cat = torch.cat((up1,aggregate3),dim=1)
        db1 = self.db1(cat)
        #db1 = db1 + self.db1_shor_cut(cat)

        up2 = self.up2(db1)
        cat = torch.cat((up2,aggregate2),dim=1)
        db2 = self.db2(cat)
        #db2 = db2 + self.db2_shor_cut(cat)


        up3 = self.up3(db2)
        cat = torch.cat((up3,aggregate1),dim=1)
        db3 = self.db3(cat)
        #db3 = db3 + self.db3_shor_cut(cat)

        up4 = self.up4(db3)
        cat = torch.cat((up4, bn_out_128x128), dim=1)
        db4 = self.db4(cat)
        #db4 = db4 + self.db4_shor_cut(cat)

        up5 = self.up5(db4)
        db5 = self.db5(up5)




        out = self.final_out(db5)



        return out



class Encoder(nn.Module):
    def __init__(self, pretrained=True, head_layers=[512,512,512,512,512,512,512,512,128], num_classes=2,data_type=None):
        super(Encoder, self).__init__()
        #self.resnet18 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=pretrained)
        self.resnet18 = resnet18(pretrained=pretrained)
        self.resnet18.avgpool=nn.Identity()
        self.resnet18.fc = nn.Identity()
        self.Init()




    def Init(self,):
        for param in self.resnet18.parameters():
            param.requires_grad = False
        for param in self.resnet18.layer4.parameters():
            param.requires_grad = True




    def forward(self, x):

        #layer1_out_64x64, layer2_out_32x32, layer3_out_16x16, layer_final_256x256 = self.get_feature(x)

        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        forward_out = self.resnet18.layer4(x)

        #bn_out_128x128 = 5555
        return forward_out,#bn_out_128x128,layer1_out_64x64,layer2_out_32x32,layer3_out_16x16,layer_final_256x256




class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)



class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, int(in_channels/2), kernel_size=3, padding=1)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, int(in_channels/4), kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(int(in_channels/4), int(in_channels/4), kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(int(in_channels/4), int(in_channels/4), kernel_size=3, padding=1)

        self.branch1x1 = BasicConv2d(in_channels, int(in_channels/4), kernel_size=1)
    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch1x1 = self.branch1x1(x)

        outputs = [branch3x3, branch3x3dbl, branch1x1]
        return torch.cat(outputs, 1)


class SE(nn.Module):
    def __init__(self, in_chnls, ratio=7,out_chnls=66):
        super(SE, self).__init__()
        self.f = BasicConv2d(in_chnls,in_chnls,kernel_size=3, padding=1)
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        # b, c, h, w = x.size()
        #out = self.f(x)
        out = self.squeeze(x)
        # out = out.view(b,c)
        out = self.compress(out)
        out = F.relu(out)
        out_t = self.excitation(out)  # .view(b,c,1,1)
        out = torch.sigmoid(out_t)
        return out

class Aggregate (nn.Module):
    def __init__(self,use_se=None,duochidu=True):
        super(Aggregate , self).__init__()
        self.use_se = use_se
        self.duochidu = duochidu


        self.inception1 = InceptionB(in_channels = 64)
        self.inception2 = InceptionB(in_channels = 128)
        self.inception3 = InceptionB(in_channels = 256)

        self.layer1_64x64_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.se1 = SE(in_chnls=128, ratio=4)
        self.layer1_64x64_2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer2_32x32_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.se2 = SE(in_chnls=256, ratio=4)
        self.layer2_32x32_2 = nn.Sequential(
        nn.Conv2d(256, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        )


        self.layer3_16x16_1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.se3 = SE(in_chnls=512, ratio=4)
        self.layer3_16x16_2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )




        self.layer3_up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(256, 128, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(inplace=True))
        self.layer2_up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(128, 64, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True))

        self.coordatt128 = CoordAtt(inp=128,oup=128)
        self.coordatt256 = CoordAtt(inp=256, oup=256)
        self.coordatt512 = CoordAtt(inp=512, oup=512)

        self.Init()


    def Init(self):
        mo_list = [self.se1,self.se2,self.se3,self.inception1,self.inception2,self.inception3,
                   self.layer1_64x64_1,self.layer1_64x64_2,self.layer2_32x32_1,self.layer2_32x32_2,self.layer3_16x16_1,self.layer3_16x16_2,
                   self.layer3_up,self.layer2_up,
                   self.coordatt128,self.coordatt256,self.coordatt512]
        for m in mo_list:
        #for m in self.block_down1.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, layer1_out_64x64_o, layer2_out_32x32_o, layer3_out_16x16_o, layer1_out_64x64_m, layer2_out_32x32_m,layer3_out_16x16_m):


        score_map1_temp  = torch.mean(layer1_out_64x64_m, dim=1).unsqueeze(1)
        score_map2_temp = torch.mean(layer2_out_32x32_m, dim=1).unsqueeze(1)
        score_map3_temp = torch.mean(layer3_out_16x16_m, dim=1).unsqueeze(1)


        score_map2 = F.interpolate(score_map2_temp, size=64,
                                   mode='bilinear', align_corners=False)  # 对任何尺度 上采样到224,224
        score_map3 = F.interpolate(score_map3_temp, size=64,
                                   mode='bilinear', align_corners=False)  # 对任何尺度 上采样到224,224
        score_map1 = score_map1_temp * score_map2 * score_map3


        score_map3 = F.interpolate(score_map3_temp, size=32,
                                   mode='bilinear', align_corners=False)  # 对任何尺度 上采样到224,224
        score_map2 = score_map2_temp*score_map3
        score_map3 = score_map3_temp



        layer1_out_64x64 = torch.cat((layer1_out_64x64_o, layer1_out_64x64_m),dim=1)
        layer2_out_32x32 = torch.cat((layer2_out_32x32_o, layer2_out_32x32_m),dim=1)
        layer3_out_16x16 = torch.cat((layer3_out_16x16_o, layer3_out_16x16_m),dim=1)

        out1 = self.layer1_64x64_1(layer1_out_64x64)
        if self.use_se:
            weight1 = self.se1(layer1_out_64x64)
        else:
            #print('ssssssssss')
            weight1 = self.coordatt128(layer1_out_64x64)
        out1_temp  = out1*weight1
        out1 = self.layer1_64x64_2(out1_temp)


        out2 = self.layer2_32x32_1(layer2_out_32x32)
        if self.use_se:
            weight2 = self.se2(layer2_out_32x32)
        else:
            weight2 = self.coordatt256(layer2_out_32x32)
        out2_temp = out2*weight2
        out2 = self.layer2_32x32_2(out2_temp)

        out3 = self.layer3_16x16_1(layer3_out_16x16)
        if self.use_se:
            weight3 = self.se3(layer3_out_16x16)
        else:
            weight3 = self.coordatt512(layer3_out_16x16)
        out3_temp = out3*weight3
        out3 = self.layer3_16x16_2(out3_temp)


        aggregate1 =  out1 #B,64,64,64
        aggregate2 =  out2 #B,128,32,32
        aggregate3 = out3 #B,256,16,16

        if self.duochidu:
            #print("ddd")
            temp3 = self.layer3_up(aggregate3)
            aggregate2 = aggregate2 + temp3

            temp2 = self.layer2_up(aggregate2)
            aggregate1 = aggregate1+temp2
        else:
            #print('ppp')
            pass


        return aggregate1*score_map1, aggregate2*score_map2 ,aggregate3*score_map3

        #return aggregate1 , aggregate2 , aggregate3

class ProjectionNet(nn.Module):
    def __init__(self,out_features=False,num_classes = 2,data_type=None,use_se=None,use_duibi=False,duochidu=True):
        super(ProjectionNet, self).__init__()
        self.encoder_segment = Encoder(data_type=data_type)
        self.decoder_segment = Decoder (base_width=64)
        self.aggregate = Aggregate(use_se=use_se,duochidu = duochidu)
        #self.segment_act = torch.nn.Sigmoid()
        self.out_features = out_features
        self.use_duibi = use_duibi

        self.avgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(256, 64)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.bott_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            )


        for m in self.bott_conv.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)
    def forward(self, x, layer1_out_64x64_o, layer2_out_32x32_o, layer3_out_16x16_o, layer1_out_64x64_m, layer2_out_32x32_m,layer3_out_16x16_m,bn_out_128x128):
        forward_out = self.encoder_segment(x)
        forward_out = forward_out[0]
        aggregate1,aggregate2,aggregate3 = self.aggregate(layer1_out_64x64_o, layer2_out_32x32_o, layer3_out_16x16_o, layer1_out_64x64_m, layer2_out_32x32_m,layer3_out_16x16_m)
        output_segment = self.decoder_segment(forward_out,aggregate1,aggregate2,aggregate3,bn_out_128x128,x)
        layer_final_256x256 = 1

        if self.use_duibi:
            cls_result = forward_out.flatten(start_dim=1)
        else:
            cls_result = 1



        if self.out_features:
            return output_segment,cls_result
        else:
            return output_segment,cls_result,layer_final_256x256






if __name__ == '__main__':
    model = ProjectionNet(data_type='wood')
    for name, value in model.named_parameters():
        print(name)
        print(value.requires_grad)

    #print(model)
    model.to(torch.device("cuda"))
    summary(model, (3, 256, 256))




import torch
import torch.nn as nn
import pretrainedmodels
from torchvision.models import resnet18, resnet34, resnet50, squeezenet1_1



#class UNet_UP(nn.Module):
#	def __init__(self):

class UpSample(nn.Module):
	def __init__(self, scale_factor):
		super(UpSample, self).__init__()
		self.scale_factor = scale_factor

	def forward(self, x):
		return F.interpolate(x, None, self.scale_factor, 'bilinear', align_corners = True)

class ConvBlock(nn.Module):
	def __init__(self, ic, oc, ks = 3, stride = 1, pad = 1, use_bias = False):
		super(ConvBlock, self).__init__()
		self.conv = nn.Conv2d(ic, oc, ks, stride, pad, bias = use_bias)
		self.bn = nn.BatchNorm2d(oc)
		self.relu = nn.ReLU(inplace = True)

	def forward(self, x):
		out = self.relu(self.bn(self.conv(x)))
		return out

class UNet_Decoder(nn.Module):
	def __init__(self, ic_upsample, ic, mc, oc):
		super(UNet_Decoder, self).__init__()
		self.upsample = nn.Sequential(
			UpSample(2),
			nn.Conv2d(ic_upsample, ic_upsample, 1, 1, 0, True)
		)
		self.conv1 = ConvBlock(ic, mc)
		self.conv2 = ConvBlock(mc, oc)

	def forward(self, x, x_encoder):
		# (bs, ic_upsample, h, w)
		out = self.upsample(x)
		if(x_encoder is not None):
			# (bs, ic_upsample, h * 2, w * 2)
			out = torch.cat((x, x_encoder), 1)
		# (bs, ic, h * 2, w * 2)
		out = self.conv1(out)
		out = self.conv2(out)
		# (bs, oc, h * 2, w * 2)
		return out

class UNet_Middle(nn.Module):
	def __init__(self, ic, oc):
		super(UNet_Middle, self).__init__()
		self.conv1 = ConvBlock(ic, oc)
		self.conv2 = ConvBlock(oc, oc)
		self.maxpool = nn.MaxPool2d(2, 2)

	def forward(self, x):
		out = self.maxpool(self.conv2(self.conv1(x)))
		return out

class UNet_ResNet34(nn.Module):
	def __init__(self, oc):
		super(UNet_ResNet34, self).__init__()
		self.oc = oc
		self.base_bone = resnet34(True)

		self.init_layers = nn.Sequential(*list(self.base_bone.children())[:3])
		self.enc_layer1 = self.base_bone.layer1
		self.enc_layer2 = self.base_bone.layer2
		self.enc_layer3 = self.base_bone.layer3
		self.enc_layer4 = self.base_bone.layer4

		self.middle_layer = UNet_Middle(512, 512)

		self.dec_layer1 = UNet_Decoder(256, 256+512, 512, 64)
		self.dec_layer2 = UNet_Decoder(64, 64+256, 256, 64)
		self.dec_layer3 = UNet_Decoder(64, 64+128, 128, 64)
		self.dec_layer4 = UNet_Decoder(64, 64+64, 64, 64)
		self.dec_layer5 = UNet_Decoder(64, 64, 32, 64)

		self.last_conv = nn.Conv2d(64, self.oc, 1, 1, 0, True)

	def forward(self, x):
		# (bs, nc, h, w)
		en1 = self.enc_layer1(self.init_layers(x))
		# (bs, 64, h / 2, w / 2)
		en2 = self.enc_layer2(en1)
		# (bs, 128, h / 4, w / 4)
		en3 = self.enc_layer3(en2)
		# (bs, 256, h / 8, w / 8)
		en4 = self.enc_layer4(en3)
		# (bs, 512, h / 16, w / 16)

		mid = self.middle_layer(en4)
		# (bs, 512, h / 32, w / 32)

		de1 = self.dec_layer1(mid, en4)
		# (bs, 64, h / 16, w / 16)
		de2 = self.dec_layer2(de1, en3)
		# (bs, 64, h / 8, w / 8)
		de3 = self.dec_layer3(de2, en2)
		# (bs, 64, h / 4, w / 4)
		de4 = self.dec_layer4(de3, en1)
		# (bs, 64, h / 2, w / 2)
		de5 = self.dec_layer5(de4, None)
		# (bs, 64, h, w)

		out = self.last_conv(de5)
		# (bs, oc, h, w)

		return out




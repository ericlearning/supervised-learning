import torch
import torch.nn as nn
import pretrainedmodels
from torchvision.models import resnet18, resnet34, squeezenet1_1

class Network_18(nn.Module):
	def __init__(self, num_classes):
		super(Network_18, self).__init__()
		pretrained_model = resnet18(pretrained = True)

		self.group1 = nn.Sequential(*list(pretrained_model.children())[0:6])
		self.group2 = nn.Sequential(*list(pretrained_model.children())[6:8])
		self.group3 = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			Flatten(),
			nn.Linear(512, num_classes)
		)

	def forward(self, image):
		out = self.group1(image)
		out = self.group2(out)
		out = self.group3(out)
		return out

class Network_34(nn.Module):
	def __init__(self, num_classes):
		super(Network_34, self).__init__()
		pretrained_model = resnet34(pretrained = True)

		self.group1 = nn.Sequential(*list(pretrained_model.children())[0:6])
		self.group2 = nn.Sequential(*list(pretrained_model.children())[6:8])
		self.group3 = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			Flatten(),
			nn.Linear(512, num_classes)
		)

	def forward(self, image):
		out = self.group1(image)
		out = self.group2(out)
		out = self.group3(out)
		return out

class Network_101(nn.Module):
	def __init__(self, num_classes):
		super(Network_101, self).__init__()
		model_name = 'resnext101_64x4d'
		model_prev = pretrainedmodels.__dict__[model_name](num_classes = 1000, pretrained = 'imagenet')

		model_after = [*list(list(model_prev.children())[0].children())[0:4],
				 *list(list(list(model_prev.children())[0].children())[4].children()),
				 *list(list(list(model_prev.children())[0].children())[5].children()),
				 *list(list(list(model_prev.children())[0].children())[6].children()),
				 *list(list(list(model_prev.children())[0].children())[7].children()),
				 nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(2048, num_classes)]

		self.group1 = nn.Sequential(*model_after[0:21])
		self.group2 = nn.Sequential(*model_after[21:37])
		self.group3 = nn.Sequential(*model_after[37:40])
	
	def forward(self, image):
		out = self.group1(image)
		out = self.group2(out)
		out = self.group3(out)
		return out

class Network_Squeeze(nn.Module):
	def __init__(self, num_classes):
		super(Network_Squeeze, self).__init__()
		pretrained_model = squeezenet1_1(pretrained = True)

		self.group1 = nn.Sequential(*list(list(pretrained_model.children())[0])[:8])
		self.group2 = nn.Sequential(*list(list(pretrained_model.children())[0])[8:])
		self.group3 = nn.Sequential(
			nn.Dropout(p = 0.5),
			nn.Conv2d(512, num_classes, 1, 1, 0),
			nn.AdaptiveAvgPool2d(1),
			Flatten()
		)

	def forward(self, image):
		out = self.group1(image)
		out = self.group2(out)
		out = self.group3(out)
		return out

class Classifier_28x28(nn.Module):
	def __init__(self, nc, num_classes):
		super(Classifier_28x28, self).__init__()
		self.model = nn.Sequential(
			nn.Conv2d(nc, 16, 3, 1, 1),
			nn.ReLU(inplace = True),
			nn.BatchNorm2d(16),
			nn.MaxPool2d(2),
			nn.Conv2d(16, 32, 3, 1, 1),
			nn.ReLU(inplace = True),
			nn.BatchNorm2d(32),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, 3, 1, 1),
			nn.ReLU(inplace = True),
			nn.BatchNorm2d(64),
			nn.MaxPool2d(2),
			Flatten(),
			nn.Linear(3*3*64, num_classes)
		)

	def forward(self, image):
		out = self.model(image)
		return out

class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)

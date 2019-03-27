import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from skimage import transform, io
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

class Dataset():
	def __init__(self, train_dir, val_dir, basic_types = None):
		self.train_dir = train_dir
		self.val_dir = val_dir
		self.basic_types = basic_types

	def get_loader(self, sz, bs, get_size = False, get_class_names = False, get_each_class_size = False, data_transforms = None):
		if(self.basic_types == None):
			if(data_transforms is None):
				data_transforms = {
					'train' : transforms.Compose([
						transforms.RandomResizedCrop(sz),
						transforms.RandomHorizontalFlip(),
						transforms.ToTensor(),
						transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
					]),
					'val' : transforms.Compose([
						transforms.Resize(int(sz*1.2)),
						transforms.CenterCrop(sz),
						transforms.ToTensor(),
						transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
					])
				}
				
			train_dataset = datasets.ImageFolder(self.train_dir, data_transforms['train'])
			val_dataset = datasets.ImageFolder(self.val_dir, data_transforms['val'])
			class_names = train_dataset.classes

			train_classes_count = []
			for cur_dir in class_names:
				count = len([file for file in os.listdir(os.path.join(self.train_dir, cur_dir)) if file[0] != '.'])
				train_classes_count.append(count)

			val_classes_count = []
			for cur_dir in class_names:
				count = len([file for file in os.listdir(os.path.join(self.val_dir, cur_dir)) if file[0] != '.'])
				val_classes_count.append(count)
			
			train_loader = DataLoader(train_dataset, batch_size = bs, shuffle = True)
			val_loader = DataLoader(val_dataset, batch_size = bs, shuffle = False)

			train_dataset_size = len(train_dataset)
			val_dataset_size = len(val_dataset)
			sizes = {
				'train_dset_size' : train_dataset_size,
				'val_dset_size' : val_dataset_size
			}
			each_class_size = {
				'train_classes_count' : train_classes_count,
				'val_classes_count' : val_classes_count
			}
			
			returns = (train_loader, val_loader)
			if(get_size):
				returns = returns + (sizes,)
			if(get_class_names):
				returns = returns + (class_names,)
			if(get_each_class_size):
				returns = returns + (each_class_size,)

		if(self.basic_types == 'MNIST'):
			if(data_transforms is None):
				data_transforms = {
					'train' : transforms.Compose([
						transforms.Resize(sz),
						transforms.ToTensor(),
						transforms.Normalize([0.5], [0.5])
					]),
					'val' : transforms.Compose([
						transforms.Resize(sz),
						transforms.ToTensor(),
						transforms.Normalize([0.5], [0.5])
					])
				}
				
			train_dataset = datasets.MNIST(self.train_dir, train = True, download = True, transform = data_transforms['train'])
			val_dataset = datasets.MNIST(self.val_dir, train = False, download = True, transform = data_transforms['val'])
			
			train_loader = DataLoader(train_dataset, batch_size = bs, shuffle = True)
			val_loader = DataLoader(val_dataset, batch_size = bs, shuffle = False)

			train_dataset_size = len(train_dataset)
			val_dataset_size = len(val_dataset)
			sizes = {
				'train_dset_size' : train_dataset_size,
				'val_dset_size' : val_dataset_size
			}
			
			returns = (train_loader, val_loader)
			if(get_size):
				returns = returns + (sizes,)

		elif(self.basic_types == 'CIFAR'):
			if(data_transforms is None):
				data_transforms = {
					'train' : transforms.Compose([
						transforms.Resize(sz),
						transforms.ToTensor(),
						transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
					]),
					'val' : transforms.Compose([
						transforms.Resize(sz),
						transforms.ToTensor(),
						transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
					])
				}
				
			train_dataset = datasets.CIFAR10(self.train_dir, train = True, download = True, transform = data_transforms['train'])
			val_dataset = datasets.CIFAR10(self.val_dir, train = False, download = True, transform = data_transforms['val'])
			
			train_loader = DataLoader(train_dataset, batch_size = bs, shuffle = True)
			val_loader = DataLoader(val_dataset, batch_size = bs, shuffle = False)

			train_dataset_size = len(train_dataset)
			val_dataset_size = len(val_dataset)
			sizes = {
				'train_dset_size' : train_dataset_size,
				'val_dset_size' : val_dataset_size
			}
			
			returns = (train_loader, val_loader)
			if(get_size):
				returns = returns + (sizes,)

		elif(self.basic_types == 'Segmentation'):


		return returns

class Segmentation_Dataset():
	def __init__(self, input_dir, target_dir, input_transform, target_transform, RGB_list):
		self.input_dir = input_dir
		self.target_dir = target_dir
		self.input_transform = input_transform
		self.target_transform = target_transform
		self.RGB_list = RGB_list

		self.image_name_list = []
		for file in os.listdir(input_dir):
			if(file.endswith('.png') or file.endswith('.jpeg') or file.endswith('.jpg') or file.endswith('.bmp')):
				self.image_name_list.append(file)

		def __len__(self):
			return len(self.image_name_list)

		def __getitem__(self, idx):
			input_img = Image.open(os.path.join(self.input_dir, self.image_name_list[idx]))
			target_img = Image.open(os.path.join(self.target_dir, self.image_name_list[idx]))
			target_img = np.array(target_img)
			for i, (name, color) in enumerate(self.RGB_list):
				pos = np.where((target_img[:, :, 0] == color[2]) & (target_img[:, :, 1] == color[1]) & (target_img[:, :, 2] == color[0]))
				target_img[pos] = i
			target_img = target_img[:, :, 0]

			input_img = self.input_transform(input_img)
			target_img = self.target_transform(target_img)

			sample = (input_img, target_img)
			return sample

RGB_list = [
	{'Animal', (64, 128, 64)},
	{'Archway', (192, 0, 128)},
	{'Bicyclist', (0, 128, 192)},
	{'Bridge', (0, 128, 64)},
	{'Building', (128, 0, 0)},
	{'Car', (64, 0, 128)},
	{'CartLuggagePram', (64, 0, 192)},
	{'Child', (192, 128, 64)},
	{'Column_Pole', (192, 192, 128)},
	{'Fence', (64, 64, 128)},
	{'LaneMkgsDriv', (128, 0, 192)},
	{'LaneMkgsNonDriv', (192, 0, 64)},
	{'Misc_Text', (128, 128, 64)},
	{'MotorcycleScooter', (192, 0, 192)},
	{'OtherMoving', (128, 64, 64)},
	{'ParkingBlock', (64, 192, 128)},
	{'Pedestrian', (64, 64, 0)},
	{'Road', (128, 64, 128)},
	{'RoadShoulder', (128, 128, 192)},
	{'Sidewalk', (0, 0, 192)},
	{'SignSymbol', (192, 128, 128)},
	{'Sky', (128, 128, 128)},
	{'SUVPickupTruck', (64, 128, 192)},
	{'TrafficCone', (0, 0, 64)},
	{'TrafficLight', (0, 64, 64)},
	{'Train', (192, 64, 128)},
	{'Tree', (128, 128, 0)},
	{'Truck_Bus', (192, 128, 192)},
	{'Tunnel', (64, 0, 64)},
	{'VegetationMisc', (192, 192, 0)},
	{'Void', (0, 0, 0)},
	{'Wall', (64, 192, 0)}
]
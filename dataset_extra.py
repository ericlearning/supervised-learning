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
	def __init__(self, train_csv, train_dir, val_csv, val_dir, use_class_zero = True):
		self.train_csv = train_csv
		self.train_dir = train_dir
		self.val_csv = val_csv
		self.val_dir = val_dir
		self.use_class_zero = use_class_zero

	def get_loader(self, sz, bs, get_size = False, get_class_names = False, get_each_class_size = False, data_transforms = None):
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
			
		train_dataset = CSV_Dataset(self.train_csv, self.train_dir, data_transforms['train'], 'img_file', 'class', self.use_class_zero)
		val_dataset = CSV_Dataset(self.val_csv, self.val_dir, data_transforms['val'], 'img_file', 'class', self.use_class_zero)
		class_names = list(set([td[1] for td in train_dataset.data_list]))

		train_classes_count = [0] * len(class_names)
		for d in train_dataset.data_list:
			if(self.use_class_zero):
				train_classes_count[d[1]] += 1
			else:
				train_classes_count[d[1] - 1] += 1

		val_classes_count = [0] * len(class_names)
		for d in val_dataset.data_list:
			if(self.use_class_zero):
				val_classes_count[d[1]] += 1
			else:
				val_classes_count[d[1] - 1] += 1
		
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

		return returns

class CSV_Dataset():
	def __init__(self, input_csv, input_dir, input_transform, image_file_label = 'img_file', class_label = 'class', use_class_zero = True):
		self.input_dir = input_dir
		self.input_transform = input_transform
		self.use_class_zero = use_class_zero

		self.image_name_list = []
		for file in os.listdir(input_dir):
			if(file.endswith('.png') or file.endswith('.jpeg') or file.endswith('.jpg') or file.endswith('.bmp')):
				self.image_name_list.append(file)

		self.data = pd.read_csv(os.path.join(input_dir, input_csv))
		self.data_size = len(self.data)
		self.data_list = []
		for i in range(self.data_size):
			self.data_list.append((os.path.join(input_dir, self.data[image_file_label][i]), self.data[class_label][i]))

	def __len__(self):
		return self.data_size

	def __getitem__(self, idx):
		input_img = Image.open(self.data_list[idx][0]).convert('RGB')
		input_img = self.input_transform(input_img)
		input_class = self.data_list[idx][1]

		if(self.use_class_zero):
			sample = (input_img, input_class)
		else:
			sample = (input_img, input_class - 1)
		return sample

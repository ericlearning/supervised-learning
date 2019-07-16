import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from skimage import transform, io
from transforms_unused import ToTensor_Landmark
from torchvision import transforms

class FaceLandmark_Dataset():
	def __init__(self, input_dir, target_dir, sz, transform, use_image_transformations = True):
		self.input_dir = input_dir
		self.target_dir = target_dir
		self.sz = sz
		self.use_image_transformations = use_image_transformations

		self.landmark_name_list = []
		for file in os.listdir(target_dir):
			if(file.endswith('.txt')):
				self.landmark_name_list.append(os.path.join(target_dir, file))

		# gets PIL as input
		self.transform = transform
		# gets PIL as input, only effects image (should output a PIL of sz*sz)
		self.image_transformations = transforms.Compose([
			transforms.ColorJitter(brightness = 0.5, contrast = 0.3, saturation = 0.05, hue = 0.1)
		])
		# gets PIL as input, effects both image & target
		self.to_tensor = ToTensor_Landmark(self.sz)
		# gets Tensor as input, only effects image
		self.normalization = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		
	def __len__(self):
		return len(self.landmark_name_list)

	def __getitem__(self, idx):
		filename = self.landmark_name_list[idx]
		xs, ys = [], []
		with open(filename) as f:
			for i, l in enumerate(f):
				if(i == 0):
					imagename = l[:-1]
				else:
					splits = l.split(',')
					x, y = splits[0], splits[1]
					xs.append(float(x))
					ys.append(float(y))

		input_img = Image.open(os.path.join(self.input_dir, imagename))
		target_coord = np.array([*xs, *ys])

		input_img, target_coord = self.transform((input_img, target_coord))
		if(self.use_image_transformations):
			input_img = self.image_transformations(input_img)
		input_img, target_coord = self.to_tensor((input_img, target_coord))
		input_img = self.normalization(input_img)
		input_img, target_coord = input_img.float(), target_coord.float()

		sample = (input_img, target_coord)
		return sample

class FaceDetection_Dataset():
	def __init__(self, input_dir, target_csv, sz, transform):
		self.input_dir = input_dir
		self.target_csv = target_csv
		self.sz = sz
		self.df = pd.read_csv(target_csv)
		self.transform = transform

		self.filenames = self.df['fn']
		self.point1 = self.df['p1']
		self.point2 = self.df['p2']

		# gets PIL as input, effects both image & target
		self.to_tensor = ToTensor_Landmark(self.sz)
		# gets Tensor as input, only effects image
		self.normalization = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		filename = self.filenames[idx]
		p1 = self.point1[idx]
		p2 = self.point2[idx]

		input_img = Image.open(os.path.join(self.input_dir, filename))
		point = np.array([float(p1.split(', ')[0][1:]), float(p1.split(', ')[1][:-1]), float(p2.split(', ')[0][1:]), float(p2.split(', ')[1][:-1])])

		input_img, target_coord = self.transform((input_img, point))
		input_img, target_coord = self.to_tensor((input_img, target_coord))
		input_img = self.normalization(input_img)
		input_img, target_coord = input_img.float(), target_coord.float()

		sample = (input_img, target_coord)
		return sample
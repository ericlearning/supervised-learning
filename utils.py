import os
import glob
import torch
import pandas as pd
import seaborn as sn
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import confusion_matrix
from PIL import Image

def set_lr(optimizer, lrs):
	if(len(lrs) == 1):
		for param in optimizer.param_groups:
			param['lr'] = lrs[0]
	else:
		for i, param in enumerate(optimizer.param_groups):
			param['lr'] = lrs[i]

def set_base_lr(optimizer, lrs):
	if(len(lrs) == 1):
		for param in optimizer.param_groups:
			param['initial_lr'] = lrs[0]
	else:
		for i, param in enumerate(optimizer.param_groups):
			param['initial_lr'] = lrs[i]

def get_lr(optimizer):
	optim_param_groups = optimizer.param_groups
	if(len(optim_param_groups) == 1):
		return optim_param_groups[0]['lr']
	else:
		lrs = []
		for param in optim_param_groups:
			lrs.append(param['lr'])
		return lrs

def get_children_groups(model_children, param_places):
	cur_place = 0
	children_groups = []

	for param_place in param_places:
		children_groups.append(model_children[cur_place:param_place])
		cur_place = param_place

	return children_groups

def get_params(children):
	params_use_grad = []
	for child in children:
		for param in child.parameters():
			if(param.requires_grad == True):
				params_use_grad.append(param)

	return params_use_grad

def get_optimizer(model, lrs, param_places):
	model_children = list(model.children())

	# only 1 learning rate
	if(len(lrs) == 1):
		# from the model's childrens, only get the parameters that use grad
		param_use_grad = get_params(model_children)

		# set an Adam optimizer with the params that use grad, and the lr
		optimizer = optim.Adam(param_use_grad, lrs[0])

	# multiple learning rates
	else:
		# from the param_places, get chunks of children from model_children
		# children_groups is a list, and each item will be a list of children
		children_groups = get_children_groups(model_children, param_places)

		# from children_groups, get each of its children group's grad using params
		# param_groups_use_grad is a list, and each item will be a list of params that use grad
		param_groups_use_grad = []

		for children_group in children_groups:
			param_group_use_grad = get_params(children_group)
			param_groups_use_grad.append(param_group_use_grad)

		# zip param_groups_use_grad together with lrs
		# in order to feed in the corresponding lr to a given param_group
		param_groups_use_grad_with_lrs = zip(param_groups_use_grad, lrs)
		optimizer = optim.Adam([{'params' : p, 'lr' : l}
			for p, l in param_groups_use_grad_with_lrs])

	return optimizer

def freeze_until(model, idx):
	for i, child in enumerate(model.children()):
		if(i <= idx):
			for param in child.parameters():
				param.requires_grad = False
		else:
			for param in child.parameters():
				param.requires_grad = True

def histogram_sizes(img_dir, h_lim = None, w_lim = None):
	hs, ws = [], []
	for file in glob.iglob(os.path.join(img_dir, '**/*.*')):
		try:
			with Image.open(file) as im:
				h, w = im.size
				hs.append(h)
				ws.append(w)
		except:
			print('Not an Image file')

	if(h_lim is not None and w_lim is not None):
		hs = [h for h in hs if h<h_lim]
		ws = [w for w in ws if w<w_lim]

	plt.figure('Height')
	plt.hist(hs)

	plt.figure('Width')
	plt.hist(ws)

	plt.show()

	return hs, ws

def plot_confusion_matrix(model, dl, names, classes_count, device, figsize):
	true_label = []
	predicted_label = []

	for batch in dl:
		(images, labels) = batch
		y_real = list(labels.data.cpu().numpy())
		y_pred = list(torch.argmax(model(images.to(device)), dim=1).data.cpu().numpy())
		
		true_label.extend(y_real)
		predicted_label.extend(y_pred)

	cm = confusion_matrix(true_label, predicted_label)
	names_with_cnt = [str(name) + ' : ' + str(cnt) for name, cnt in zip(names, classes_count)]
	df = pd.DataFrame(cm, index = names_with_cnt, columns = names_with_cnt)

	plt.figure(figsize = figsize)
	ax = plt.subplot(111)
	sn.heatmap(df, annot = True, ax = ax, fmt='g')
	
	plt.show()

def freeze_cur_bn(module):
	classname = module.__class__.__name__
	if(classname.find('BatchNorm') != -1):
		module.eval()

def freeze_bn(model):
	model.apply(freeze_cur_bn)
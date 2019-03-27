import os
import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scheduler import CosineAnnealingLR, CyclicLR
from torch.optim.lr_scheduler import StepLR
from utils import freeze_until, get_optimizer, set_lr, set_base_lr, get_lr

class Trainer():
	def __init__(self, model, criterion, train_dl, val_dl, device, sizes, no_acc = False):
		self.model = model
		self.criterion = criterion
		self.train_dl = train_dl
		self.val_dl = val_dl
		self.train_iteration_per_epoch = len(self.train_dl)

		self.train_dset_size = sizes['train_dset_size']
		self.val_dset_size = sizes['val_dset_size']

		self.lr_list = []
		self.device = device
		self.no_acc = no_acc

	def train(self, batch, optimizer, scheduler):
		self.model.train()

		if(scheduler is not None):
			scheduler.step()

		(images, labels) = batch

		images = images.to(self.device)
		labels = labels.to(self.device)

		optimizer.zero_grad()

		with torch.set_grad_enabled(True):
			outputs = self.model(images)
			if(self.no_acc == False):
				_, preds = torch.max(outputs, 1)
			loss = self.criterion(outputs, labels)

			loss.backward()
			optimizer.step()

		cur_loss = loss.item() * labels.size(0)
		cur_corrects = -1
		if(self.no_acc == False):
			cur_corrects = torch.sum(preds == labels).item()

		return cur_loss, cur_corrects

	def evaluate(self, dataloader, dset_size):
		self.model.eval()
		running_loss = 0.0
		running_corrects = 0

		for batch in dataloader:
			(images, labels) = batch

			images = images.to(self.device)
			labels = labels.to(self.device)

			with torch.set_grad_enabled(False):
				outputs = self.model(images)
				if(self.no_acc == False):
					_, preds = torch.max(outputs, 1)
				loss = self.criterion(outputs, labels)

			running_loss += loss.item() * labels.size(0)
			if(self.no_acc == False):
				running_corrects += torch.sum(preds == labels).item()

		eval_loss = running_loss / dset_size
		eval_acc = -1
		if(self.no_acc == False):
			eval_acc = running_corrects / dset_size * 100.0

		return eval_loss, eval_acc

	def train_last_layer(self, lr, cycle_num, cycle_len = None, cycle_mult = None):
		use_sgdr = True
		if(cycle_len == None):
			use_sgdr = False
			cycle_len, cycle_mult = 1, 1

		children_num = len(list(self.model.children()))
		freeze_until(self.model, children_num - 2)

		optimizer = get_optimizer(self.model, [lr], None)

		cur_epoch = 0
		for cycle in range(cycle_num):

			scheduler = None
			if(use_sgdr == True):
				scheduler = CosineAnnealingLR(optimizer, 0, total = self.train_iteration_per_epoch * cycle_len)

			for epoch in range(cycle_len):
				cur_epoch += 1

				running_loss = 0.0
				running_corrects = 0

				for batch in tqdm(self.train_dl):
					cur_lr = get_lr(optimizer)
					self.lr_list.append(cur_lr)

					cur_loss, cur_corrects = self.train(batch, optimizer, scheduler)
					running_loss += cur_loss
					running_corrects += cur_corrects

				epoch_loss_train = running_loss / self.train_dset_size
				epoch_acc_train = -1
				if(self.no_acc == False):
					epoch_acc_train = running_corrects / self.train_dset_size * 100.0
				epoch_loss_val, epoch_acc_val = self.evaluate(self.val_dl, self.val_dset_size)

				print('Epoch : {}, Train Loss : {:.6f}, Train Acc : {:.6f}, Val Loss : {:.6f}, Val Acc : {:.6f}'.format(
						cur_epoch, epoch_loss_train, epoch_acc_train, epoch_loss_val, epoch_acc_val))

			cycle_len *= cycle_mult

		return self.model

	def train_last_layer_clr(self, lr_max, lr_min, epoch_num, div):
		children_num = len(list(self.model.children()))
		freeze_until(self.model, children_num - 2)

		optimizer = get_optimizer(self.model, [lr_min], None)
		scheduler = CyclicLR(optimizer, [lr_max], div, total = self.train_iteration_per_epoch * epoch_num)

		for epoch in range(epoch_num):
			running_loss = 0.0
			running_corrects = 0

			for batch in tqdm(self.train_dl):
				cur_lr = get_lr(optimizer)
				self.lr_list.append(cur_lr)

				cur_loss, cur_corrects = self.train(batch, optimizer, scheduler)
				running_loss += cur_loss
				running_corrects += cur_corrects

			epoch_loss_train = running_loss / self.train_dset_size
			epoch_acc_train = -1
			if(self.no_acc == False):
				epoch_acc_train = running_corrects / self.train_dset_size * 100.0
			epoch_loss_val, epoch_acc_val = self.evaluate(self.val_dl, self.val_dset_size)

			print('Epoch : {}, Train Loss : {:.6f}, Train Acc : {:.6f}, Val Loss : {:.6f}, Val Acc : {:.6f}'.format(
					epoch, epoch_loss_train, epoch_acc_train, epoch_loss_val, epoch_acc_val))

		return self.model

	def train_all_layers(self, lrs, param_places=[1,2,3], cycle_num = 3, cycle_len = None, cycle_mult = None):
		use_sgdr = True
		if(cycle_len == None):
			use_sgdr = False
			cycle_len, cycle_mult = 1, 1

		freeze_until(self.model, -1)

		optimizer = get_optimizer(self.model, lrs, param_places)

		cur_epoch = 0
		for cycle in range(cycle_num):

			scheduler = None
			if(use_sgdr == True):
				scheduler = CosineAnnealingLR(optimizer, 0, total = self.train_iteration_per_epoch * cycle_len)

			for epoch in range(cycle_len):
				cur_epoch += 1

				running_loss = 0.0
				running_corrects = 0

				for batch in tqdm(self.train_dl):
					cur_lr = get_lr(optimizer)
					self.lr_list.append(cur_lr)

					cur_loss, cur_corrects = self.train(batch, optimizer, scheduler)
					running_loss += cur_loss
					running_corrects += cur_corrects

				epoch_loss_train = running_loss / self.train_dset_size
				epoch_acc_train = -1
				if(self.no_acc == False):
					epoch_acc_train = running_corrects / self.train_dset_size * 100.0

				epoch_loss_val, epoch_acc_val = self.evaluate(self.val_dl, self.val_dset_size)

				print('Epoch : {}, Train Loss : {:.6f}, Train Acc : {:.6f}, Val Loss : {:.6f}, Val Acc : {:.6f}'.format(
						cur_epoch, epoch_loss_train, epoch_acc_train, epoch_loss_val, epoch_acc_val))

			cycle_len *= cycle_mult

		return self.model

	# self, lr_max, lr_min, epoch_num, div
	def train_all_layers_clr(self, lrs_max, lrs_min, epoch_num, div, param_places=[1,2,3]):
		freeze_until(self.model, -1)

		optimizer = get_optimizer(self.model, lrs_min, param_places)
		scheduler = CyclicLR(optimizer, lrs_max, div, total = self.train_iteration_per_epoch * epoch_num)

		for epoch in range(epoch_num):
			running_loss = 0.0
			running_corrects = 0

			for batch in tqdm(self.train_dl):
				cur_lr = get_lr(optimizer)
				self.lr_list.append(cur_lr)

				cur_loss, cur_corrects = self.train(batch, optimizer, scheduler)
				running_loss += cur_loss
				running_corrects += cur_corrects

			epoch_loss_train = running_loss / self.train_dset_size
			epoch_acc_train = -1
			if(self.no_acc == False):
				epoch_acc_train = running_corrects / self.train_dset_size * 100.0

			epoch_loss_val, epoch_acc_val = self.evaluate(self.val_dl, self.val_dset_size)

			print('Epoch : {}, Train Loss : {:.6f}, Train Acc : {:.6f}, Val Loss : {:.6f}, Val Acc : {:.6f}'.format(
					epoch, epoch_loss_train, epoch_acc_train, epoch_loss_val, epoch_acc_val))

		return self.model

	def train_all_layers_scratch(self, lr, cycle_num = 3, cycle_len = None, cycle_mult = None):
		use_sgdr = True
		if(cycle_len == None):
			use_sgdr = False
			cycle_len, cycle_mult = 1, 1

		freeze_until(self.model, -1)

		optimizer = get_optimizer(self.model, [lr], None)

		cur_epoch = 0
		for cycle in range(cycle_num):

			scheduler = None
			if(use_sgdr == True):
				scheduler = CosineAnnealingLR(optimizer, 0, total = self.train_iteration_per_epoch * cycle_len)

			for epoch in range(cycle_len):
				cur_epoch += 1

				running_loss = 0.0
				running_corrects = 0

				for batch in tqdm(self.train_dl):
					cur_lr = get_lr(optimizer)
					self.lr_list.append(cur_lr)

					cur_loss, cur_corrects = self.train(batch, optimizer, scheduler)
					running_loss += cur_loss
					running_corrects += cur_corrects

				epoch_loss_train = running_loss / self.train_dset_size
				epoch_acc_train = -1
				if(self.no_acc == False):
					epoch_acc_train = running_corrects / self.train_dset_size * 100.0

				epoch_loss_val, epoch_acc_val = self.evaluate(self.val_dl, self.val_dset_size)

				print('Epoch : {}, Train Loss : {:.6f}, Train Acc : {:.6f}, Val Loss : {:.6f}, Val Acc : {:.6f}'.format(
						cur_epoch, epoch_loss_train, epoch_acc_train, epoch_loss_val, epoch_acc_val))

			cycle_len *= cycle_mult

		return self.model

	def train_all_layers_scratch_clr(self, lr_max, lr_min, epoch_num, div):
		freeze_until(self.model, -1)

		optimizer = get_optimizer(self.model, [lr_min], None)
		scheduler = CyclicLR(optimizer, [lr_max], div, total = self.train_iteration_per_epoch * epoch_num)

		for epoch in range(epoch_num):
			running_loss = 0.0
			running_corrects = 0

			for batch in tqdm(self.train_dl):
				cur_lr = get_lr(optimizer)
				self.lr_list.append(cur_lr)

				cur_loss, cur_corrects = self.train(batch, optimizer, scheduler)
				running_loss += cur_loss
				running_corrects += cur_corrects

			epoch_loss_train = running_loss / self.train_dset_size
			epoch_acc_train = -1
			if(self.no_acc == False):
				epoch_acc_train = running_corrects / self.train_dset_size * 100.0
			epoch_loss_val, epoch_acc_val = self.evaluate(self.val_dl, self.val_dset_size)

			print('Epoch : {}, Train Loss : {:.6f}, Train Acc : {:.6f}, Val Loss : {:.6f}, Val Acc : {:.6f}'.format(
					epoch, epoch_loss_train, epoch_acc_train, epoch_loss_val, epoch_acc_val))

		return self.model

	def set_dl(self, train_dl, val_dl, sizes):
		self.train_dl = train_dl
		self.val_dl = val_dl
		self.train_iteration_per_epoch = len(self.train_dl)

		self.train_dset_size = sizes['train_dset_size']
		self.val_dset_size = sizes['val_dset_size']

	def lr_find(self, lr_start = 1e-6, lr_multiplier = 1.1, max_loss = 3.0, print_value = True):
		init_model_states = copy.deepcopy(self.model.state_dict())

		children_num = len(list(self.model.children()))
		freeze_until(self.model, children_num - 2)

		optimizer = get_optimizer(self.model, [lr_start], None)
		scheduler = StepLR(optimizer, step_size = 1, gamma = lr_multiplier)

		records = []
		lr_found = 0

		while(1):
			for images, labels in self.train_dl:
				# train a single iteration
				self.model.train()
				scheduler.step()
				images = images.to(self.device)
				labels = labels.to(self.device)
				optimizer.zero_grad()

				with torch.set_grad_enabled(True):
					outputs = self.model(images)
					_, preds = torch.max(outputs, 1)
					loss = self.criterion(outputs, labels)

					loss.backward()
					optimizer.step()

				cur_lr = optimizer.param_groups[0]['lr']
				cur_loss = loss.item()
				records.append((cur_lr, cur_loss))

				if(print_value == True):
					print('Learning rate : {} / Loss : {}'.format(cur_lr, cur_loss))

				if(cur_loss > max_loss):
					lr_found = 1
					break

			if(lr_found == 1):
				break
	    
		self.model.load_state_dict(init_model_states)
		return records

	def lr_find_plot(self, records):
		lrs = [e[0] for e in records]
		losses = [e[1] for e in records]

		plt.figure(figsize = (6, 8))
		plt.scatter(lrs, losses)
		plt.xlabel('learning rates')
		plt.ylabel('loss')
		plt.xscale('log')
		plt.yscale('log')

		axes = plt.gca()
		axes.set_xlim([lrs[0], lrs[-1]])
		axes.set_ylim([min(losses) * 0.8, losses[0] * 4])
		plt.show()
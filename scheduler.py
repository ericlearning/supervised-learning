import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingLR(_LRScheduler):
	def __init__(self, optimizer, min_lr, total, last_epoch = -1):
		self.min_lr = min_lr
		self.total = total
		super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		return [self.min_lr + (base_lr - self.min_lr) * (1 + math.cos(math.pi * self.last_epoch / self.total)) / 2.0 for base_lr in self.base_lrs]

class CyclicLR(_LRScheduler):
	def __init__(self, optimizer, max_lrs, div, total, last_epoch = -1):
		self.max_lrs = max_lrs
		self.s1 = int(total * div)
		self.s2 = total - self.s1
		self.total = total
		super(CyclicLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		return [self.get_value(base_lr, self.max_lrs[i], self.last_epoch, self.s1) if(self.last_epoch < self.s1) else self.get_value(self.max_lrs[i], 0, self.last_epoch - self.s1, self.s2) for i, base_lr in enumerate(self.base_lrs)]

	def get_value(self, base_, max_, last_, total_):
		return max_ + (base_ - max_) * (1 + math.cos(math.pi * last_ / total_)) / 2.0
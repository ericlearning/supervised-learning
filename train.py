import os
import torch
import torch.nn as nn
from trainer import Trainer
from dataset import Dataset
from architectures.classification import Network_34
from utils import histogram_sizes, plot_confusion_matrix

data = Dataset('data/flower/train', 'data/flower/val')
model = Network_34(num_classes = 5)
criterion = nn.CrossEntropyLoss()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# histogram_sizes('data/train', 1000, 1000)

sz, bs = 64, 32
trn_dl, val_dl, sizes, names, each_class_size = data.get_loader(sz, bs, get_size = True, get_class_names = True, get_each_class_size = True)
trainer = Trainer(model, criterion, trn_dl, val_dl, device, sizes)

# lr_find_records = trainer.lr_find()
# trainer.lr_find_plot(lr_find_records)

trainer.train_last_layer(0.003, 3, 1, 2)
trainer.train_all_layers([0.003/100, 0.003/10, 0.003], [1, 2, 3], 4, 1, 2)

sz, bs = 128, 32
trn_dl, val_dl, sizes = data.get_loader(sz, bs, get_size = True)
trainer.set_dl(trn_dl, val_dl, sizes)

lr_find_records = trainer.lr_find()
trainer.lr_find_plot(lr_find_records)

trainer.train_last_layer(0.003, 3, 1, 2)
trainer.train_all_layers([0.003/100, 0.003/10, 0.003], [1, 2, 3], 4, 1, 2)

plot_confusion_matrix(model, trn_dl, names, each_class_size['train_classes_count'], device, (30, 25))
plot_confusion_matrix(model, val_dl, names, each_class_size['val_classes_count'], device, (30, 25))

torch.save(model.state_dict(), 'saved/model.pth')
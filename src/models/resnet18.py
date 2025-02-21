import json
from torch.nn import CrossEntropyLoss, Conv2d
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.models import resnet18
from src.attacks.pgd import pgd_linf
from src.data.loader import dataloader_train, dataloader_test, dataloader_val, num_classes, device
from src.training.base_trainer import train_epoch, eval_epoch, train_epoch_adversarial, eval_epoch_adversarial

# HYPERPARAMETERS
batch_size = 128
lr = 0.1

# TRAINING PARAMETERS
num_epochs = 100
weight_decay = 5e-4
momentum = 0.9
milestones = [60, 120, 160]
gamma = 0.2
epsilon = 8/255
alpha = 2/255
num_iter = 7

weight_file = 'pretrained/resnet18.pt'
###########################
# LOAD THE MODEL

model_adv = resnet18()

# CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
model_adv.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model_adv.fc = torch.nn.Linear(model_adv.fc.in_features, num_classes)

pretrained_weights = torch.load(weight_file, weights_only=True)
model_adv.load_state_dict(pretrained_weights)
model_adv = model_adv.to(device)
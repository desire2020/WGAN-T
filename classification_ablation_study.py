import os
import sys
import dataset
import torch
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 64




################################ Euclidean Transformer ######################################
from modeling_euct import EuclideanTransformerPreTrainedModel, EuclideanTransformerModel
########################### End of EuclideanTransformer #########

import transformers
import random
from transformers import XLNetConfig
from tqdm import tqdm
from transformers.optimization import AdamW

class Classifier(EuclideanTransformerPreTrainedModel):
    def __init__(self, config: XLNetConfig):
        super().__init__(config)
        self.config = config
        self.model = EuclideanTransformerModel(config=config)
        self.output_layer = nn.Linear(config.d_model, 10)
        self.mask_reconstruction = nn.Linear(config.d_model, 3 * 8 * 8)

        self.init_weights()
        #####################################################################################

    def forward(self, x, mask_info=None):
        ########################################Your Code####################################
        batch_size, C, H, W = x.shape
        pixel_embeddings, semantic_embeddings = self.model(x)
        semantic_embeddings = pixel_embeddings.mean(dim=1).mean(dim=1)
        if mask_info is None:
            x = self.output_layer(semantic_embeddings.reshape(batch_size, self.config.d_model))
        else:
            return torch.zeros((batch_size,), device=x.device)
        
        #####################################################################################
        return x, None, None

config = XLNetConfig.from_pretrained("xlnet-base-cased")
FastConv = Classifier
random.seed(0)
torch.random.manual_seed(0)
model = FastConv(config).cuda()
loss_fn = torch.nn.CrossEntropyLoss()
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, get_constant_schedule
optimizer = AdamW(lr=5e-5, weight_decay=0.02,
            eps=1e-8, params=model.parameters())
# lr_scheduler = get_constant_schedule(opt)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=1000000, num_warmup_steps=100)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class PatchMaskWrapperDataset(object):
    def __init__(self, base_dataset):
        self.data = base_dataset

    def __getitem__(self, item):
        inputs, labels = self.data[item]
        x, y = random.randint(0, 3), random.randint(0, 3)
        inputs_ = inputs + 0.0

        mask_tgt = inputs[:, 8 * x: 8 * x + 8, 8 * y: 8 * y + 8]
        inputs_[:, 8 * x: 8 * x + 8, 8 * y: 8 * y + 8] = 0.0
        return inputs, labels, inputs_, x, y, mask_tgt

    def __len__(self):
        return self.data.__len__()

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
training_loader = torch.utils.data.DataLoader(PatchMaskWrapperDataset(trainset), batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
validation_loader = torch.utils.data.DataLoader(PatchMaskWrapperDataset(testset), batch_size=batch_size,
                                         shuffle=False, num_workers=2)




def train_one_epoch(epoch_index, mode="both"):
    running_loss = 0.
    running_recons_loss = 0.
    last_loss = 0.
    last_recons_loss = 0.
    len_samples = 0
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    iterator = tqdm((training_loader))
    for i, data in enumerate(iterator):
        # Every data instance is an input + label pair
        inputs, labels, inputs_masked, x, y, mask_tgt = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        inputs_masked = inputs_masked.cuda()
        x = x.cuda()
        y = y.cuda()
        mask_tgt = mask_tgt.cuda()

        # inputs = inputs.permute(0,3,1,2)
        # inputs_masked = inputs_masked.permute(0,3,1,2)
        len_samples += inputs.shape[0]
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        
        # Make predictions for this batch
        # recons_loss = model(inputs_masked, mask_info=(x, y, mask_tgt))
        outputs, _, _ = model(inputs)

        # Get

        # Compute the loss and its gradients
        loss = outputs.log_softmax(dim=-1).gather(dim=-1, index=labels.unsqueeze(dim=-1)).reshape((outputs.shape[0],))
        loss = -loss.mean()
        recons_loss = 0.0 # recons_loss.mean()

        if mode == "both":
            (loss + 50.0 * recons_loss).backward()
        elif mode == "finetune":
            loss.backward()
            
        # Adjust learning weights
        optimizer.step()
        lr_scheduler.step()
        
        # Gather data and report
        running_loss += loss.item() * outputs.shape[0]
        # running_recons_loss += recons_loss.item() * outputs.shape[0]
        if i % 20 == 0:
            last_loss = running_loss / len_samples
            # last_recons_loss = running_recons_loss / len_samples
            # iterator.write('  batch {} loss: {} recons_loss: {} lr: {}'.format(i + 1, last_loss, last_recons_loss, get_lr(optimizer)))
            iterator.write('  batch {} loss: {} lr: {}'.format(i + 1, last_loss, get_lr(optimizer)))
 
    last_loss = running_loss / len_samples
    # last_recons_loss = running_recons_loss / len_samples
    # print('  batch {} loss: {} recons_loss: {} lr: {}'.format(i + 1, last_loss, last_recons_loss, get_lr(optimizer)))
    print('  batch {} loss: {} lr: {}'.format(i + 1, last_loss, get_lr(optimizer)))

    return last_loss, last_recons_loss

EPOCHS = 1000
test_data = dataset.MiniPlaces(
    split='test',
    root_dir = GOOGLE_DRIVE_PATH
)
best_vloss = 0.00

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch))
    
    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss, avg_recons_loss = train_one_epoch(epoch, mode="finetune")
    
    # We don't need gradients on to do reporting
    model.eval()
    
    running_vloss = 0.0
    running_recons_vloss = 0.
    total_vacc = 0.
    len_samples = 0
    # for params in optimizer.param_groups:
    #     params['lr'] *= 0.9
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            inputs, labels, inputs_masked, x, y, mask_tgt = vdata
            inputs = inputs.cuda()
            labels = labels.cuda()
            inputs_masked = inputs_masked.cuda()
            x = x.cuda()
            y = y.cuda()
            mask_tgt = mask_tgt.cuda()

            len_samples += inputs.shape[0]
            # inputs = inputs.permute(0,3,1,2)
            # inputs_masked = inputs_masked.permute(0,3,1,2)
            outputs, _, _ = model(inputs)
            # recons_loss = model(inputs_masked, mask_info=(x, y, mask_tgt))

            loss = loss_fn(outputs, labels)

            outputs = torch.argmax(outputs, -1)
            acc = torch.sum(torch.eq(outputs, labels))
            total_vacc += acc.item()

            running_vloss += loss.item() * inputs.shape[0]
            # running_recons_vloss += recons_loss.mean().item() * inputs.shape[0]

        print ('validation acc is: %f'%(total_vacc/len_samples))
        if (total_vacc/len_samples) > best_vloss:
            best_vloss = total_vacc/len_samples
            # with open("test_UID.txt", "w") as fout:
            #     for i, vdata in tqdm(enumerate(test_data)):
            #         vinputs, _, title = vdata
            #         vinputs = vinputs.cuda()
            #         vinputs = vinputs.unsqueeze(dim=0).permute(0,3,1,2)
            #         voutputs, _, _ = model(vinputs)
            #         print("%s %d" % (title, voutputs.argmax(dim=-1).item()), file=fout)

        print ('best validation acc is: %f'%(best_vloss))

        avg_vloss = running_vloss / (len_samples)
        print('LOSS train {} train-recons {} valid {}'.format(avg_loss, avg_recons_loss, avg_vloss))
        # avg_recons_vloss = running_recons_vloss / (len_samples)
        # print('LOSS train {} train-recons {} valid {} valid-recons {}'.format(avg_loss, avg_recons_loss, avg_vloss, avg_recons_vloss))



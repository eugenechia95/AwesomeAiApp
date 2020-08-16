from scipy.io import loadmat
import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import json


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_class = 5
n_epoch = 200
batch_size = 1
label_train = "CoNSeP/Train/Labels/"
img_train = "CoNSeP/Train/Images/"
label_test = "CoNSeP/Test/Labels/"
img_test = "CoNSeP/Test/Images/"

trans = transforms.Compose([
    transforms.Pad(12),    # given image is 1000x1000, pad it to make it 1024x1024
    transforms.ToTensor(),
    transforms.Normalize([0.790595  , 0.67119867, 0.8091853 ], [0.05493367, 0.05985045, 0.04747279]) 
])
    

class NucleiDataset(Dataset):
    def __init__(self, image_path, mask_path, transform=None):
        self.img_ls = [image_path+i for i in sorted(os.listdir(image_path))]
        self.mask_ls = [ mask_path+i for i in sorted(os.listdir(mask_path))]
        self.transform = transform

    def __len__(self):
        return len(self.img_ls)

    def __getitem__(self, idx):
        img_name = self.img_ls[idx]
        img = Image.open(img_name).convert('RGB')
        img.load()
        mask_name = self.mask_ls[idx]
        mask = loadmat(mask_name)['type_map']
        mask = np.pad(mask, 12)
        mask0 = mask == 0
        mask1 = (mask == 1) 
        mask2 = mask == 2
        mask3 = (mask == 3) + (mask == 4)
        mask4 = (mask == 5) + (mask == 6) + (mask == 7)
        masks = [mask0, mask1, mask2, mask3, mask4]
        mask = np.stack(masks, axis=0).astype('float')
        if self.transform:
            img = self.transform(img)

        return img, mask


def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.790595  , 0.67119867, 0.8091853 ])
    std = np.array([0.05493367, 0.05985045, 0.04747279])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp

######################
# Model Architecture #
######################

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        
        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


###################################################################
# Loss function that combines Binary Cross Entropy with Dice loss #
###################################################################

def weighted_loss(pred,targ,bce_weight=0.5, smooth=1.):
    
    bce = F.binary_cross_entropy_with_logits(pred.squeeze(dim=1), targ)
    
    pred = torch.sigmoid(pred)
    
    pred = pred.contiguous().squeeze(dim=1)  
    targ = targ.contiguous()  

    intersection = (pred * targ).sum(dim=1).sum(dim=1)
    dice = (1 - ((2. * intersection + smooth) / (pred.sum(dim=1).sum(dim=1) + targ.sum(dim=1).sum(dim=1) + smooth)))
    
    loss = bce * bce_weight + dice.mean() * (1 - bce_weight)
    
    return loss


def f1(true, pred):
    inter = (pred * true).sum()
    total = (pred + true).sum()
    return 2 * inter /  (total + 1.0e-8)


#################
# Training Step #
#################
def train_model(model, train_loader, test_loader, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    best_f1 = 0
    
    loss_ls = []
    val_loss_ls = []
    f1_dict = defaultdict(list)
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()
        
        loss = 0
        model.train()
        for inputs,masks in train_loader:
            inputs = inputs.to(device)
            masks = masks.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            epoch_loss = weighted_loss(outputs,masks,bce_weight=0.3)
            epoch_loss.backward()
            optimizer.step()
            
            n_samples = len(inputs)
            loss+=(epoch_loss/n_samples).item()
        
        loss /= len(train_loader)
        print("epoch loss:",loss)
        loss_ls.append(loss)
        
        model.eval()
        val_loss = 0
        avg_f1 = 0
        
        with torch.set_grad_enabled(False):
            f1_scores = defaultdict(list)
            for inputs, masks in test_loader:
                inputs = inputs.to(device)
                masks = masks.to(device).float()
                outputs = model(inputs)
                val_loss += (weighted_loss(outputs,masks,bce_weight=0.3) / len(inputs) ).item()
                pred = torch.sigmoid(outputs).to('cpu').detach().numpy()[0]
                for i in range(1,5):
                    threshold = 0.5
                    pred_i = pred[i]
                    pred_i[pred_i >= threshold] = 1
                    pred_i[pred_i < threshold] = 0

                    mask = masks.to('cpu').detach().numpy().astype(int)[0][i]
                    f1_scores[f'Class {i}'] .append( f1(mask,pred_i) )
            inputs, masks = next(iter(test_loader))
            save_image(model, inputs, masks, epoch)

                    
        for c in f1_scores:
            f1_c = np.mean(f1_scores[c])
            print(c, f1_c)
            f1_dict[c].append(f1_c)
            avg_f1 += (f1_c/4)
            
        val_loss /= len(test_loader)
        val_loss_ls.append(val_loss)
        print("Val Loss: ", val_loss)
        
        
        if avg_f1 > best_f1:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_f1 = avg_f1
            print(f'new lowest average f1 found {avg_f1}')
        
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60),"\n")
    
    return loss_ls, val_loss_ls, best_model_wts, f1_dict


def save_image(model, inputs, masks, epoch):
    fig, axs = plt.subplots(3, figsize=(20,20))
    axs[0].imshow(reverse_transform(inputs[0].cpu()))
    axs[0].set_title('Input')
    outputs = model(inputs.to(device))

    mask = masks.cpu().numpy()[0]
    target = np.zeros(mask[0].shape)
    for i in range(5):
        mask_i = mask[i]
        target += i * mask_i
    axs[1].imshow(target, cmap='jet', vmax=np.max(target), vmin=np.min(target))
    axs[1].set_title('Target')

    threshold = 0.8
    pred = outputs.to('cpu').detach().numpy()[0]
    output = np.zeros(pred[0].shape)
    for i in range(5):
        pred_i = pred[i]
        pred_i[pred_i >= threshold] = 1
        pred_i[pred_i < threshold] = 0
        output += i * pred_i
    axs[2].imshow(output,  cmap='jet', vmax=np.max(target), vmin=np.min(target))
    axs[2].set_title('Predicted')
    plt.savefig(f'seg_results/seg_epoch{epoch}.png')

    
def main():    
    os.makedirs('seg_results', exist_ok=True)
    
    train_set = NucleiDataset(img_train,label_train, transform = trans)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_set = NucleiDataset(img_test,label_test, transform = trans)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    model = ResNetUNet(num_class).to(device)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    loss_ls, val_loss_ls, best_model_wts, f1_dict = train_model(model, train_loader, test_loader, optimizer_ft, exp_lr_scheduler, num_epochs=n_epoch)

    torch.save(best_model_wts, f"./best_weight_bs_{batch_size}.pt")

    with open(f'loss_val_train_bs_{batch_size}.json', 'w') as file:
        json.dump({'val': val_loss_ls, "train": loss_ls}, file, indent=4)

    with open(f'f1_bs_{batch_size}.json', 'w') as file:
        json.dump(f1_dict, file, indent=4)
        
    x = [i for i in range(n_epoch)]
    plt.figure(figsize=(20,10))
    plt.plot(x, val_loss_ls, label='Validation Loss')
    plt.plot(x, loss_ls, label='Train Loss')
    plt.legend()
    plt.savefig(f'loss_bs_{batch_size}.png')
    
    x = [i for i in range(n_epoch)]
    plt.figure(figsize=(20,10))
    plt.plot(x, f1_dict['Class 1'][:], label='Class 1 other')
    plt.plot(x, f1_dict['Class 2'][:], label='Class 2 inflammatory')
    plt.plot(x, f1_dict['Class 3'][:], label='Class 3 epithelial nuclei')
    plt.plot(x, f1_dict['Class 4'][:], label='Class 4 spindel-shaped')
    plt.plot(x, [(y1 + y2 + y3 +y4)/4 for y1, y2, y3, y4 in zip(f1_dict['Class 1'], f1_dict['Class 2'], f1_dict['Class 3'], f1_dict['Class 4'])], linestyle='-', linewidth=3, label='Average F1 Score')
    plt.legend()
    plt.savefig(f'f1_bs_{batch_size}.png')


    
if __name__ == "__main__":
    main()

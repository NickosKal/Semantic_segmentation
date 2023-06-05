import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from time import time
from PIL import Image

data_path = 'WARWICK'
train_data_path = os.path.join(data_path, 'Train')
test_data_path = os.path.join(data_path, 'Test')

train_files = glob.glob(train_data_path + os.sep + '*')
test_files = glob.glob(test_data_path + os.sep + '*')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_data(files):
    image_data = {}
    label_data = {}

    for file in files:
        name = file.split(os.sep)[-1][:-4]
        ide = int(name[6:])
        if name[:5] == 'image':
            image_data[ide] = file
        else:
            label_data[ide] = file
    return image_data, label_data


train_images, train_labels = generate_data(train_files)
test_images, test_labels = generate_data(test_files)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # nn.Dropout(),
        )

    def forward(self, x):
        return self.conv(x)


class SegmentationCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64]):
        super(SegmentationCNN, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))

            self.ups.append(DoubleConv(feature*2, feature))

        self.prefinal = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, image):
        global x
        skip_connections = []
        for down in self.downs:
            x = down(image)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.prefinal(x)
        skip_connections = skip_connections[::-1]

        for index in range(0, len(self.ups), 2):
            x = self.ups[index](x)
            skip_connections = skip_connections[index//2]

            if x.shape != skip_connections.shape:
                x = F.resize(x, size=skip_connections.shape[2:])

            concat_skip = torch.cat((skip_connections, x), dim=1)
            x = self.ups[index+1](concat_skip)

        return self.final_conv(x)


class WarwickDataset(Dataset):

    def __init__(self, images, labels, transform=None):
        self.image_files = images
        self.label_files = labels
        self.transform = transform
        self.label_transform = transforms.Compose([transforms.ToTensor()])
        self.mapper = {}
        for i, key in enumerate(self.image_files.keys()):
            self.mapper[i] = key

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        image = Image.open(self.image_files[self.mapper[item]]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.label_transform(Image.open(self.label_files[self.mapper[item]]))
        label = label.unsqueeze(0)
        label = F.interpolate(label, size=(128, 128), mode='bilinear', align_corners=False)

        return image, label


train_transform = transforms.Compose([transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])

train_dset = WarwickDataset(train_images, train_labels, train_transform)
test_dset = WarwickDataset(test_images, test_labels, test_transform)

train_loader = DataLoader(train_dset, batch_size=8, shuffle=True, drop_last=True)

test_loader = DataLoader(test_dset, batch_size=1, shuffle=True, drop_last=False)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    @staticmethod
    def forward(y_pred, y_true):
        smooth = 1e-5

        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        intersection = torch.sum(y_pred * y_true)
        union = torch.sum(y_pred) + torch.sum(y_true)

        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice_score

        return dice_loss


dice_loss = DiceLoss()


def calculate_dice(test_loader, net, device):
    dice_value = 0
    net.eval()
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        output = net(images).squeeze(0)
        output = torch.sigmoid(output)
        predicted = (output > 0.5).float()
        dice_value += 1 - dice_loss(predicted.detach().cpu().long(), labels.squeeze(0).cpu().long())
    dice_value = dice_value / len(test_loader)
    return dice_value


# Set up the model and optimizer
model = SegmentationCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train the model
num_epochs = 100
train_loss, train_acc = [], []
dice_coefficient = []

start = time()
for epoch in range(num_epochs):
    # Train the model for one epoch
    running_train_loss, running_train_acc = 0.0, 0.0
    total_dice = 0
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        batch = len(images)
        outputs = outputs.view(batch, -1)
        labels = labels.view(batch, -1)
        loss = criterion(outputs, labels)
        running_train_loss += loss.item()
        loss.backward()
        optimizer.step()

        predicted_labels = outputs > 0.5
        correct_prediction = (predicted_labels == labels).sum().item()
        batch_acc = correct_prediction / (labels.numel())
        running_train_acc += batch_acc

    epoch_train_loss = running_train_loss / len(train_dset)
    epoch_train_acc = running_train_acc / len(train_dset)

    total_dice = calculate_dice(test_loader, model, device)

    train_loss.append(epoch_train_loss)
    train_acc.append(epoch_train_acc)
    dice_coefficient.append(total_dice)

    if epoch % 5 == 0:
        print(f"Epoch: {epoch}, Loss: {running_train_loss / len(train_loader)}, Dice Value: {total_dice.item()}")

stop = time()

# # Plot the train and test losses
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
# ax[0].plot(train_loss)
# ax[0].set_xlabel('Epoch')
# ax[0].set_ylabel('Loss')
# ax[0].set_title('Train Loss')
# ax[0].legend()
#
# # Plot the train and test accuracies
# ax[1].plot(dice_coefficient)
# ax[1].set_xlabel('Epoch')
# ax[1].set_ylabel('Dice value')
# ax[1].set_title('Dice Score')
# ax[1].legend()
#
# avg_dice = sum(dice_coefficient) / len(dice_coefficient)
# print(f"Average Dice coefficient in test set: {avg_dice}")
# print(f"time passed: {stop - start} seconds")

image, label = next(iter(test_loader))
image, label = image.to(device), label.to(device)
ide = 0


plt.subplot(1, 3, 1)
plt.imshow(image[ide].permute(1, 2, 0).cpu().numpy())
plt.title("Image")

output = model(image)[ide].squeeze(0)
output = F.sigmoid(output)
output = (output > 0.5).float()

plt.subplot(1, 3, 2)
plt.imshow(output.squeeze(0).detach().cpu().numpy(), cmap='gray')
plt.title("Predicted mask")

plt.subplot(1, 3, 3)
plt.imshow(label[ide].squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap='gray')
plt.title("Actual mask")
plt.show()

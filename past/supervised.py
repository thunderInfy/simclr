import numpy as np
import torch
from torchvision import transforms as T
# from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# import torch.nn.functional as F

import os
from PIL import Image
from PIL import ImageOps
# import requests
# import zipfile
# import io
import random
from torchvision.models import resnet18
from collections import OrderedDict

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
tsne = TSNE()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_folder = 'imagenet-5-categories'

names_train = random.sample(os.listdir(root_folder + '/train'), len(os.listdir(root_folder + '/train')))
names_test = random.sample(os.listdir(root_folder + '/test'), len(os.listdir(root_folder + '/test')))

mapping = {'car': 0, 'dog': 1, 'elephant': 2, 'cat': 3, 'airplane': 4}
inverse_mapping = ['car','dog','elephant','cat','airplane']
labels_train = [mapping[x.split('_')[0]] for x in names_train]
labels_test = [mapping[x.split('_')[0]] for x in names_test]


class MyDataset(Dataset):
    def __init__(self, root_dir, filenames, labels):
        self.root_dir = root_dir
        self.file_names = filenames
        self.labels = labels

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.file_names[idx])
        image = Image.open(img_name)
        label = self.labels[idx]
        image = T.Resize((224, 224))(image)
        # image = T.RandomHorizontalFlip()(image)
        image = T.ToTensor()(image)
        image = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)
        sample = {'image': image, 'label': label}
        return sample


training_dataset = MyDataset(root_folder + '/train', names_train, labels_train)
testing_dataset = MyDataset(root_folder + '/test', names_test, labels_test)
training_dataloader = DataLoader(training_dataset, batch_size=250, shuffle=True, num_workers=1)
testing_dataloader = DataLoader(testing_dataset, batch_size=250, shuffle=True, num_workers=1)

resnet = resnet18(pretrained=False)

classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(resnet.fc.in_features, 5))
]))

resnet.fc = classifier

resnet.to(device)

# print(summary(resnet, (3, 224, 224)))

losses_train = []
losses_test = []
acc_train = []
acc_test = []

if not os.path.exists('results'):
    os.makedirs('results')

optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)
num_epochs = 1

if(os.path.isfile("results/model.pth")):
    resnet.load_state_dict(torch.load("results/model.pth"))
    resnet.train()
    optimizer.load_state_dict(torch.load("results/optimizer.pth"))

    for g in optimizer.param_groups:
      g['lr'] = 0.0005

    temp = np.load("results/losses_train_file.npz")
    losses_train = list(temp['arr_0'])
    temp = np.load("results/losses_test_file.npz")
    losses_test = list(temp['arr_0'])
    temp = np.load("results/acc_train_file.npz")
    acc_train = list(temp['arr_0'])
    temp = np.load("results/acc_test_file.npz")
    acc_test = list(temp['arr_0'])


def get_mean_of_list(L):
    return sum(L) / len(L)


max_test_acc = 0.788
training = False

for epoch in range(num_epochs):

    if training:
        epoch_losses_train = []
        epoch_acc_train_num = 0.0
        epoch_acc_train_den = 0.0
        for (_, sample_batched) in enumerate(training_dataloader):
            optimizer.zero_grad()
            x = sample_batched['image']
            y_actual = sample_batched['label']

            x = x.to(device)
            y_actual = y_actual.to(device)

            y_predicted = resnet(x)
            loss = nn.CrossEntropyLoss()(y_predicted, y_actual)
            epoch_losses_train.append(loss.data.item())
            loss.backward()
            optimizer.step()

            pred = np.argmax(y_predicted.cpu().data, axis=1)
            actual = y_actual.cpu().data
            epoch_acc_train_num += (actual == pred).sum().item()
            epoch_acc_train_den += len(actual)

        losses_train.append(get_mean_of_list(epoch_losses_train))
        acc_train.append(epoch_acc_train_num / epoch_acc_train_den)
    else:
        resnet.eval()

    epoch_losses_test = []
    epoch_acc_test_num = 0.0
    epoch_acc_test_den = 0.0

    for (_, sample_batched) in enumerate(testing_dataloader):
        x = sample_batched['image']
        y_actual = sample_batched['label']

        x = x.to(device)
        y_actual = y_actual.to(device)

        y_predicted = resnet(x)
        loss = nn.CrossEntropyLoss()(y_predicted, y_actual)
        epoch_losses_test.append(loss.data.item())

        pred = np.argmax(y_predicted.cpu().data, axis=1)
        actual = y_actual.cpu().data
        epoch_acc_test_num += (actual == pred).sum().item()
        epoch_acc_test_den += len(actual)

    test_acc = epoch_acc_test_num / epoch_acc_test_den
    print(test_acc)

    if training:
        losses_test.append(get_mean_of_list(epoch_losses_test))
        acc_test.append(test_acc)
        fig = plt.figure(figsize=(10, 10))
        sns.set_style('darkgrid')
        plt.plot(losses_train)
        plt.plot(losses_test)
        plt.legend(['Training Losses', 'Testing Losses'])
        plt.savefig('losses.png')
        plt.close()

        fig = plt.figure(figsize=(10, 10))
        sns.set_style('darkgrid')
        plt.plot(acc_train)
        plt.plot(acc_test)
        plt.legend(['Training Accuracy', 'Testing Accuracy'])
        plt.savefig('accuracy.png')
        plt.close()

        print("Epoch completed")

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            torch.save(resnet.state_dict(), 'results/model.pth')
            torch.save(optimizer.state_dict(), 'results/optimizer.pth')

        np.savez("results/losses_train_file", np.array(losses_train))
        np.savez("results/losses_test_file", np.array(losses_test))
        np.savez("results/acc_train_file", np.array(acc_train))
        np.savez("results/acc_test_file", np.array(acc_test))


def deprocess_and_show(img_tensor):
    return T.Compose([
        T.Normalize((0, 0, 0), (2, 2, 2)),
        T.Normalize((-0.5, -0.5, -0.5), (1, 1, 1)),
        T.ToPILImage()
    ])(img_tensor)


MIS = False

if MIS:

    counter = 0

    if not os.path.exists('mispredictions'):
        os.makedirs('mispredictions')

    for (_, sample_batched) in enumerate(testing_dataloader):
        x = sample_batched['image']
        y_actual = sample_batched['label']

        x = x.to(device)
        y_actual = y_actual.to(device)

        y_predicted = resnet(x)

        pred = np.argmax(y_predicted.cpu().data, axis=1)
        actual = y_actual.cpu().data

        res = (pred == actual)
        mispredicted_imgs = x[res==False].cpu().data
        mispredicted_labels = [inverse_mapping[x] for x in pred[res==False]]
        correct_labels = [inverse_mapping[x] for x in actual[res==False]]

        for i in range(len(mispredicted_labels)):
            pil_img = deprocess_and_show(mispredicted_imgs[i])
            img_name = mispredicted_labels[i] + str(counter) + correct_labels[i] + '.jpg'
            counter += 1
            pil_img.save("mispredictions/"+img_name)

SAL = True

if SAL:
    if not os.path.exists('saliency'):
        os.makedirs('saliency')

    counter = 0

    resnet.eval()

    saliency_testing_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=True, num_workers=1)

    for (_, sample_batched) in enumerate(saliency_testing_dataloader):
        x = sample_batched['image']
        # y_actual = sample_batched['label']

        x = x.to(device)
        # y_actual = y_actual.to(device)

        x.requires_grad_()

        y_predicted = resnet(x)

        max_score, _ = torch.max(y_predicted, 1)

        if x.grad is not None:
            x.grad.zero_()

        max_score.backward()
        saliency, _ = torch.max(x.grad.data.abs(), dim=1)
        inp_img = deprocess_and_show(x[0].cpu().data)
        sal_img = ImageOps.colorize(T.ToPILImage()(saliency[0].cpu()*100), black ="black", white ="red")
        images = [inp_img, sal_img]
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        new_im.save('saliency/' + str(counter) + '.png')

        counter += 1

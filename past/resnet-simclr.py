import numpy as np

import torch
from torchvision import transforms as T
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

import os
from PIL import Image
from collections import OrderedDict

import requests
import zipfile
import io
import random

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
tsne = TSNE()


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = T.RandomApply([color_jitter], p=0.8)
    rnd_gray = T.RandomGrayscale(p=0.2)
    color_distort = T.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_folder = 'imagenet-5-categories'

random.seed(0)

names_train_10_percent = random.sample(os.listdir(root_folder + '/train'), len(os.listdir(root_folder + '/train')) // 10)

names_train = random.sample(os.listdir(root_folder + '/train'), len(os.listdir(root_folder + '/train')))
names_test = random.sample(os.listdir(root_folder + '/test'), len(os.listdir(root_folder + '/test')))


mapping = {'car': 0, 'dog': 1, 'elephant': 2, 'cat': 3, 'airplane': 4}
inverse_mapping = ['car', 'dog', 'elephant', 'cat', 'airplane']
labels_train = [mapping[x.split('_')[0]] for x in names_train]
labels_test = [mapping[x.split('_')[0]] for x in names_test]
labels_train_10_percent = [mapping[x.split('_')[0]] for x in names_train_10_percent]

class MyDataset(Dataset):
    def __init__(self, root_dir, filenames, labels, mutation=False):
        self.root_dir = root_dir
        self.file_names = filenames
        self.labels = labels
        self.mutation = mutation

    def __len__(self):
        return len(self.file_names)

    def tensorify(self, img):
        res = T.ToTensor()(img)
        res = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(res)
        return res

    def mutate_image(self, img):
        res = T.RandomResizedCrop(224)(img)
        res = get_color_distortion(1)(res)
        return res

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.file_names[idx])
        image = Image.open(img_name)
        label = self.labels[idx]
        image = T.Resize((250, 250))(image)

        if self.mutation:
            image1 = self.mutate_image(image)
            image1 = self.tensorify(image1)
            image2 = self.mutate_image(image)
            image2 = self.tensorify(image2)
            sample = {'image1': image1, 'image2': image2, 'label': label}
        else:
            image = T.Resize((224, 224))(image)
            image = self.tensorify(image)
            sample = {'image': image, 'label': label}

        return sample


training_dataset_mutated = MyDataset(root_folder + '/train', names_train, labels_train, mutation=True)
training_dataset = MyDataset(root_folder + '/train', names_train_10_percent, labels_train_10_percent, mutation=False)
testing_dataset = MyDataset(root_folder + '/test', names_test, labels_test, mutation=False)

dataloader_training_dataset_mutated = DataLoader(training_dataset_mutated, batch_size=250, shuffle=True, num_workers=2)
dataloader_training_dataset = DataLoader(training_dataset, batch_size=125, shuffle=True, num_workers=2)
dataloader_testing_dataset = DataLoader(testing_dataset, batch_size=250, shuffle=True, num_workers=2)

resnet = resnet18(pretrained=False)


classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(resnet.fc.in_features, 100)),
    ('added_relu1', nn.ReLU(inplace=True)),
    ('fc2', nn.Linear(100, 50)),
    ('added_relu2', nn.ReLU(inplace=True)),
    ('fc3', nn.Linear(50, 25))
]))

resnet.fc = classifier

# summary(resnet, (3, 224, 224))

resnet.to(device)

tau = 0.1


def loss_function(a, b):
    a_norm = torch.norm(a, dim=1).reshape(-1, 1)
    a_cap = torch.div(a, a_norm)
    b_norm = torch.norm(b, dim=1).reshape(-1, 1)
    b_cap = torch.div(b, b_norm)
    a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)
    a_cap_b_cap_transpose = torch.t(a_cap_b_cap)
    b_cap_a_cap = torch.cat([b_cap, a_cap], dim=0)
    sim = torch.mm(a_cap_b_cap, a_cap_b_cap_transpose)
    sim_by_tau = torch.div(sim, tau)
    exp_sim_by_tau = torch.exp(sim_by_tau)
    sum_of_rows = torch.sum(exp_sim_by_tau, dim=1)
    exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)
    numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()(a_cap_b_cap, b_cap_a_cap), tau))
    denominators = sum_of_rows - exp_sim_by_tau_diag
    num_by_den = torch.div(numerators, denominators)
    neglog_num_by_den = -torch.log(num_by_den)
    return torch.mean(neglog_num_by_den)


losses_train = []
num_epochs = 101
optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)

if not os.path.exists('results'):
    os.makedirs('results')

if(os.path.isfile("results/model.pth")):
    resnet.load_state_dict(torch.load("results/model.pth"))
    optimizer.load_state_dict(torch.load("results/optimizer.pth"))

    for param_group in optimizer.param_groups:
        param_group['weight_decay'] = 1e-6
        param_group['lr'] = 0.000003

    temp = np.load("results/lossesfile.npz")
    losses_train = list(temp['arr_0'])

TRAINING = False


def get_mean_of_list(L):
    return sum(L) / len(L)


if TRAINING:
    resnet.train()

    for epoch in range(num_epochs):

        epoch_losses_train = []

        for (_, sample_batched) in enumerate(dataloader_training_dataset_mutated):
            optimizer.zero_grad()
            x1 = sample_batched['image1']
            x2 = sample_batched['image2']

            x1 = x1.to(device)
            x2 = x2.to(device)

            y1 = resnet(x1)
            y2 = resnet(x2)

            loss = loss_function(y1, y2)
            epoch_losses_train.append(loss.cpu().data.item())
            loss.backward()
            optimizer.step()

        losses_train.append(get_mean_of_list(epoch_losses_train))

        fig = plt.figure(figsize=(10, 10))
        sns.set_style('darkgrid')
        plt.plot(losses_train)
        plt.legend(['Training Losses'])
        plt.savefig('losses.png')
        plt.close()

        torch.save(resnet.state_dict(), 'results/model.pth')
        torch.save(optimizer.state_dict(), 'results/optimizer.pth')
        np.savez("results/lossesfile", np.array(losses_train))

def plot_vecs_n_labels(v,labels,fname):
    fig = plt.figure(figsize = (10, 10))
    plt.axis('off')
    sns.set_style("darkgrid")
    sns.scatterplot(v[:,0], v[:,1], hue=labels, legend='full', palette=sns.color_palette("bright", 5))
    plt.legend(['car', 'dog', 'elephant','cat','airplane'])
    plt.savefig(fname)
    plt.close()

TSNEVIS = False

if TSNEVIS:
    resnet.eval()

    for (_, sample_batched) in enumerate(dataloader_training_dataset):
        x = sample_batched['image']
        x = x.to(device)
        y = resnet(x)
        y_tsne = tsne.fit_transform(y.cpu().data)
        labels = sample_batched['label']
        plot_vecs_n_labels(y_tsne,labels,'tsne_train_last_layer.png')
        x = None
        y = None
        y_tsne = None
        sample_batched = None

    for (_, sample_batched) in enumerate(dataloader_testing_dataset):
        x = sample_batched['image']
        x = x.to(device)
        y = resnet(x)
        y_tsne = tsne.fit_transform(y.cpu().data)
        labels = sample_batched['label']
        plot_vecs_n_labels(y_tsne,labels,'tsne_test_last_layer.png')
        x = None
        y = None
        y_tsne = None
        sample_batched = None

resnet.fc = nn.Sequential(*list(resnet.fc.children())[:-2])

LINEAR = True


class LinearNet(nn.Module):

    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = torch.nn.Linear(50, 5)

    def forward(self, x):
        x = self.fc1(x)
        return(x)



if LINEAR:

    if not os.path.exists('linear'):
        os.makedirs('linear')

    linear_classifier = LinearNet()

    linear_classifier.to(device)

    linear_optimizer = optim.SGD(linear_classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-6)

    num_epochs_linear = 1

    LINEAR_TRAINING = False

    losses_train_linear = []
    acc_train_linear = []
    losses_test_linear = []
    acc_test_linear = []

    max_test_acc = 0

    if(os.path.isfile("linear/model.pth")):
        linear_classifier.load_state_dict(torch.load("linear/model.pth"))
        linear_optimizer.load_state_dict(torch.load("linear/optimizer.pth"))

        # for g in linear_optimizer.param_groups:
        #   g['lr'] = 0.0005

        temp = np.load("linear/linear_losses_train_file.npz")
        losses_train_linear = list(temp['arr_0'])
        temp = np.load("linear/linear_losses_test_file.npz")
        losses_test_linear = list(temp['arr_0'])
        temp = np.load("linear/linear_acc_train_file.npz")
        acc_train_linear = list(temp['arr_0'])
        temp = np.load("linear/linear_acc_test_file.npz")
        acc_test_linear = list(temp['arr_0'])

    for epoch in range(num_epochs_linear):

        if LINEAR_TRAINING:
            linear_classifier.train()
            epoch_losses_train_linear = []
            epoch_acc_train_num_linear = 0.0
            epoch_acc_train_den_linear = 0.0

            for (_, sample_batched) in enumerate(dataloader_training_dataset):
                x = sample_batched['image']
                y_actual = sample_batched['label']

                x = x.to(device)
                y_actual  = y_actual.to(device)

                y_intermediate = resnet(x)

                linear_optimizer.zero_grad()
                y_predicted = linear_classifier(y_intermediate)
                loss = nn.CrossEntropyLoss()(y_predicted, y_actual)
                epoch_losses_train_linear.append(loss.data.item())
                loss.backward()
                linear_optimizer.step()

                pred = np.argmax(y_predicted.cpu().data, axis=1)
                actual = y_actual.cpu().data
                epoch_acc_train_num_linear += (actual == pred).sum().item()
                epoch_acc_train_den_linear += len(actual)

                x = None
                y_intermediate = None
                y_predicted = None
                sample_batched = None
            losses_train_linear.append(get_mean_of_list(epoch_losses_train_linear))
            acc_train_linear.append(epoch_acc_train_num_linear / epoch_acc_train_den_linear)
        else:
            linear_classifier.eval()

        epoch_losses_test_linear = []
        epoch_acc_test_num_linear = 0.0
        epoch_acc_test_den_linear = 0.0

        for (_, sample_batched) in enumerate(dataloader_testing_dataset):
            x = sample_batched['image']
            y_actual = sample_batched['label']

            x = x.to(device)
            y_actual  = y_actual.to(device)

            y_intermediate = resnet(x)

            y_predicted = linear_classifier(y_intermediate)
            loss = nn.CrossEntropyLoss()(y_predicted, y_actual)
            epoch_losses_test_linear.append(loss.data.item())

            pred = np.argmax(y_predicted.cpu().data, axis=1)
            actual = y_actual.cpu().data
            epoch_acc_test_num_linear += (actual == pred).sum().item()
            epoch_acc_test_den_linear += len(actual)

        test_acc = epoch_acc_test_num_linear / epoch_acc_test_den_linear
        print(test_acc)

        if LINEAR_TRAINING:
            losses_test_linear.append(get_mean_of_list(epoch_losses_test_linear))
            acc_test_linear.append(epoch_acc_test_num_linear / epoch_acc_test_den_linear)

            fig = plt.figure(figsize=(10, 10))
            sns.set_style('darkgrid')
            plt.plot(losses_train_linear)
            plt.plot(losses_test_linear)
            plt.legend(['Training Losses', 'Testing Losses'])
            plt.savefig('linear/losses.png')
            plt.close()

            fig = plt.figure(figsize=(10, 10))
            sns.set_style('darkgrid')
            plt.plot(acc_train_linear)
            plt.plot(acc_test_linear)
            plt.legend(['Training Accuracy', 'Testing Accuracy'])
            plt.savefig('linear/accuracy.png')
            plt.close()

            print("Epoch completed")

            if test_acc > max_test_acc:
                max_test_acc = test_acc
                torch.save(linear_classifier.state_dict(), 'linear/model.pth')
                torch.save(linear_optimizer.state_dict(), 'linear/optimizer.pth')

        np.savez("linear/linear_losses_train_file", np.array(losses_train_linear))
        np.savez("linear/linear_losses_test_file", np.array(losses_test_linear))
        np.savez("linear/linear_acc_train_file", np.array(acc_train_linear))
        np.savez("linear/linear_acc_test_file", np.array(acc_test_linear))

def deprocess_and_show(img_tensor):
    return T.Compose([
            T.Normalize((0, 0, 0), (2, 2, 2)),
            T.Normalize((-0.5, -0.5, -0.5), (1, 1, 1)),
            T.ToPILImage()
          ])(img_tensor)

MIS = True

if MIS:

    counter = 0

    if not os.path.exists('mispredictions'):
        os.makedirs('mispredictions')

    for (_, sample_batched) in enumerate(dataloader_testing_dataset):

        x = sample_batched['image']
        y_actual = sample_batched['label']

        x = x.to(device)
        y_actual  = y_actual.to(device)

        y_intermediate = resnet(x)
        y_predicted = linear_classifier(y_intermediate)


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

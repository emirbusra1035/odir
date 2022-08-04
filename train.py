import torch
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim
from tqdm import tqdm
import torchvision.models as models
torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32

import matplotlib.pyplot as plt
import numpy as np
import csv
torch.manual_seed(7)

torch.cuda.manual_seed(7)
torch.cuda.manual_seed_all(7)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


classes = ('0', '1')
# imshow(torchvision.utils.make_grid(images))
# print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

import torch.nn as nn
import torch.nn.functional as F


def get_inception():
    net = models.inception_v3(pretrained=True)
    net.AuxLogits.fc = nn.Linear(768, 2)
    net.fc = nn.Linear(2048, 2)
    counter = 0
    for child in net.children():
        if counter < 13:
            for param in child.parameters():
                param.requires_grad = False
        else:
            break
        counter += 1
    return net


def get_resnet():
    net = models.resnet50(pretrained=True)
    net.fc = nn.Linear(2048, 2)
    counter = 0
    for child in net.children():
        if counter < 7:
            for param in child.parameters():
                param.requires_grad = False
        else:
            break
        counter += 1
    return net


def get_vgg():
    net = models.vgg16(pretrained=True)
    net.classifier[6] = nn.Linear(4096, 2)
    for i, child in enumerate(net.children()):
        if i == 0:
            for c in child.parameters():
                c.requires_grad = False
    return net


def train(model_name, class_name):
    if model_name == "inception":
        net = get_inception()
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif model_name == "resnet":
        net = get_resnet()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        net = get_vgg()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    net = net.to(device)
    result_list = []
    labels = ["Epoch", "Train Loss","Train Accuracy", "Val Loss", "Val Accuracy", "Test Loss", "Test Accuracy"]
    result_list.append(labels)
    train_set = torchvision.datasets.ImageFolder(root=f'/content/oia-odir-augmented-dataset/{class_name}',
                                                 transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True)

    val_set = torchvision.datasets.ImageFolder(root=f'/content/oia-odir-val-dataset/{class_name}',
                                               transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                             shuffle=True)

    test_set = torchvision.datasets.ImageFolder(root=f'/content/oia-odir-test-dataset/{class_name}',
                                                transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    best_val_loss = None
    best_val_accuracy = None
    counter = 0
    for epoch in range(200):  # loop over the dataset multiple times

        net.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if model_name == "inception":
                outputs, aux = net(inputs)
                loss = 0.0
                loss += criterion(outputs, labels)
                loss += 0.4 * criterion(aux, labels)

            else:
                outputs = net(inputs)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # print statistics
            running_loss += loss.item() * labels.size(0)
        print(f'Train Accuracy : {correct / total}')
        train_loss = running_loss / total
        train_accuracy = correct / total
        net.eval()
        running_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(val_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients

            # forward + backward + optimize
            with torch.no_grad():
                outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # loss = 0.0
            loss = criterion(outputs, labels)
            # loss += criterion(outputs, labels)
            # loss += 0.4*criterion(aux, labels)

            # print statistics
            running_loss += loss.item() * labels.size(0)
            batch_loss = loss.item()
        print(f'Val Accuracy : {correct / total}')
        val_accuracy = correct / total
        val_loss = running_loss / total

        running_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(test_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients

            # forward + backward + optimize
            with torch.no_grad():
                # outputs, aux = net(inputs)
                outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # loss = 0.0
            loss = criterion(outputs, labels)
            # loss += criterion(outputs, labels)
            # loss += 0.4*criterion(aux, labels)

            # print statistics
            running_loss += loss.item() * labels.size(0)
            batch_loss = loss.item()
        print(f'Test Accuracy : {correct / total}')
        test_accuracy = correct / total
        test_loss = running_loss / total
        result = [epoch + 1, train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy]
        result_list.append(result)
        if best_val_accuracy is None:
            best_val_accuracy = val_accuracy
            counter = 0

        elif val_accuracy >= best_val_accuracy:
            counter += 1
            if counter >= 7:
                break
        else:
            best_val_accuracy = val_accuracy
            counter = 0
        torch.save(net, f"/content/drive/MyDrive/oia-odir/oia-odir-best-models/{model_name}_{class_name}_{epoch+1}.pth")

    with open(f'/content/drive/MyDrive/oia-odir/oia-odir-csv-results/{model_name}_{class_name}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(result_list)


# class_names = ["N", "D", "G", "C", "A", "H", "M", "O"]
class_names = ["N", "D", "G", "C", "A", "H", "M", "O"]
# model_names = ["inception", "resnet", "vgg"]
model_names = ["inception"]
for model_name in tqdm(model_names):
    print(model_name)
    for class_name in class_names:
        print(class_name)
        train(model_name=model_name, class_name=class_name)

import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms, datasets, models
from dataloader import Classdataset
from torchvision.models import VGG16_Weights
from model import VGG16
import argparse
import config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='toxic_nontoxic')
    parser.add_argument('--epoch',type = int, default=config.epochs)
    parser.add_argument('--batch_size_train', type=int, default=config.batch_size_train)
    parser.add_argument('--batch_size_test', type=int, default=config.batch_size_test)
    parser.add_argument('--num_workers', type=int, default=config.num_workers)
    parser.add_argument('--classes_num', type=int, default=config.classes_num)
    parser.add_argument('--data_path', type=str, default=config.data_path)
    parser.add_argument('--model_save_path', type=str, default=config.model_save_path)

    args = parser.parse_args()
    random_state = 1
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)

    torch.cuda.manual_seed_all(random_state)
    np.random.seed(random_state)
    epochs = args.epoch
    batch_size_train = args.batch_size_train
    batch_size_test = args.batch_size_test
    num_workers = args.num_workers
    classes_num = args.classes_num
    data_path = args.data_path
    model_save_path = args.model_save_path

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_path = os.path.join(data_path,'train')
    test_path = os.path.join(data_path,'test')
    train_dataset = Classdataset(train_path)
    test_dataset = Classdataset(test_path)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size_train,
                                               shuffle=True,
                                               num_workers=num_workers)
    #
    # test_dataset = datasets.ImageFolder(root='G:/vgg/test', transform=data_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True,
                                              num_workers=num_workers)

    vgg16 = VGG16(classes_num)

    vgg16_pre = models.vgg16(weights=VGG16_Weights.DEFAULT)
    pretrained_dict = vgg16_pre.state_dict()
    model_dict = vgg16.state_dict()
    num_fc = vgg16_pre.classifier[6].in_features
    vgg16_pre.classifier[6] = torch.nn.Linear(num_fc, classes_num)
    vgg16_pre.classifier.add_module("softmax", nn.Softmax())


    def WeightCopy(target, source):
        for i in range(len(target)):
            target[i] = source[i]
    WeightCopy(vgg16.state_dict()['conv1_1.weight'], vgg16_pre.state_dict()['features.0.weight'])
    WeightCopy(vgg16.state_dict()['conv1_1.bias'], vgg16_pre.state_dict()['features.0.bias'])
    WeightCopy(vgg16.state_dict()['conv1_2.weight'], vgg16_pre.state_dict()['features.2.weight'])
    WeightCopy(vgg16.state_dict()['conv1_2.bias'], vgg16_pre.state_dict()['features.2.bias'])
    WeightCopy(vgg16.state_dict()['conv2_1.weight'], vgg16_pre.state_dict()['features.5.weight'])
    WeightCopy(vgg16.state_dict()['conv2_1.bias'], vgg16_pre.state_dict()['features.5.bias'])
    WeightCopy(vgg16.state_dict()['conv2_2.weight'], vgg16_pre.state_dict()['features.7.weight'])
    WeightCopy(vgg16.state_dict()['conv2_2.bias'], vgg16_pre.state_dict()['features.7.bias'])
    WeightCopy(vgg16.state_dict()['conv3_1.weight'], vgg16_pre.state_dict()['features.10.weight'])
    WeightCopy(vgg16.state_dict()['conv3_1.bias'], vgg16_pre.state_dict()['features.10.bias'])
    WeightCopy(vgg16.state_dict()['conv3_2.weight'], vgg16_pre.state_dict()['features.12.weight'])
    WeightCopy(vgg16.state_dict()['conv3_2.bias'], vgg16_pre.state_dict()['features.12.bias'])
    WeightCopy(vgg16.state_dict()['conv3_3.weight'], vgg16_pre.state_dict()['features.14.weight'])
    WeightCopy(vgg16.state_dict()['conv3_3.bias'], vgg16_pre.state_dict()['features.14.bias'])
    WeightCopy(vgg16.state_dict()['conv4_1.weight'], vgg16_pre.state_dict()['features.17.weight'])
    WeightCopy(vgg16.state_dict()['conv4_1.bias'], vgg16_pre.state_dict()['features.17.bias'])
    WeightCopy(vgg16.state_dict()['conv4_2.weight'], vgg16_pre.state_dict()['features.19.weight'])
    WeightCopy(vgg16.state_dict()['conv4_2.bias'], vgg16_pre.state_dict()['features.19.bias'])
    WeightCopy(vgg16.state_dict()['conv4_3.weight'], vgg16_pre.state_dict()['features.21.weight'])
    WeightCopy(vgg16.state_dict()['conv4_3.bias'], vgg16_pre.state_dict()['features.21.bias'])
    WeightCopy(vgg16.state_dict()['conv5_1.weight'], vgg16_pre.state_dict()['features.24.weight'])
    WeightCopy(vgg16.state_dict()['conv5_1.bias'], vgg16_pre.state_dict()['features.24.bias'])
    WeightCopy(vgg16.state_dict()['conv5_2.weight'], vgg16_pre.state_dict()['features.26.weight'])
    WeightCopy(vgg16.state_dict()['conv5_2.bias'], vgg16_pre.state_dict()['features.26.bias'])
    WeightCopy(vgg16.state_dict()['conv5_3.weight'], vgg16_pre.state_dict()['features.28.weight'])
    WeightCopy(vgg16.state_dict()['conv5_3.bias'], vgg16_pre.state_dict()['features.28.bias'])
    WeightCopy(vgg16.state_dict()['fc1.weight'], vgg16_pre.state_dict()['classifier.0.weight'])
    WeightCopy(vgg16.state_dict()['fc1.bias'], vgg16_pre.state_dict()['classifier.0.bias'])
    WeightCopy(vgg16.state_dict()['fc2.weight'], vgg16_pre.state_dict()['classifier.3.weight'])
    WeightCopy(vgg16.state_dict()['fc2.bias'], vgg16_pre.state_dict()['classifier.3.bias'])
    WeightCopy(vgg16.state_dict()['fc3.weight'], vgg16_pre.state_dict()['classifier.6.weight'])
    WeightCopy(vgg16.state_dict()['fc3.bias'], vgg16_pre.state_dict()['classifier.6.bias'])

    optimizer = torch.optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=4,
                                                           verbose=True)

    loss_fn = nn.CrossEntropyLoss()
    vgg16 = vgg16.cuda()

    for epoch in range(epochs):
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        train_turn = 0
        for i, data in enumerate(train_loader, 0):

            inputs, train_labels = data
            if use_gpu:
                inputs, labels = Variable(inputs).cuda(), Variable(train_labels).cuda()
            else:
                inputs, labels = Variable(inputs), Variable(train_labels)
            optimizer.zero_grad()
            outputs = vgg16(inputs)
            _, train_predicted = torch.max(outputs.data, 1)
            train_correct += (train_predicted == labels.data).sum()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_total += train_labels.size(0)
            # print('###### output ######')
        #     print(outputs)
        #     print(train_labels)
        # print('train %d epoch %d loss: %.3f  acc: %.3f ' % (epoch + 1, train_turn, running_loss / train_total, 100 * train_correct / train_total))

        print('train %d epoch loss: %.3f  acc: %.3f ' % (
        epoch + 1, running_loss / train_total, 100 * train_correct / train_total))

        correct = 0
        test_loss = 0.0
        test_total = 0
        best_test = 0
        vgg16.eval()
        for data in test_loader:
            images, labels = data
            if use_gpu:
                images, labels = Variable(images).cuda(), Variable(labels).cuda()
            else:
                images, labels = Variable(images), Variable(labels)
            outputs = vgg16(images).cuda()
            # print(len(outputs))
            final = []
            final.append(outputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
            test_total += labels.size(0)
            correct += (predicted == labels.data).sum()
        acc = 100 * correct / test_total
        if acc > best_test:
            best_test = acc
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            model_path = os.path.join(model_save_path, 'vgg16_10classes.path')
            torch.save(vgg16, model_path)

        print('test  %d epoch loss: %.3f  acc: %.3f ' % (epoch + 1, test_loss / test_total, 100 * correct / test_total))
        # print(final)
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
import numpy as np
class Classdataset(Dataset):

    def __init__(self,file_path):
        if not os.path.isdir(file_path):
            raise ValueError('Please varify file path')
        self.file_path = file_path
        self.labels ={'toxic_Western_Poison_Oak':0,'toxic_Eastern_Poison_Oak':1,'toxic_Eastern_Poison_Ivy':2,\
                      'toxic_Western_Poison_Ivy':3,'toxic_Poison_Sumac':4,'nontoxic_Virginia_creeper':5,\
                      'nontoxic_Boxelder':6,'nontoxic_Jack_in_the_pulpit':7,'nontoxic_Bear_Oak':8,'nontoxic_Fragrant_Sumac':9}
        # self.labels = {"cats":0,"dogs":1}
        # self.labels = {"nontoxic": 0, "toxic": 1}
        self.image_list = self.get_data(self.file_path)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.length = len(self.image_list)
    def __getitem__(self, index):
        img_path, label = self.image_list[index]
        img =Image.open(img_path)
        # img = torch.from_numpy(img)
        img = self.transform(img)
        return img, label
    def __len__(self):
        return self.length


    def get_data(self,path_dir):
        data = list()
        for root, dirs, _ in os.walk(path_dir):
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = self.labels[sub_dir]
                    data.append((path_img, int(label)))
        return data

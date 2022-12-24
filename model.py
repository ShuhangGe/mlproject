
import torch.nn as nn
class VGG16(nn.Module):
    def __init__(self,classes_num):
        super(VGG16, self).__init__()
        # 第一组卷积层 2层 3*3 卷积代替5*5
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=(1, 1))  # 输出尺寸：64 * 224 * 224
        # 生成矩阵大小为[20, 64, 224, 224]，20张图像，都为64*224*224
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))  # 输出尺寸：64 * 224* 224   padding保证卷积结果等同于一层卷积
        self.maxpool = nn.MaxPool2d((2, 2))  # pooling 64 * 112 * 112
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=(1, 1))  # 128 * 112 * 112
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))  # 128 * 112 * 112
        # self.maxpool2 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 128 * 56 * 56
        # 第三组卷积层 3层 3*3 卷积代替7*7
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=(1, 1))  # 256 * 56 * 56
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 56 * 56
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 56 * 56
        # self.maxpool3 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 256 * 28 * 28
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=(1, 1))  # 512 * 28 * 28
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 28 * 28
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 28 * 28
        # self.maxpool4 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 512 * 14 * 14
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 14 * 14
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 14 * 14
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 14 * 14
        # self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 512 * 7 * 7
        # view
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, classes_num)
        # softmax 1 * 1 * 1000
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        out = self.conv1_1(x)  # 222
        out = self.relu(out)
        out = self.conv1_2(out)  # 222

        out = self.relu(out)
        out = self.maxpool(out)  # 112

        out = self.conv2_1(out)  # 110
        out = self.relu(out)
        out = self.conv2_2(out)  # 110
        out = self.relu(out)
        out = self.maxpool(out)  # 56
        out = self.conv3_1(out)  # 54
        out = self.relu(out)
        out = self.conv3_2(out)  # 54
        out = self.relu(out)
        out = self.conv3_3(out)  # 54
        out = self.relu(out)
        out = self.maxpool(out)  # 28

        out = self.conv4_1(out)  # 26
        out = self.relu(out)
        out = self.conv4_2(out)  # 26
        out = self.relu(out)
        out = self.conv4_3(out)  # 26
        out = self.relu(out)
        out = self.maxpool(out)  # 14

        out = self.conv5_1(out)  # 12
        out = self.relu(out)
        out = self.conv5_2(out)  # 12
        out = self.relu(out)
        out = self.conv5_3(out)  # 12
        out = self.relu(out)
        out = self.maxpool(out)  # 7

        # 展平
        out = out.view(-1, 512 * 7 * 7)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.softmax(out)

        return out
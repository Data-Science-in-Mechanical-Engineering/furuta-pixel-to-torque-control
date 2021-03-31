"""Models for vision to state (getting an image of the scene as input and outputting the corresponding state).

@Author: Steffen Bleher
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

DROPOUT_RATE = 0.1
KERNEL_SIZE = (3, 3)
POOLING_KERNEL = (2, 2)
STRIDE = 1

class VisionToStateNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 50, KERNEL_SIZE, stride=STRIDE)
        self.conv1_bn = nn.BatchNorm2d(50)
        self.pool1 = nn.MaxPool2d(POOLING_KERNEL)
        self.conv2 = nn.Conv2d(50, 30, KERNEL_SIZE, stride=STRIDE)
        self.conv2_bn = nn.BatchNorm2d(30)
        self.pool2 = nn.MaxPool2d(POOLING_KERNEL)
        self.conv3 = nn.Conv2d(30, 20, KERNEL_SIZE, stride=STRIDE)
        self.conv3_bn = nn.BatchNorm2d(20)
        self.pool3 = nn.MaxPool2d(POOLING_KERNEL)
        self.conv4 = nn.Conv2d(20, 20, KERNEL_SIZE, stride=STRIDE)
        self.conv4_bn = nn.BatchNorm2d(20)
        self.pool4 = nn.MaxPool2d(POOLING_KERNEL)

        self.fc0 = nn.Linear(2420, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 4)

        self.dropout = nn.Dropout(0.1)

    def encode(self, x):
        x = self.dropout(self.pool1(F.relu(self.conv1_bn(self.conv1(x)))))
        x = self.dropout(self.pool2(F.relu(self.conv2_bn(self.conv2(x)))))
        x = self.dropout(self.pool3(F.relu(self.conv3_bn(self.conv3(x)))))
        x = self.dropout(self.pool4(F.relu(self.conv4_bn(self.conv4(x)))))
        x = x.view(-1, self.num_flat_features(x))
        x = self.dropout(F.relu(self.fc0(x)))
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.dropout(F.relu(self.fc5(x)))
        return x

    def forward(self, x):
        x = torch.tensor(x).cuda()
        # if not torch.is_tensor(x):
        #     x = torch.tensor(x, device=next(self.parameters()).device).cuda()
        x = self.encode(x)
        x = torch.tanh(self.fc6(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# class VisionToStateNet(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 30, KERNEL_SIZE, stride=1)
#         self.conv1_bn = nn.BatchNorm2d(30)
#         self.conv2 = nn.Conv2d(30, 40, KERNEL_SIZE, stride=2)
#         self.conv2_bn = nn.BatchNorm2d(40)
#         self.conv3 = nn.Conv2d(40, 40, KERNEL_SIZE, stride=2)
#         self.conv3_bn = nn.BatchNorm2d(40)
#         self.conv4 = nn.Conv2d(40, 10, KERNEL_SIZE, stride=2)
#         self.conv4_bn = nn.BatchNorm2d(10)
#         self.conv5 = nn.Conv2d(10, 5, KERNEL_SIZE, stride=2)
#         self.conv5_bn = nn.BatchNorm2d(5)
#
#         self.fc1 = nn.Linear(720, 512)
#         self.fc1_bn = nn.BatchNorm1d(512)
#         self.fc2 = nn.Linear(512, 128)
#         self.fc2_bn = nn.BatchNorm1d(128)
#         self.fc3 = nn.Linear(128, 64)
#         self.fc3_bn = nn.BatchNorm1d(64)
#         self.fc4 = nn.Linear(64, 32)
#         self.fc4_bn = nn.BatchNorm1d(32)
#         self.fc5 = nn.Linear(32, 16)
#         self.fc5_bn = nn.BatchNorm1d(16)
#         self.fc6 = nn.Linear(16, 8)
#         self.fc6_bn = nn.BatchNorm1d(8)
#         self.fc7 = nn.Linear(8, 4)
#
#     def encode(self, x):
#         x = F.relu(self.conv1_bn(self.conv1(x)))
#         x = F.relu(self.conv2_bn(self.conv2(x)))
#         x = F.relu(self.conv3_bn(self.conv3(x)))
#         x = F.relu(self.conv4_bn(self.conv4(x)))
#         x = F.relu(self.conv5_bn(self.conv5(x)))
#         x = x.view(-1, self.num_flat_features(x))
#         x = F.relu(self.fc1_bn(self.fc1(x)))
#         x = F.relu(self.fc2_bn(self.fc2(x)))
#         x = F.relu(self.fc3_bn(self.fc3(x)))
#         x = F.relu(self.fc4_bn(self.fc4(x)))
#         x = F.relu(self.fc5_bn(self.fc5(x)))
#         x = F.relu(self.fc6_bn(self.fc6(x)))
#         return x
#
#     def forward(self, x):
#         x = torch.tensor(x).cuda()
#         # if not torch.is_tensor(x):
#         #     x = torch.tensor(x, device=next(self.parameters()).device).cuda()
#         x = self.encode(x)
#         x = torch.tanh(self.fc7(x))
#         return x
#
#     def num_flat_features(self, x):
#         size = x.size()[1:]  # all dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features


if __name__ == '__main__':
    net = VisionToStateNet().to("cuda")
    net.eval()
    input = torch.randn(1, 3, 220, 220)
    t = time.time()
    for i in range(1000):
        out = net(input)
    print((time.time()-t))

    out = net(input)
    print(out.size())

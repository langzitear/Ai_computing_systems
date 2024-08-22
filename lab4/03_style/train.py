import cv2
import numpy
import torch
from zipfile import ZipFile

from torch import nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


#
# from lab4.02.evaluate import ResBlock,TransNet,COCODataSet


class COCODataSet(Dataset):
    def __init__(self):
        super().__init__()
        self.zip_file = ZipFile('./data/train2014_small.zip')
        self.data_set = []
        for file_name in self.zip_file.namelist():
            if file_name.endswith('.jpg'):
                self.data_set.append(file_name)


    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, item):
        file_path = self.data_set[item]
        image = self.zip_file.read(file_path)
        image = numpy.asarray(bytearray(image),dtype='uint8')
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.resize(image,(512,512),interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).float() / 256.0
        image = image.permute(2, 0, 1)
        return image


class ResBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1,bias=False),
            nn.InstanceNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1,bias=False),
            nn.InstanceNorm2d(c)
        )

    def forward(self,x):
        return torch.nn.functional.relu(self.layer(x)+x)


class TransNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            #上采样层
            nn.Conv2d(3, 32, kernel_size=9, padding=4,stride = 1,bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2,bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2,bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            #残差层
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            #下采样层
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            #输出层
            nn.Conv2d(32, 3, kernel_size=9, padding=4, stride=1),
            nn.Sigmoid()

        )


    def forward(self,x):
        # return self.layer(x)
        return torch.nn.functional.relu(self.layer(x))






def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(512,512),interpolation=cv2.INTER_AREA)
    image = torch.from_numpy(image).float()/256.0
    image = image.permute(2,0,1).unsqueeze(0)
    return image




class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        a = models.vgg19(pretrained=True)
        a = a.features
        self.layer1 = nn.Sequential(*a[:3])
        self.layer2 = nn.Sequential(*a[3:6])
        self.layer3 = nn.Sequential(*a[6:11])
        self.layer4 = nn.Sequential(*a[11:16])

    def forward(self,input_):
        out1 = self.layer1(input_)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        return  out1,out2,out3,out4



def get_gram_matrix(f_map):
    """
    """
    n, c, h, w = f_map.shape
    if n == 1:
        f_map = f_map.reshape(c, h * w)
        gram_matrix = torch.mm(f_map, f_map.t())
        return gram_matrix
    else:
        f_map = f_map.reshape(n, c, h * w)
        gram_matrix = torch.matmul(f_map, f_map.transpose(1, 2))
        return gram_matrix

if __name__ == '__main__':
    image_style =load_image('./image/udnie.jpg').cpu()
    net = VGG19().cpu()
    g_net = TransNet().cpu()
    print("g_net build PASS!")
    optimizer = torch.optim.Adam(g_net.parameters(), lr=0.001)
    loss_func = torch.nn.MSELoss()
    print("build loss PASS!\n")
    data_set = COCODataSet()
    print("build data_set PASS!\n")
    batch_size = 1
    data_loader = DataLoader(data_set,batch_size,True,drop_last=True)
    s1,s2,s3,s4 = net(image_style)
    s1 = get_gram_matrix(s1).detach()
    s2 = get_gram_matrix(s2).detach()
    s3 = get_gram_matrix(s3).detach()
    s4 = get_gram_matrix(s4).detach()
    j = 0
    count =0
    epochs = 100
    while j <= epochs:
        for i ,image in enumerate(data_loader):
            image_g = g_net(image)
            out1,out2,out3,out4 = net(image_g)
            #计算风格损失
            loss_s1 = loss_func(get_gram_matrix(out1),s1)
            loss_s2 = loss_func(get_gram_matrix(out2),s2)
            loss_s3 = loss_func(get_gram_matrix(out3),s3)
            loss_s4 = loss_func(get_gram_matrix(out4),s4)

            loss_s = loss_s1 + loss_s2 + loss_s3 + loss_s4

            #计算内容损失
            c1, c2, c3, c4 = net(image)
            loss_c2 = loss_func(out2, c2.detach())
            loss_c = loss_c2

            #计算总loss
            loss = loss_c + 0.000000005 * loss_s
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('j:', j, 'i:', i, 'loss:', loss.item(), 'loss_c:', loss_c.item(), 'loss_s:', loss_s.item())
            count += 1
            if i % 10 ==0:
                torch.save(g_net.state_dict(),'./pth/fst_train.pth')
                save_image(torch.cat((image,image_g),-1),f'./image/train/result_{j}_{i}.jpg')
        j += 1
    print("TRAIN RESULT PASS!\n")










import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image


vgg19 = models.vgg19(pretrained=True)
vgg19.eval()
# 定义图像预处理步骤
preprocess = transforms.Compose([
    transforms.Resize(256),          # 调整图像大小
    transforms.CenterCrop(224),     # 中心裁剪
    transforms.ToTensor(),           # 转换为张量
    transforms.Normalize(             # 归一化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# 加载图像并应用预处理
input_image = Image.open("strawberries.jpg")  # 替换为你的图像路径
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # 增加一个维度，形状为 (1, 3, 224, 224)
with torch.no_grad():  # 不需要计算梯度
    output = vgg19(input_batch)


# 获取预测结果
_, predicted = torch.max(output.data, 1)

# 打印预测的类别索引
print(f'Predicted class index: {predicted.item()}')


# 加载ImageNet类别标签
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
import requests
import json

response = requests.get(LABELS_URL)
labels = json.loads(response.text)

# 打印预测的类别名称
print(f'Predicted class label: {labels[predicted.item()]}')
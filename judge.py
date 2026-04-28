import cv2
import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

# 模型定义
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2), 
            nn.ReLU(), 

            nn.Conv2d(8, 16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2), 
            nn.ReLU(), 

            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2), 
            nn.ReLU(), 
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(288, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
    
# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor()
])

type = random.choice(['cat', 'dog'])
number = random.randint(4001, 5000)
img_path = f'dogs-vs-cats/test_set/{type}.{number}.jpg'
print(f'Processing image: {img_path}')
pil_img = Image.open(img_path).convert('RGB')
input_tensor = transform(pil_img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)
    probs = F.softmax(output, dim=1).cpu().numpy()[0] # 获取预测概率
    pred = np.argmax(probs) # 获取预测类别
    confidence = probs[pred] # 获取预测置信度

# 获取原图
img_cv = cv2.imread(img_path)
if img_cv is None:
    raise FileNotFoundError(f'Image not found: {img_path}')
img_cv = cv2.resize(img_cv, (400, 400))

label_text = f"Prediction: {'Dog' if pred == 1 else 'Cat'} ({confidence*100:.2f}%)"
cv2.putText(img_cv, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# 显示图像
cv2.imshow('Prediction', img_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
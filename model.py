import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from PIL import Image

class HotdogClassifier:
    def __init__(self):
        # 載入預訓練的 ResNet50 模型
        self.model = mobilenet_v2(pretrained=False)
        self.model.load_state_dict(torch.load('model/mobilenet_v2.pth'))
        self.model.eval()
        
        # 圖像預處理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 如果有 GPU 就使用 GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def predict(self, image_path):
        # 載入並預處理圖片
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # 進行預測
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        # 獲取預測結果（假設類別 934 是熱狗）
        hotdog_prob = probabilities[0][934].item()
        is_hotdog = hotdog_prob > 0.5
        
        return is_hotdog, hotdog_prob 
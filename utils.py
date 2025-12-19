import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms as T
from PIL import Image
import requests
from io import BytesIO
import time

DEVICE = "cpu"  # можно заменить на "cuda" если есть GPU

# Список всех 100 классов
CLASSES = [
    'air hockey', 'ampute football', 'archery', 'arm wrestling', 'axe throwing', 
    'balance beam', 'barell racing', 'baseball', 'basketball', 'baton twirling',
    'bike polo', 'billiards', 'bmx', 'bobsled', 'bowling', 'boxing', 'bull riding',
    'bungee jumping', 'canoe slamon', 'cheerleading', 'chuckwagon racing', 'cricket',
    'croquet', 'curling', 'disc golf', 'fencing', 'field hockey', 'figure skating men',
    'figure skating pairs', 'figure skating women', 'fly fishing', 'football',
    'formula 1 racing', 'frisbee', 'gaga', 'giant slalom', 'golf', 'hammer throw',
    'hang gliding', 'harness racing', 'high jump', 'hockey', 'horse jumping',
    'horse racing', 'horseshoe pitching', 'hurdles', 'hydroplane racing', 'ice climbing',
    'ice yachting', 'jai alai', 'javelin', 'jousting', 'judo', 'lacrosse', 'log rolling',
    'luge', 'motorcycle racing', 'mushing', 'nascar racing', 'olympic wrestling',
    'parallel bar', 'pole climbing', 'pole dancing', 'pole vault', 'polo', 'pommel horse',
    'rings', 'rock climbing', 'roller derby', 'rollerblade racing', 'rowing', 'rugby',
    'sailboat racing', 'shot put', 'shuffleboard', 'sidecar racing', 'ski jumping',
    'sky surfing', 'skydiving', 'snow boarding', 'snowmobile racing', 'speed skating',
    'steer wrestling', 'sumo wrestling', 'surfing', 'swimming', 'table tennis', 'tennis',
    'track bicycle', 'trapeze', 'tug of war', 'ultimate', 'uneven bars', 'volleyball',
    'water cycling', 'water polo', 'weightlifting', 'wheelchair basketball',
    'wheelchair racing', 'wingsuit flying'
]

# Трансформации изображений
trnsfrms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

# --- Модель ---
class MyResNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(512, num_classes)
        # заморозка всех слоев кроме последнего
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc.weight.requires_grad = True
        self.model.fc.bias.requires_grad = True

    def forward(self, x):
        return self.model(x)

# --- Функции ---
def load_model(weights_path, device=DEVICE):
    model = MyResNet(num_classes=len(CLASSES))
    model.to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def process_image(image_input):
    """
    image_input: PIL.Image.Image или URL
    Возвращает (PIL.Image, torch.Tensor)
    """
    if isinstance(image_input, str) and image_input.startswith("http"):
        response = requests.get(image_input)
        img = Image.open(BytesIO(response.content)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise ValueError("Неверный формат изображения")
    tensor = trnsfrms(img).unsqueeze(0)  # batch dimension
    return img, tensor

def predict(model, input_tensor, device=DEVICE):
    input_tensor = input_tensor.to(device)
    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
    elapsed = time.time() - start_time
    return CLASSES[int(pred.item())], elapsed
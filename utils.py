import torch
from torchvision import transforms as T
from PIL import Image
import requests
from io import BytesIO
import time

# Preprocessing трансформации
trnsfrms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

# Загрузка модели
def load_model(model_path, weights_path=None, device='cpu', num_classes=10):
    model = torch.load(model_path, map_location=device)
    if weights_path:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model

# Преобразование изображения из файла или ссылки
def process_image(image_input):
    """
    image_input: PIL.Image.Image
    """
    if isinstance(image_input, str) and image_input.startswith('http'):
        response = requests.get(image_input)
        img = Image.open(BytesIO(response.content)).convert('RGB')
    elif isinstance(image_input, Image.Image):
        img = image_input.convert('RGB')
    else:
        raise ValueError("Неверный формат изображения")
    tensor = trnsfrms(img).unsqueeze(0)  # добавляем batch dimension
    return img, tensor

# Предсказание класса
def predict(model, input_tensor, device='cpu'):
    input_tensor = input_tensor.to(device)
    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
    elapsed = time.time() - start_time
    return int(pred.item()), elapsed
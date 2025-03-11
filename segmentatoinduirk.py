import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import imageio
import os

# 1. Cargar el video y extraer fotogramas (reducidos)
video_path = '1.mp4'
reader = imageio.get_reader(video_path)
frames = [cv2.resize(frame, (320, 240)) for frame in reader]

# 2. Detección de contornos con OpenCV y cálculo de gradientes
def detect_contours_with_orientation(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    orientation = np.arctan2(sobely, sobelx)
    return edges, magnitude, orientation

# 3. Preparar datos para el modelo
class ObjectDataset(Dataset):
    def __init__(self, frames):
        self.frames = frames
        self.data = []
        for frame in frames:
            edges, magnitude, orientation = detect_contours_with_orientation(frame)
            y_indices, x_indices = np.where(edges == 255)
            for i in range(len(y_indices)):
                x, y = x_indices[i], y_indices[i]
                # Filtrar por orientación (ejemplo: bordes casi horizontales)
                if abs(np.mean(orientation[max(0, y-5):min(frame.shape[0], y+5), max(0, x-5):min(frame.shape[1], x+5)])) < 0.2:
                    roi = frame[max(0, y-32):min(frame.shape[0], y+32), max(0, x-32):min(frame.shape[1], x+32)]
                    if roi.shape[0] > 10 and roi.shape[1] > 10:
                        roi = np.pad(roi, ((0, max(0, 64-roi.shape[0])), (0, max(0, 64-roi.shape[1])), (0, 0)), mode='constant')
                        roi = roi[:64, :64, :]
                        self.data.append((roi, (x, y)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, _ = self.data[idx]
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        return image, torch.tensor(0)

dataset = ObjectDataset(frames)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 4. Modelo de aprendizaje profundo optimizado
class ObjectDetector(nn.Module):
    def __init__(self):
        super(ObjectDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 16 * 16, 1)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 16 * 16)
        x = torch.sigmoid(self.fc(x))
        return x

# 5. Cargar o entrenar el modelo
model_path = 'model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if os.path.exists(model_path):
    model = ObjectDetector()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
else:
    model = ObjectDetector()
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    torch.save(model.state_dict(), model_path)

# 6. Aplicar el modelo y dibujar contornos con colores aleatorios y orientación
def draw_colored_contours_with_orientation(frame, edges, magnitude, orientation, model):
    model.to(device)
    y_indices, x_indices = np.where(edges == 255)
    for i in range(len(y_indices)):
        x, y = x_indices[i], y_indices[i]
        # Filtrar y dibujar solo si el modelo detecta un objeto
        if abs(np.mean(orientation[max(0, y-5):min(frame.shape[0], y+5), max(0, x-5):min(frame.shape[1], x+5)])) < 0.2:
            roi = frame[max(0, y-32):min(frame.shape[0], y+32), max(0, x-32):min(frame.shape[1], x+32)]
            if roi.shape[0] > 10 and roi.shape[1] > 10:
                roi = np.pad(roi, ((0, max(0, 64-roi.shape[0])), (0, max(0, 64-roi.shape[1])), (0, 0)), mode='constant')
                roi = roi[:64, :64, :]
                roi_tensor = torch.from_numpy(roi).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                output = model(roi_tensor)
                if output > 0.5:
                    color = tuple(np.random.randint(0, 255, 3).tolist())
                    frame[y, x] = color
                    # Dibujar una línea que representa la orientación
                    length = 10
                    angle = np.mean(orientation[max(0, y-5):min(frame.shape[0], y+5), max(0, x-5):min(frame.shape[1], x+5)])
                    x2 = int(x + length * np.cos(angle))
                    y2 = int(y + length * np.sin(angle))
                    cv2.line(frame, (x, y), (x2, y2), color, 2)
    return frame

# 7. Procesar el video y guardar el resultado como frames PNG
output_dir = 'resultados'
os.makedirs(output_dir, exist_ok=True)

for i, frame in enumerate(frames):
    edges, magnitude, orientation = detect_contours_with_orientation(frame)
    frame_with_contours = draw_colored_contours_with_orientation(frame, edges, magnitude, orientation, model)
    imageio.imwrite(os.path.join(output_dir, f'frame_{i:04d}.png'), frame_with_contours)

print(f"Frames guardados en '{output_dir}'")
print(f"Usando device: {device}")
from tadaconv.utils.config import Config
from tadaconv.models.base.builder import build_model
import torch
import cv2
import tadaconv.utils.checkpoint as cu


labels_path = "/Users/konradgoldenbaum/Developement/TAdaConv/data/kinetics710/k710_label_map.txt"  # change to your labels file
with open(labels_path, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(labels)} labels")

cfg = Config(load=True)
model, model_ema = build_model(cfg)

cu.load_test_checkpoint(cfg, model, model_ema, None)

print("Model loaded.")

def crop_center(img, crop_width, crop_height):
    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2
    x1 = max(center_x - crop_width // 2, 0)
    y1 = max(center_y - crop_height // 2, 0)
    x2 = x1 + crop_width
    y2 = y1 + crop_height
    cropped = img[y1:y2, x1:x2]
    return cropped

video_path = "/Volumes/KG1TB/data/valid/action_4/clip_0.mp4"  # change to your file

vid = cv2.VideoCapture(video_path)

count, success = 0, True
num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
indices = torch.linspace(0, num_frames - 1, steps=16).long()
mask = torch.zeros(num_frames, dtype=torch.bool)
mask[indices] = True
frames = []
for i in range(num_frames):
    success, image = vid.read() # Read frame
    if mask[i]:
        frames.append(crop_center(image, 224, 224))

vid.release()
selected = torch.stack([torch.from_numpy(frame) for frame in frames])

# (1, 3, 16, H, W), float32 normalized to [0, 1]
video_tensor = selected.permute(3, 0, 1, 2).unsqueeze(0).float()/ 255.0
print(video_tensor.shape)
model.eval()
with torch.no_grad():
    out = model(video_tensor)



print(torch.argmax(out[0], dim=1))
print("Predicted label:", labels[torch.argmax(out[0], dim=1).item()])


import torch
import torchvision.transforms as T
from PIL import Image
import json
import argparse
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def get_transform():
    return T.Compose([T.ToTensor()])

def get_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def generate_metadata(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0)

    model = get_model()
    with torch.no_grad():
        outputs = model(image_tensor)

    return outputs[0]

def save_metadata(image_path, metadata):
    metadata_path = image_path.replace('.jpg', '.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Image Features")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()

    metadata = generate_metadata(args.image_path)
    save_metadata(args.image_path, metadata)

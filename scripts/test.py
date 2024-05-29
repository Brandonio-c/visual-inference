import os
import json
import torch
import argparse
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from PIL import Image
from model import VisualInferenceModel, generate_context_and_inferences

class TestDataset(Dataset):
    def __init__(self, image_folder):
        self.image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".jpg")]
        self.transform = ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        return image_path, image_tensor

def load_model(model_path):
    model = VisualInferenceModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def test_model(model, dataloader, output_folder):
    for image_path, image_tensor in dataloader:
        image_path = image_path[0]
        image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():
            context_inferences = generate_context_and_inferences(image_tensor)

        output_path = os.path.join(output_folder, os.path.basename(image_path).replace('.jpg', '_inference.json'))
        with open(output_path, 'w') as f:
            json.dump(context_inferences.tolist(), f)
        print(f"Saved context and inferences for {image_path} to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the trained model")
    parser.add_argument("model_path", type=str, help="Path to the trained model")
    parser.add_argument("image_folder", type=str, help="Path to the folder containing test images")
    parser.add_argument("output_folder", type=str, help="Path to the folder to save the output")
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    dataset = TestDataset(args.image_folder)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = load_model(args.model_path)
    test_model(model, dataloader, args.output_folder)

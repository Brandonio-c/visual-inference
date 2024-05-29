import torch
from torch.utils.data import DataLoader, Dataset
from model import VisualInferenceModel

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_model(test_data):
    dataset = CustomDataset(test_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = VisualInferenceModel()
    model.load_state_dict(torch.load('visual_inference_model.pth'))
    model.eval()

    total_loss = 0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for inputs in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            total_loss += loss.item()

    print(f"Total loss: {total_loss}")

if __name__ == "__main__":
    test_data = []  # Load your test data here
    evaluate_model(test_data)

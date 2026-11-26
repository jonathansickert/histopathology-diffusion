from pathlib import Path

import kagglehub
import torch
import torchvision.transforms as T
from PIL import Image

transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])


class HistopathologyDataset(torch.utils.data.Dataset):
    def __init__(self, root=None):
        if root is None:
          root = kagglehub.dataset_download("rushilprajapati/data-final")
          root = Path(root) / "data" / "Histopathology"
        root = Path(root)

        samples = []
        for file in sorted(root.iterdir()):
          if file.suffix.lower() in [".jpeg", ".jpg", ".png"]:
              samples.append(file)

        self.samples = samples
        self.transform = transform
        print(f"Loaded {len(samples)} samples")
      

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image


if __name__ == "__main__":
    data = HistopathologyDataset()

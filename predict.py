import os
import albumentations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pretrainedmodels

from torch.nn import functional as F
from wtfml.data_loaders.image import ClassificationLoader
from wtfml.engine import Engine

class SEResnext50_32x4d(nn.Module):
    def __init__(self, pretrained='imagenet'):
        super(SEResnext50_32x4d, self).__init__()
        
        self.base_model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](pretrained=None)
        if pretrained is not None:
            self.base_model.load_state_dict(
                torch.load(
                    "C:/Users/alica/Desktop/MelanomaModel/se_resnext50_32x4d-a260b3a4.pth"
                )
            )

        self.l0 = nn.Linear(2048, 1)
    
    def forward(self, image, targets):
        batch_size, _, _, _ = image.shape
        
        x = self.base_model.features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        
        out = self.l0(x)
        loss = nn.BCEWithLogitsLoss()(out, targets.view(-1, 1).type_as(x))

        return out, loss

def predict(fold):
    test_data_path = "C:/archive/test"
    df = pd.read_csv("C:/archive/test.csv")
    device = torch.device("cpu")
    model_path=f"model_fold_{fold}.bin"

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    aug = albumentations.Compose(
        [
            albumentations.Resize(224, 224),
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ]
    )

    images = df.image_name.values.tolist()
    images = [os.path.join(test_data_path, i + ".jpg") for i in images]
    targets = np.zeros(len(images))

    test_dataset = ClassificationLoader(
        image_paths=images,
        targets=targets,
        resize=None,
        augmentations=aug,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=4
    )

    model = SEResnext50_32x4d(pretrained=None)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    predictions = Engine.predict(test_loader, model, device=device)
    #predictions = np.vstack((predictions)).ravel()
    predictions = np.concatenate(predictions, axis=0)
    predictions = torch.tensor(predictions)
    predictions = torch.sigmoid(predictions).numpy()
    
    binary_predictions = (predictions >= 0.5).astype(int)
    output_file = "predictions.txt"

    with open(output_file, "w") as f:
        for image_name, pred, bin_pred in zip(images, predictions, binary_predictions):
            f.write(f"{image_name},{pred},{bin_pred}\n")
    
    print(f"Predictions saved to {output_file}")
    
    return predictions

if __name__ == "__main__":
    predict(0)
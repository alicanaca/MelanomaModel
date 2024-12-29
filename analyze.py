import os
import sys
import albumentations
import numpy as np
import torch
import torch.nn as nn
import pretrainedmodels
from torch.nn import functional as F
from wtfml.data_loaders.image import ClassificationLoader
from wtfml.engine import Engine


class SEResnext50_32x4d(nn.Module):
    def __init__(self, pretrained="imagenet"):
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

    def forward(self, image, targets=None):
        batch_size, _, _, _ = image.shape

        x = self.base_model.features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)

        out = self.l0(x)
        loss = None
        if targets is not None:
            loss = nn.BCEWithLogitsLoss()(out, targets.view(-1, 1).type_as(x))

        return out, loss


def analyze_single_image(image_path):
    # Augmentations
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    aug = albumentations.Compose(
        [
            albumentations.Resize(224, 224),
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    # Model paths
    model_paths = [f"C:/Users/alica/Desktop/MelanomaModel/model_fold_{i}.bin" for i in range(5)]
    device = torch.device("cpu")

    # Prepare dataset
    test_dataset = ClassificationLoader(
        image_paths=[image_path],
        targets=[0],  # Dummy target
        resize=None,
        augmentations=aug,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=1
    )

    final_predictions = []

    # Iterate through models
    for model_path in model_paths:
        model = SEResnext50_32x4d(pretrained=None)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        predictions = Engine.predict(test_loader, model, device=device)
        predictions = np.concatenate(predictions, axis=0)
        predictions = torch.tensor(predictions)
        predictions = torch.sigmoid(predictions).numpy()

        final_predictions.append(predictions[0])

    # Average predictions
    averaged_prediction = np.mean(final_predictions)
    binary_prediction = int(averaged_prediction >= 0.45)

    if binary_prediction == 1:
        averaged_prediction = 1 - averaged_prediction

    return binary_prediction, averaged_prediction

if __name__ == "__main__":
    try:
        image_path = sys.argv[1]
        print(f"Analyzing image: {image_path}")  # Adım ekledik
        binary_prediction, confidence = analyze_single_image(image_path)
        print(f"Prediction: {binary_prediction}, Confidence: {confidence}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

def plot_comparison(dataset_logits, val_dataset, num_samples=5):
    preds = (torch.sigmoid(dataset_logits[0][0]) > 0.5).float()
    fig, axes = plt.subplots(5, 3, figsize=(10, 20))
    for i in range(num_samples):
        example = val_dataset.__getitem__(i)
        image_array = example["image"]
        mask_array = example["mask"]
        image = Image.fromarray(image_array.transpose(1,2,0), 'RGB')
        mask = Image.fromarray((np.squeeze(mask_array, axis=0)*255).astype(np.uint8))
        prediction = Image.fromarray((np.squeeze(preds[i].cpu().numpy(), axis=0)*255).astype(np.uint8))
        axes[i][0].imshow(image)
        axes[i][0].set_title("Image")
        axes[i][1].imshow(mask)
        axes[i][1].set_title("Mask")
        axes[i][2].imshow(prediction)
        axes[i][2].set_title("Prediction")

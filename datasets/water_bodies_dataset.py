import os
import torch
import numpy as np

from PIL import Image


class WaterBodiesDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "val", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        if self.mode == "train" or self.mode == "val":
            mode_dir = "trainset"
        else:
            mode_dir = "testset"
        self.images_directory = os.path.join(self.root, f"{mode_dir}/images")
        self.masks_directory = os.path.join(self.root, f"{mode_dir}/masks")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))

        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

        sample = dict(image=image, mask=mask)
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32) / 255.0
        mask = np.where(mask > 0.5, 1.0, 0.0)
        return mask

    def _read_split(self):
        # split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        # split_filepath = os.path.join(self.root, "Annotations", split_filename)
        # with open(split_filepath) as f:
        #     split_data = f.read().strip("\n").split("\n")
        filenames = [image.replace(".jpg", "") for image in os.listdir(self.images_directory)]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "val":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames
    

class SimpleWaterBodiesDataset(WaterBodiesDataset):
    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)

        # resize images
        # image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR))
        ##mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        
        image = np.array(Image.fromarray(sample["image"]))
        mask = np.array(Image.fromarray(sample["mask"]))
        if image.shape != (480, 640, 3):
            image = np.array(Image.fromarray(image).resize((640, 480), Image.BILINEAR))
        if mask.shape != (480, 640):
            mask = np.array(Image.fromarray(mask).resize((640, 480), Image.NEAREST))
        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)

        return sample
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torchvision.transforms as T


class MyDataset(Dataset):
    def __init__(self, root_dir, csv_path, is_train=True, **kwargs):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_path (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.data_root_dir = root_dir
        self.data = pd.read_csv(csv_path).values.tolist()
        # self.data = csv_file.values.tolist()
        self.transform = self.get_transform(is_train, **kwargs)
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_name = os.path.abspath(os.path.join(self.data_root_dir, str(self.data[idx][1]), self.data[idx][0]))
        image = Image.open(img_name)

        label = self.data[idx][1]

        if self.transform:
            image = self.transform(image)

        return image, label

    @staticmethod
    def get_transform(is_train, **tools):
        transform = T.Compose([])
        transform.transforms.append(T.ToTensor())

        if is_train: #For train

            # Hue Saturation
            if tools['color_params']['flag']:
                hue_saturation_params = tools['color_params']
                brightness = hue_saturation_params['brightness']
                contrast = hue_saturation_params['contrast']
                hue = hue_saturation_params['hue']
                saturation = hue_saturation_params['saturation']
                transform.transforms.append(T.transforms.ColorJitter(brightness=brightness, contrast=contrast,
                                                                     saturation=saturation, hue=hue))

            # Resize
            if tools['resize']['flag']:
                resize_params = tools['resize']
                size = resize_params['size']
                transform.transforms.append(T.transforms.Resize(size=size))

            # Random Crop
            if tools['random_crop_params']['flag']:
                random_crop_params = tools['random_crop_params']
                size = random_crop_params['size']
                padding = random_crop_params['padding']
                transform.transforms.append(T.transforms.RandomCrop(size=size, padding=padding))

            # Normalize
            if tools['normalize']['flag']:
                normalize_params = tools['normalize']
                normalize_mean = normalize_params['mean']
                normalize_std = normalize_params['std']
                transform.transforms.append(T.transforms.Normalize(mean=normalize_mean, std=normalize_std))

            if tools['horizontal_flip']['flag']:
                horizontal_flip_params = tools['horizontal_flip']
                p = horizontal_flip_params['p']
                transform.transforms.append(T.transforms.RandomHorizontalFlip(p=p))

        else: # For validation
            if tools['normalize']['flag']:
                normalize_params = tools['normalize']
                normalize_mean = normalize_params['mean']
                normalize_std = normalize_params['std']
                transform.transforms.append(T.transforms.Normalize(mean=normalize_mean, std=normalize_std))

            if tools['resize']['flag']:
                resize_params = tools['random_crop_params']
                size = resize_params['size']
                transform.transforms.append(T.transforms.Resize(size=(size,size)))

        return transform

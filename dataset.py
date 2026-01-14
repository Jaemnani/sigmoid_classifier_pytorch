
import cv2
import torch
import numpy as np
import albumentations as A
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_path, image_paths, input_shape, class_names, aug_brightness=0.0, aug_contrast=0.0, aug_rotate=0, aug_h_flip=False, is_training=True):
        self.root_path = root_path
        self.image_paths = image_paths
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.input_shape = input_shape  # Expected (H, W, C) from arguments
        self.is_training = is_training
        
        aug_methods = []
        if is_training:
            if aug_brightness > 0.0 or aug_contrast > 0.0:
                aug_methods.append(A.RandomBrightnessContrast(p=0.5, brightness_limit=aug_brightness, contrast_limit=aug_contrast))
            if aug_rotate > 0:
                aug_methods.append(A.Rotate(limit=aug_rotate, border_mode=0, value=0))
            if aug_h_flip:
                aug_methods.append(A.HorizontalFlip(p=0.5))
            # GaussianBlur was present in original code
            aug_methods.append(A.GaussianBlur(p=0.5, blur_limit=(7, 7)))
            
        self.transform = A.Compose(aug_methods)
        self.augmentation = len(aug_methods) > 0

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        
        # Determine strict loading mode based on input channels
        # If input_shape[-1] is 1, load grayscale. Else color.
        is_gray = self.input_shape[-1] == 1
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE if is_gray else cv2.IMREAD_COLOR)

        # Preprocess
        # Resize to (W, H) - cv2 uses (width, height)
        img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
        
        if self.augmentation:
            img = self.transform(image=img)['image']
            
        if not is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
            
        # Normalize and reshape
        # Original: (H, W, C) -> (C, H, W) for PyTorch
        if is_gray:
            img = img[..., np.newaxis] # Add channel dimension if it was dropped by cv2 in grayscale
            
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1)) # HWC -> CHW
        
        # Label processing
        dir_name = path.replace(self.root_path, '').split('/')[1]
        label = np.zeros((self.num_classes,), dtype=np.float32)
        if dir_name != 'unknown' and dir_name in self.class_names:
            label[self.class_names.index(dir_name)] = 1.0
            
        return torch.from_numpy(img), torch.from_numpy(label)

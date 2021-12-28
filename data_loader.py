import cv2
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from random import shuffle
from torch.utils.data import Dataset, DataLoader  ##, random_split

class ImageFolder(Dataset):
  def __init__(self, root, c_order, model_type, image_size=512, mode='training', warmup=True, multi_style=False, verbose=True):
    self.c_order = c_order
    self.gt_dir = root.replace('image', 'mask')
    self.img_extension = os.listdir(root)[0].split('.')[1]
    self.image_size = image_size
    self.mode = mode
    self.model_type = model_type
    self.root = root
    self.multi_style = multi_style
    self.verbose = verbose
    self.warmup = warmup
    self.trans, self.trans_space, self.trans_color, self.trans_noise, self.to_tensor = self.transformers()

    if os.path.exists(self.gt_dir):
        self.gt_extension = os.listdir(self.gt_dir)[0].split('.')[1]
        img_paths = [os.path.join(root, x) for x in os.listdir(root) if x.replace(self.img_extension, self.gt_extension) in os.listdir(self.gt_dir)]

        self.reverse = False
        if 'Damage' in self.model_type:
            if 'training' in self.mode:
                self.post_dir = self.root.replace('train', 'fake') 
            else:
                self.post_dir = self.root.replace('pre', 'post')
            self.post_gt_dir = self.post_dir.replace('image', 'mask')
            post_img_paths = [os.path.join(self.post_dir, x.split('/')[-1]) for x in img_paths if x.split('/')[-1].replace(self.img_extension, self.gt_extension) in os.listdir(self.post_gt_dir) and x.split('/')[-1] in os.listdir(self.post_dir)]
            if 'training' in self.mode:
                self.image_paths, self.post_image_paths = [], []
                self.image_paths.extend(img_paths)
                self.image_paths.extend(post_img_paths)
                if self.reverse:
                    self.post_image_paths.extend(post_img_paths)
                    self.post_image_paths.extend(img_paths)
            else:
                self.image_paths = img_paths
                self.post_image_paths = post_img_paths
            print(f'{len(self.post_image_paths)}, {len(self.post_image_paths)} pre and post image_gt pairs loaded')

        else:
            self.image_paths = img_paths
            self.post_image_paths = None
            
        print(f'{len(self.image_paths)} image_gt pairs loaded')
        print(f'{len(self.post_image_paths)} post image_gt pairs loaded')
    else:
        self.image_paths = [os.path.join(root, x) for x in os.listdir(root)]

  def get_random_crop(self, image, gt, crop_height, crop_width):
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height
    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)
    crop = image[y: y + crop_height, x: x + crop_width]
    crop_gt = gt[y: y + crop_height, x: x + crop_width]
    return crop, crop_gt

  def rand_crop_resize(self, img, gt):
    pad_size = 6
    crop_or_resize = random.uniform(0, 1)
    img_size = self.image_size - pad_size * 2  ## use 6 paddings each side
    if crop_or_resize >= 0.5 and self.mode == 'training':
        img, gt = self.get_random_crop(img, gt, img_size, img_size)
    else:
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        if gt is not None:
            gt = cv2.resize(gt, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return img, gt
  
  def transformers(self):
      trans = A.Compose([
          A.HorizontalFlip(p=0.5),
          A.VerticalFlip(p=0.5),
          A.Rotate(limit=30, p=0.5),
          A.Transpose(p=0.5),
          A.RandomGamma(p=0.25),
          A.ShiftScaleRotate(rotate_limit=30)
      ])
      trans_space = A.Compose([
          A.OneOf([
              A.MotionBlur(p=1.0),
              A.MedianBlur(blur_limit=3, p=1.0),
              A.Blur(blur_limit=3, p=1.0),
              A.GlassBlur(p=1.0),
          ], p=1.0),
          A.OneOf([
              A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
              A.GridDistortion(p=0.5),
              A.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)                  
          ], p=1.0),
          ToTensorV2(transpose_mask=True)
      ])
      trans_color = A.Compose([
          A.HueSaturationValue(hue_shift_limit=50, sat_shift_limit=60, val_shift_limit=50, p=1.0),
          A.ColorJitter(),
          A.Equalize(mode='cv'),
          ToTensorV2(transpose_mask=True)
      ])
      trans_noise = A.Compose([
          A.Sharpen(p=1.0),
          A.CLAHE(clip_limit=40),
          A.MultiplicativeNoise(),
          A.Emboss(),          
          A.OneOf([
              A.GaussNoise(),
              A.ISONoise(color_shift=(0.1, 0.5), intensity=(1.0, 1.5)),
          ], p=1.0),
          ToTensorV2(transpose_mask=True)
      ])
      to_tensor = A.Compose([ToTensorV2(transpose_mask=True)])
      return trans, trans_space, trans_color, trans_noise, to_tensor

  def __getitem__(self, index):
    """Reads an image from a file and preprocesses it and returns."""
    image_path = self.image_paths[index]
    filename = image_path.split('/')[-1].split('.')[0]
    GT_path = image_path.replace('image', 'mask').replace(self.img_extension, self.gt_extension)
    if self.post_image_paths is not None:
        post_image_path = self.post_image_paths[index]
        post_GT_path = post_image_path.replace('image', 'mask').replace(self.img_extension, self.gt_extension)

    else:
        post_image_path = image_path
        post_GT_path = GT_path

    img, gt = self.get_img_gt(image_path, GT_path)
    if os.path.exists(post_image_path):
        post_img, post_gt = self.get_img_gt(post_image_path, post_GT_path)
    else:
        post_img = img
        post_gt = gt
    return (img, post_img), (gt, post_gt), filename

  def get_img_gt(self, image_path, GT_path):
    img = cv2.imread(image_path)
    if os.path.exists(GT_path):
        gt = cv2.imread(GT_path)
        if gt.max() > 1:
            gt = gt / gt.max()
    else:
        gt = None

    if img.shape[0] != 500 or img.shape[1] != 500:
        img = cv2.resize(img, (500, 500))
        if gt is not None:
            gt = cv2.resize(gt, (500, 500))

    if 'training' in self.mode:
        if self.multi_style == 4:
            ## data augmentation
            transformed = self.trans(image=img, mask=gt)
            space_transformed = self.trans_space(image=transformed['image'], mask=transformed['mask'])
            color_transformed = self.trans_color(image=transformed['image'], mask=transformed['mask'])
            noise_transformed = self.trans_noise(image=transformed['image'], mask=transformed['mask'])
            tensored = self.to_tensor(image=transformed['image'], mask=transformed['mask'])
            img_list = [tensored['image'], space_transformed['image'], color_transformed['image'], noise_transformed['image']]
            gt_list = [tensored['mask'], space_transformed['mask'], color_transformed['mask'], noise_transformed['mask']]
            gt_list = [(g > 0).float() for g in gt_list]
            if self.verbose:
                print(f'training, 4 augmentation transform')
                self.verbose = False

        elif self.multi_style == 1:
            trans_list = [self.trans, self.trans_space, self.trans_color, self.trans_noise]
            i = np.random.randint(0, len(trans_list))
            tensored = trans_list[i](image=img, mask=gt)
            if i == 0:
                tensored = self.to_tensor(image=tensored['image'], mask=tensored['mask'])
            img_list = [tensored['image']]
            gt_list = [(tensored['mask'] > 0).float()]
            if self.verbose:
                print(f'training, 1 augmentation transform')
                self.verbose = False
        
        elif self.multi_style == 0:
            transformed = self.trans(image=img, mask=gt)
            tensored = self.to_tensor(image=transformed['image'], mask=transformed['mask'])
            img_list = [tensored['image']]
            gt_list = [(tensored['mask'] > 0).float()]
            
            if self.verbose:
                print(f'training, 0 augmentation transform')
                self.verbose = False
    else:
        ## no augmentation in validation or testing
        tensored = self.to_tensor(image=img, mask=gt)
        img_list = [tensored['image']]
        gt_list = [(tensored['mask'] > 0).float()]
        
        if self.verbose:
            print(f'{self.mode}, 0 augmentation transform')
            self.verbose = False
    return torch.stack(img_list, 0), torch.stack(gt_list, 0)

  def __len__(self):
    """Returns the total number of font files."""
    return len(self.image_paths)

def get_loader(image_path, image_size, batch_size, shuffle, mode, model_type, multi_style, warmup, c_order, num_workers=2):
    dataset = ImageFolder(root=image_path, image_size=image_size, model_type=model_type, c_order=c_order, warmup=warmup, mode=mode, multi_style=multi_style, verbose=True)
    data_loader = DataLoader(dataset=dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    pin_memory=True)
    return data_loader

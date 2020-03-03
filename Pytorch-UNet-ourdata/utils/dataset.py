from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    @classmethod
    def mask_preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        print("Size of the pic: ", pil_img.size)
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        # if len(img_nd.shape) == 2: 
        #     # mask target image
        #     img_nd = np.expand_dims(img_nd, axis=2)
        # else:
        #     # grayscale input image
        #     # scale between 0 and 1
        #     img_nd = img_nd / 255
        # # HWC to CHW
        # img_trans = img_nd.transpose((4, 0, 1))
        
        seg_labels = np.zeros((newW, newH))
        unlabeled = [0, 0, 0] # background
        A = [255, 0, 80] # red, inflammatory cell
        B = [128, 255, 192] # light blue, nuclei
        C = [64, 255, 64] # green, cytoplasm

        for w in range(newW):
            for h in range(newH):
                if [img_nd[w,h,0],img_nd[w,h,1],img_nd[w,h,2]] == unlabeled:
                    seg_labels[w,h] = 0
                elif [img_nd[w,h,0],img_nd[w,h,1],img_nd[w,h,2]] == A:
                    seg_labels[w,h] = 1
                elif [img_nd[w,h,0],img_nd[w,h,1],img_nd[w,h,2]] == B:
                    seg_labels[w,h] = 2
                else:
                    seg_labels[w,h] = 3              
        # for c in range(4):
        #     seg_labels[c,:,:] = (pil_img == c)
        print("The label has the shape ", seg_labels.shape)
        # return img_trans.astype(float)
        return seg_labels


        

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        width, height = img.size
        new_width, new_height = mask.size
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2

        img = img.crop((left, top, right, bottom))


        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.mask_preprocess(mask, self.scale)

        # print(mask)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}

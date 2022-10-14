import io
import json
import logging
import os
import torch
import torchvision.transforms as trn
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from utils import ori_classes,simp_classes
from ress import txt_dict
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize,RandomHorizontalFlip,RandomVerticalFlip
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import clip
# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True

# from clip
def _convert_image_to_rgb(image):
    return image.convert("RGB")
def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        RandomHorizontalFlip(0.25),
        RandomVerticalFlip(0.25),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
def _transform_val(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class ClipData(Dataset):
    def __init__(
        self,
        train,
        root='./psgdata/coco/',
        num_classes=56,
    ):
        super(ClipData, self).__init__()
        with open('./psgdata/psg/psg_cls_basic.json') as f:
            dataset = json.load(f)
        self.imglist = [
            d for d in dataset['data']
            if d['image_id'] in dataset[f'{train}_image_ids']
        ]
        self.root = root
        self.transform_image = _transform(224)# clip size
        self.transform_image_val = _transform_val(224)# clip size
        self.num_classes = num_classes
        self.train=train

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        sample = self.imglist[index]
        path = os.path.join("/home/yuanxi/proj/psgdata/coco/", sample['file_name'])
        with open(path, 'rb') as f:
                content = f.read()
                filebytes = content
                buff = io.BytesIO(filebytes)
                image = Image.open(buff).convert('RGB')
                # data= self.transform_image(image) #sample['data'] = self.transform_image(image)
                if self.train=='train':
                    data = self.transform_image(image)  # sample['data'] = self.transform_image(image)
                else:
                    data = self.transform_image_val(image)  # sample['data'] = self.transform_image(image)
                txt=txt_dict[sample['file_name']].split(' ')
                # clip_txt=torch.cat([clip.tokenize(f"{txt} {c} ") for c in simp_classes])
                clip_txt=torch.cat([clip.tokenize(f"{txt[0]} {c} {''.join(txt[1:])}") for c in simp_classes])
        # Generate Soft Label
        soft_label = torch.Tensor(self.num_classes-6)
        soft_label.fill_(0)
        soft_label[[k-6 for k in sample['relations']]] = 1
        # soft_label[sample['relations']] = 1

        #sample['soft_label'] = soft_label
        del sample#['relations']
        return data, soft_label,clip_txt#sample

class ClipData_token(Dataset):
    def __init__(
        self,
        train,
        root='./psgdata/coco/',
        num_classes=56,
    ):
        super(ClipData_token, self).__init__()
        with open('/home/yuanxi/proj/psgdata/psg/psg_cls_basic.json') as f:
            dataset = json.load(f)
        self.imglist = [
            d for d in dataset['data']
            # if d['image_id'] in dataset[f'{train}_image_ids']
            if d['image_id'] in dataset[f'train_image_ids'] or d['image_id'] in dataset[f'val_image_ids']
        ]
        self.root = root
        self.transform_image = _transform(224)# clip size
        self.transform_image_val = _transform_val(224)# clip size
        self.num_classes = num_classes
        self.db=torch.load('token_feature_woclass.pth')
        self.train=train

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        sample = self.imglist[index]
        path = os.path.join(self.root, sample['file_name'])
        with open(path, 'rb') as f:
                content = f.read()
                filebytes = content
                buff = io.BytesIO(filebytes)
                image = Image.open(buff).convert('RGB')
                if self.train=='train':
                    data= self.transform_image(image)
                else:
                    data= self.transform_image_val(image)
                clip_txt=self.db[(path,)]
        # Generate Soft Label
        soft_label = torch.zeros(self.num_classes-6)
        soft_label[[k-6 for k in sample['relations']]] = 1# soft_label[sample['relations']] = 1

        del sample
        return data, soft_label,clip_txt#sample

import torch,json
import easydict
import os
from tensorboardX import SummaryWriter
import logging
import io

import yaml

from model import ClipModel,SimpleClip,ClipModel_gen_token
from utils import backup, set_seed,simp_classes
from torch.utils.data import DataLoader,Dataset
from PIL import Image, ImageFile
#used by clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize,RandomHorizontalFlip,RandomVerticalFlip
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import clip
torch.autograd.set_detect_anomaly(True)
from ress import txt_dict
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
        # print(self.imglist[0])#'image_id': '107905', 'file_name': 'train2017/000000466766.jpg', 'relations': [42, 11, 14, 20, 22]
        self.root = root
        self.train=train


    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        sample = self.imglist[index]
        path = os.path.join(self.root, sample['file_name'])

        txt = txt_dict[sample['file_name']].split(' ')
        # clip_txt = torch.cat([clip.tokenize(f"{txt[0]} {c} {''.join(txt[1:])}") for c in simp_classes])
        clip_txt = clip.tokenize(f"{txt[0]} {''.join(txt[1:])}") 

        return clip_txt,path#, soft_label#,clip_txt#sample

def work(cfg):
    torch.set_printoptions(precision=5, sci_mode=False)
    set_seed(cfg.seed)
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.device=device

    mapp={}

    bz=1
    train_dataset = ClipData(train='train')
    train_dataloader = DataLoader(train_dataset,batch_size=bz,shuffle=False,num_workers=8)
    val_dataset = ClipData(train='val')
    val_dataloader = DataLoader(val_dataset,batch_size=bz,shuffle=False,num_workers=8)
    test_dataset = ClipData(train='test')
    test_dataloader = DataLoader(test_dataset, batch_size=bz,shuffle=False, num_workers=8)

    model = ClipModel_gen_token(cfg)  # to(device)
    model = model.to(device)
    model.eval()

    def work_data(dataloader):
        for sentence, name in dataloader:
            with torch.no_grad():
                sentence = sentence.cuda()
                emb = model.model.encode_text(sentence.squeeze(0))
                emb = emb / emb.norm(dim=-1, keepdim=True)
                emb = emb.detach().cpu()
                mapp[name] = emb
    work_data(test_dataloader)
    work_data(train_dataloader)
    work_data(val_dataloader)

    torch.save(mapp,'token_feature.pth')

if __name__=="__main__":
    cfg=easydict.EasyDict(yaml.load(open('cfg.yaml'),yaml.FullLoader))
    work(cfg)

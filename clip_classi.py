import torch,json
import easydict
import os
from tensorboardX import SummaryWriter
import logging
import io

import yaml

from model import ClipModel,SimpleClip
from utils import backup, set_seed
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
def _convert_image_to_rgb(image):
    return image.convert("RGB")
def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        RandomHorizontalFlip(p=0.25),
        RandomVerticalFlip(p=0.25),
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
        path = os.path.join(self.root, sample['file_name'])
        try:
            with open(path, 'rb') as f:
                content = f.read()
                filebytes = content
                buff = io.BytesIO(filebytes)
                image = Image.open(buff).convert('RGB')
                if self.train=='train':
                    data= self.transform_image(image) #sample['data'] = self.transform_image(image)
                else:
                    data= self.transform_image_val(image) #sample['data'] = self.transform_image(image)

                # txt=txt_dict[sample['file_name']].split(' ')
                # clip_txt=torch.cat([clip.tokenize(f"{txt} {c} ") for c in simp_classes])
                # clip_txt=torch.cat([clip.tokenize(f"{txt[0]} {c} {''.join(txt[1:])}") for c in simp_classes])
        except Exception as e:
            logging.error('Error, cannot read [{}]'.format(path))
            raise e
        # Generate Soft Label
        soft_label = torch.Tensor(self.num_classes-6)
        soft_label.fill_(0)
        soft_label[[k-6 for k in sample['relations']]] = 1
        # soft_label[sample['relations']] = 1

        #sample['soft_label'] = soft_label
        del sample#['relations']
        return data, soft_label#,clip_txt#sample
class SimpleData(Dataset):
    def __init__(
            self,
            train,
            root='./psgdata/coco/',
            num_classes=56,
    ):
        super(SimpleData, self).__init__()
        with open('./psgdata/psg/psg_cls_basic.json') as f:
            dataset = json.load(f)
        self.imglist = [
            d for d in dataset['data']
            if d['image_id'] in dataset[f'train_image_ids'] or  d['image_id'] in dataset[f'val_image_ids']
        ]
        self.root = root
        self.transform_image = _transform(224)  # clip size
        self.transform_image_val = _transform_val(224)  # clip size
        self.num_classes = num_classes
        self.train = train
        self.db=torch.load('visual_feature.pth')

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        sample = self.imglist[index]
        path = os.path.join(self.root, sample['file_name'])
        data=self.db[path]
        '''
        try:
            with open(path, 'rb') as f:
                content = f.read()
                filebytes = content
                buff = io.BytesIO(filebytes)
                image = Image.open(buff).convert('RGB')
                if self.train == 'train':
                    data = self.transform_image(image)  # sample['data'] = self.transform_image(image)
                else:
                    data = self.transform_image_val(image)  # sample['data'] = self.transform_image(image)

                # txt=txt_dict[sample['file_name']].split(' ')
                # clip_txt=torch.cat([clip.tokenize(f"{txt} {c} ") for c in simp_classes])
                # clip_txt=torch.cat([clip.tokenize(f"{txt[0]} {c} {''.join(txt[1:])}") for c in simp_classes])
        except Exception as e:
            logging.error('Error, cannot read [{}]'.format(path))
            raise e
        '''
        # Generate Soft Label
        soft_label = torch.Tensor(self.num_classes - 6)
        soft_label.fill_(0)
        soft_label[[k - 6 for k in sample['relations']]] = 1
        # soft_label[sample['relations']] = 1
        # sample['soft_label'] = soft_label
        del sample  # ['relations']
        return data, soft_label  # ,clip_txt#sample
def train(cfg):

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.set_printoptions(precision=5, sci_mode=False)

    save_path=backup(cfg.log_path)
    logf=open(os.path.join(save_path,'log.txt'),'w')
    set_seed(cfg.seed)
    writer = SummaryWriter(cfg.log_path)
    device= 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg.device=device

    train_dataset = SimpleData(train='train')
    # train_dataset = ClipData(train='train')
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,num_workers=8)
    val_dataset = SimpleData(train='val')
    # val_dataset = ClipData(train='val')
    val_dataloader = DataLoader(val_dataset,batch_size=4,
                                shuffle=False,num_workers=8)

    # test_dataset = PsgData(train='test')
    # test_dataloader = DataLoader(test_dataset,batch_size=32,
    #                              shuffle=False,num_workers=8)



    model=SimpleClip(cfg)#to(device)
    if cfg.load_path!='':
        model.load_state_dict(torch.load(cfg.load_path).state_dict())
    model=model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler= torch.optim.lr_scheduler.StepLR(optimizer,gamma=cfg.gamma,step_size=cfg.step_size)
    iters=0
    best_acc = -1
    for e in range(1,cfg.epoch+1):
        model.train()
        train_avg_loss = 0.0
        train_acc = 0.0
        train_num = 0
        torch.cuda.empty_cache()
        for batch_idx, data in enumerate(train_dataloader):

            img,label=data[0].cuda(),data[1].cuda()#,data[2].cuda()
            optimizer.zero_grad()

            pred=model(img)
            loss=model.loss(pred,label)

            pred_label=pred.topk(3)
            gt=label.topk(3)
            acc=torch.sum(pred_label.indices==gt.indices)/pred_label.indices.numel()

            loss.backward()
            optimizer.step()
            iters+=1
            writer.add_scalar('train/loss', loss.item(), iters)
            writer.add_scalar('train/acc', acc.item(), iters)

            train_acc += acc.item()*label.shape[0]
            train_avg_loss+= loss.item()*label.shape[0]
            train_num += label.numel() // 3

            # print(img.shape,end=' ')
            # if batch_idx % 40 ==0:
            #     logline='Train Epoch: {} Batch: {}\t Loss: {:.6f}\t Acc:{:.3f}'.format(e,batch_idx,
            #             loss.item(),acc.item())
            #     print(logline)
            #     logf.write(logline+'\n')
        print('start validate')
        print('Train Epoch: {} \t Total Loss: {:.6f}\t Total Acc:{:.3f}\t lr {:.5f}'.format(e,
                        train_avg_loss/5000,train_acc/5000,optimizer.param_groups[0]['lr']))
        val_avg_loss=0.0
        val_acc=0.0
        val_num=0
        model.eval()
        torch.cuda.empty_cache()
        for batch_idx, data in enumerate(val_dataloader):
            with torch.no_grad():
                img, label = data[0].cuda(), data[1].cuda()#, data[2].cuda()

                pred = model(img)
                loss = model.loss(pred, label)

            pred_label = pred.topk(3)
            gt = label.topk(3)
            acc = torch.sum(pred_label.indices == gt.indices) / pred_label.indices.numel()

            val_acc+=acc.item()*label.shape[0]
            val_avg_loss+=loss.item()*label.shape[0]
            val_num+=label.numel()//3

        val_acc/=500
        val_avg_loss/=500
        writer.add_scalar('val/loss', val_acc, iters)
        writer.add_scalar('val/acc', val_avg_loss, iters)

        # is_best=(val_acc*0.1+train_acc/4500*0.9)>best_acc
        is_best=train_acc/5000  >best_acc
        if is_best:
            torch.save(model,os.path.join(save_path,'model.pth'))
            best_acc=max(train_acc/5000,best_acc)
            # best_acc=max(val_acc*0.1+train_acc/4500*0.9,best_acc)
            print('Cur best: ',best_acc,' model saved')


        logline='Val Epoch: {}, Avg Loss {:.6f}, Acc {:.3f} '.format(e,val_avg_loss,val_acc)
        print(logline)
        logf.write(logline+'\n')
        logline="---->This epoch: {:.6f}".format(train_acc/5000)+", Cur Best: {:.6f}".format(best_acc)
        # logline="---->This epoch: {:.6f}".format(val_acc*.1+train_acc/4500*.9)+", Cur Best: {:.6f}".format(best_acc)
        print(logline)
        logf.write(logline+'\n')
        scheduler.step()
        torch.cuda.empty_cache()

    logf.close()




if __name__=="__main__":
    cfg=easydict.EasyDict(yaml.load(open('cfg.yaml'),yaml.FullLoader))
    train(cfg)

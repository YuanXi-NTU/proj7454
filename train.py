import torch
import easydict
import os
from tensorboardX import SummaryWriter

import yaml

from dataset import PsgData
from model import Tmodel,ClipModel
from utils import backup, set_seed
from torch.utils.data import DataLoader



def train(cfg):

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.set_printoptions(precision=5, sci_mode=False)
    save_path=backup(cfg.log_path)
    logf=open(os.path.join(save_path,'log.txt'),'w')
    set_seed(cfg.seed)
    writer = SummaryWriter(cfg.log_path)
    device= 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg.device=device

    train_dataset = PsgData(train='train')
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,num_workers=8)
    val_dataset = PsgData(train='val')
    val_dataloader = DataLoader(val_dataset,batch_size=32,
                                shuffle=False,num_workers=8)

    # test_dataset = PsgData(train='test')
    # test_dataloader = DataLoader(test_dataset,batch_size=32,
    #                              shuffle=False,num_workers=8)



    model=Tmodel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    iters=0
    best_acc = -1

    for e in range(1,cfg.epoch+1):
        for batch_idx, data in enumerate(train_dataloader):
            img,label=data[0].to(device),data[1].to(device)
            optimizer.zero_grad()

            pred=model(img)
            loss=model.loss(pred,label)

            loss.backward()
            optimizer.step()
            iters+=1
            writer.add_scalar('losses/train_loss', loss.item(), iters)
            if batch_idx % 20 ==0:
                logline='Train Epoch: {} [{:.0f}%]\tLoss: {:.6f}'.format(e,batch_idx,
                        loss.item())
                print(logline)
                logf.write(logline+'\n')

        val_avg_loss=0.0
        val_acc=0.0
        val_num=0

        for img,label in val_dataloader:
            img,label=img.to(device),label.to(device)
            pred = model(img)
            loss = model.loss(pred, label)
            acc = top3(pred)==label


            val_acc+=acc
            val_avg_loss+=loss
            val_num+=label.numel()//3

        val_acc/=val_num
        val_avg_loss/=val_num

        is_best=val_acc>best_acc
        if is_best:
            torch.save(model,os.path.join(save_path,'model.pth'))
            best_acc=val_acc
        logline='Val Epoch: {}, Avg Loss {}, Acc {} '.format(e,val_avg_loss,val_acc)
        print(logline)
        logf.write(logline+'\n')
    logf.close()




if __name__=="__main__":
    train(easydict.EasyDict(yaml.load(open('cfg.yaml'),yaml.FullLoader)))

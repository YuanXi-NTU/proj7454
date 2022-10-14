import torch
import easydict
import os
from tensorboardX import SummaryWriter

import yaml

from dataset import ClipData,ClipData_token
from model import ClipModel,SimpleClip,ClipToken
from utils import backup, set_seed
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(True)

def train(cfg):
    torch.set_printoptions(precision=5, sci_mode=False)

    if not cfg.dbg:
        save_path=backup(cfg.log_path)
        logf=open(os.path.join(save_path,'log.txt'),'w')
        writer = SummaryWriter(cfg.log_path)
    set_seed(cfg.seed)
    device= 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg.device=device

    train_dataset = ClipData_token(train='train')
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,num_workers=8)
    val_dataset = ClipData_token(train='val')
    val_dataloader = DataLoader(val_dataset,batch_size=4,
                                shuffle=False,num_workers=8)

    model=ClipToken(cfg)
    if cfg.load_path!='':
        st=torch.load(cfg.load_path).state_dict()
        model.load_state_dict(st)
    model=model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=5,min_lr=5e-6,)
    iters=0
    best_acc = -1
    best_val = -1
    for e in range(1,cfg.epoch+1):
        model.train()
        train_loss = 0
        correct_train=0
        train_sample=0

        torch.cuda.empty_cache()
        for batch_idx, data in enumerate(train_dataloader):
            img,label,txt=data[0].cuda(),data[1].cuda(),data[2].cuda()
            optimizer.zero_grad()

            pred=model(img,txt)
            loss=model.loss(pred,label)
            # loss=model.loss(pred.squeeze(1),label)

            pred_label=pred.topk(3).indices
            # pred_label=pred.squeeze(1).topk(3).indices
            gt=label.topk(3).indices

            # batch_acc=torch.sum(pred_label==gt).item()
            batch_acc=0
            for i in range(pred_label.shape[0]):
                tmp=set(pred_label[i].tolist())
                for j in gt[i]:
                    if j.item() in tmp:
                        batch_acc+=1

            train_sample+=gt.numel()#label.shape[0]
            correct_train+=batch_acc#*label.shape[0]
            train_loss+=loss*label.shape[0]

            loss.backward()
            optimizer.step()
            iters+=1
            if not cfg.dbg:
                writer.add_scalar('train/loss', loss.item(), iters)
            # writer.add_scalar('train/acc', batch_acc, iters)

        logline='Train Epoch: {} \t Total Loss: {:.6f}\t Total Acc:{:.3f}\t'.format(e,
                        train_loss/train_sample,correct_train/train_sample,optimizer.param_groups[0]['lr'])+str(correct_train)+'/'+str(train_sample)
        if not cfg.dbg:
            logf.write(logline)
        print('start validate')

        model.eval()
        torch.cuda.empty_cache()
        val_loss=0
        val_sample=0
        val_acc=0
        correct_val=0
        for batch_idx, data in enumerate(val_dataloader):
            with torch.no_grad():
                img, label,txt = data[0].cuda(), data[1].cuda(), data[2].cuda()
                pred = model(img,txt)
                loss=model.loss(pred,label)
                # loss=model.loss(pred.squeeze(1),label)

            pred_label = pred.topk(3).indices
            # pred_label = pred.squeeze(1).topk(3).indices
            gt = label.topk(3).indices
            # batch_acc = torch.sum(pred_label == gt).item()
            batch_acc = 0
            for i in range(pred_label.shape[0]):
                tmp = set(pred_label[i].tolist())
                for j in gt[i]:
                    if j.item() in tmp:
                        batch_acc += 1
            val_sample += gt.numel()#[0]
            correct_val += batch_acc
            val_loss += loss #* label.shape[0]

        if not cfg.dbg:
            writer.add_scalar('val/loss', val_loss.item()/val_sample, iters)
        # writer.add_scalar('val/acc',val_acc/val_sample, iters)

        # ep_acc=(val_acc)/(train_sample+val_sample)+train_acc/(train_sample+val_sample)
        ep_acc=(correct_train+correct_val)/(train_sample+val_sample)
        is_best=ep_acc>best_acc
        is_best_val=(correct_val/val_sample)>best_val

        logline='Val Epoch: {}, Avg Loss {:.6f}, Acc {:.3f} \t'.format(e,val_loss/val_sample,correct_val/val_sample)
        logline=logline+f'{correct_val} / {val_sample} \n ----epoch end-----'

        if is_best:
            if not cfg.dbg:
                torch.save(model,os.path.join(save_path,'model_tot.pth'))

            best_acc=max(ep_acc,best_acc)
            logline=logline+f'\nThis Epoch: {ep_acc} Cur best: {best_acc} model saved'
        if not cfg.dbg and is_best_val:
            torch.save(model, os.path.join(save_path, 'model_val.pth'))
            logline=logline+f'\nsaved best val model'
            best_val=max(best_val,correct_val/val_sample)
        if not cfg.dbg:
            logf.write(logline+'\n')
        print(logline)

        scheduler.step(best_acc)
        torch.cuda.empty_cache()
    if not cfg.dbg:
        logf.close()




if __name__=="__main__":
    cfg=easydict.EasyDict(yaml.load(open('cfg.yaml'),yaml.FullLoader))
    train(cfg)

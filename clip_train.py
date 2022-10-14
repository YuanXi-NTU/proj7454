import torch
import easydict
import os
from tensorboardX import SummaryWriter

import yaml

from dataset import PsgData,ClipData
from model import ClipModel,SimpleClip
from utils import backup, set_seed
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(True)

def train(cfg):

    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    torch.set_printoptions(precision=5, sci_mode=False)

    save_path=backup(cfg.log_path)
    logf=open(os.path.join(save_path,'log.txt'),'w')
    set_seed(cfg.seed)
    writer = SummaryWriter(cfg.log_path)
    device= 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg.device=device

    train_dataset = ClipData(train='train')
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,num_workers=8)
    val_dataset = ClipData(train='val')
    val_dataloader = DataLoader(val_dataset,batch_size=4,
                                shuffle=False,num_workers=8)

    # test_dataset = PsgData(train='test')
    # test_dataloader = DataLoader(test_dataset,batch_size=32,
    #                              shuffle=False,num_workers=8)



    model=ClipModel(cfg)#to(device)
    if cfg.load_path!='':
        import collections
        st=torch.load(cfg.load_path).state_dict()
        # st=collections.OrderedDict({k[6:]:v for k,v in st.items()})
        model.load_state_dict(st)
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
            img,label,txt=data[0].cuda(),data[1].cuda(),data[2].cuda()
            optimizer.zero_grad()

            pred=model(img,txt)
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
            if batch_idx % 60 ==0:
                logline='Train Epoch: {} Batch: {}\t Loss: {:.6f}\t Acc:{:.3f}'.format(e,batch_idx,
                        loss.item(),acc.item())
                print(logline)
                logf.write(logline+'\n')
        print('start validate')
        print('Train Epoch: {} \t Total Loss: {:.6f}\t Total Acc:{:.3f}\t lr {:.5f}'.format(e,
                        train_avg_loss/4500,train_acc/4500,optimizer.param_groups[0]['lr']))
        val_avg_loss=0.0
        val_acc=0.0
        val_num=0
        model.eval()
        torch.cuda.empty_cache()
        for batch_idx, data in enumerate(val_dataloader):
            img, label,txt = data[0].cuda(), data[1].cuda(), data[2].cuda()
            pred = model(img,txt)
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

        is_best=(val_acc*0.1+train_acc/4500*0.9)>best_acc
        if is_best:
            # torch.save(model,os.path.join(save_path,'model{:.3f}.pth'.format(best_acc)))
            best_acc=max(val_acc*0.1+train_acc/4500*0.9,best_acc)
            print('Cur best: ',best_acc,' model saved')

        logline='Val Epoch: {}, Avg Loss {:.6f}, Acc {:.3f} '.format(e,val_avg_loss,val_acc)
        print(logline)
        logf.write(logline+'\n')
        scheduler.step()
        torch.cuda.empty_cache()

    logf.close()




if __name__=="__main__":
    cfg=easydict.EasyDict(yaml.load(open('cfg.yaml'),yaml.FullLoader))
    train(cfg)

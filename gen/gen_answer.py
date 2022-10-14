import torch
import easydict
import os
from tensorboardX import SummaryWriter

import yaml

from dataset import PsgData,ClipData
from model import Tmodel,ClipModel,SimpleClip
from utils import backup, set_seed
from torch.utils.data import DataLoader
from evaluator import Evaluator
torch.autograd.set_detect_anomaly(True)

def train(cfg):

    torch.set_printoptions(precision=5, sci_mode=False)

    set_seed(cfg.seed)
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.device=device


    test_dataset = ClipData(train='test')
    test_dataloader = DataLoader(test_dataset,batch_size=8,
                                 shuffle=False,num_workers=8)



    model=ClipModel(cfg)#to(device)
    evaluator=Evaluator(model,3)
    if cfg.load_path!='':
        import collections
        st=torch.load(cfg.load_path).state_dict()
        # st=collections.OrderedDict({k[6:]:v for k,v in st.items()})
        model.load_state_dict(st)
    model=model.to(device)
    res=evaluator.submit(test_dataloader)
    f=open('result.txt','w')
    for i in res:
        for j in range(len(i)):
            if j!=2:
                f.write(str(i[j]+6)+' ')
            else:
                f.write(str(i[j]+6))
        f.write('\n')
    f.close()

    # import ipdb
    # ipdb.set_trace()

if __name__=="__main__":
    cfg=easydict.EasyDict(yaml.load(open('cfg.yaml'),yaml.FullLoader))
    train(cfg)

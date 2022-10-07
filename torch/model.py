import clip
import torch
import torchvision

import torch.nn as nn
from .utils import ori_classes, classes, simp_classes

class ClipModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model,_=clip.load('ViT-B/32', args.device)
    def forward(self,img,txt):

        with torch.no_grad():
            image_features = self.model.encode_image(img)
            text_features = self.model.encode_text(txt)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        print(similarity.shape)
        exit(0)
        values, indices = similarity[0].topk(1)
        return similarity
    def loss(self,pred,label):
        return torch.sum(pred-label)

class SegModel(nn.Module):
    pass

class Tmodel(nn.Module):
    def __init__(self,args):
        super().__init__()
        # self.seg=SegModel()
        self.seg= torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.seg.eval()
        self.clip,self.clip_pre=clip.load('ViT-B/32',args.device)
        self.T2PIL=torchvision.transforms.Compose(
            [torchvision.transforms.ToPILImage()]
        )

    def forward(self,x):
        # seg phase
        with torch.no_grad():
            res=self.seg(x)
            boxes=torch.stack([r['boxes'] for r in res])
            labeles=torch.stack([r['labels'] for r in res])
            scores=torch.stack([r['scores'] for r in res])
            masks=torch.stack([r['masks'] for r in res])


            mask=self.seg(x)
            x=x*mask
        x=[self.T2PIL(x[xx]) for xx in range(len(x.size()[0]))]

        image_input=[self.clip_pre(xx).unsqueeze(0) for xx in x] # turn to clip, batched
        # clip phase
        image_input = self.clip_pre(x).unsqueeze(0)
        txt_input = torch.cat([self.clip.tokenize(f"a photo of a {c}" for c in classes)]).unsqueeze(0).repeat(len(image_input),1,1)

        with torch.no_grad():
            img_feat = self.clip.encode_image(image_input)
        txt_feat = self.model.encode_text(txt_input)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

        sim = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)
        val, idx = sim[0].topk(3)

        return val

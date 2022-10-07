import clip
import torch
import torchvision

import torch.nn as nn
from utils import ori_classes, classes, simp_classes
#from clip, modified to fp32
def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.float()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.float()

    model.apply(_convert_weights_to_fp16)


class ClipModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model,_=clip.load('ViT-B/32', args.device,jit=False)
        self.args=args
        convert_weights(self.model)
        # self.clip_txt=torch.cat([clip.tokenize(f"an action of {c}") for c in ori_classes]).to(args.device)
        self.clip_txt=torch.cat([clip.tokenize(f"a person {c} something") for c in simp_classes]).to(args.device)
        # self.clip_txt=torch.cat([clip.tokenize(f"{c} something") for c in simp_classes]).to(args.device)
        self.loss_func=torch.nn.MSELoss()
    def forward(self,img,txt):
        similarity=torch.zeros(txt.shape[0],txt.shape[1]).to(self.args.device)
        for i in range(txt.shape[0]):
            with torch.no_grad():
                image_features = self.model.encode_image(img[i].unsqueeze(0))
                image_features2 = image_features / image_features.norm(dim=-1, keepdim=True)
            # text_features = self.model.encode_text(self.clip_txt)
            text_features = self.model.encode_text(txt[i])
            text_features2 = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity[i] += (100.0 * image_features2 @ text_features2.T).softmax(dim=-1).squeeze(0)
        return similarity
        # with torch.no_grad():
        #     image_features = self.model.encode_image(img)
        #     image_features2 = image_features / image_features.norm(dim=-1, keepdim=True)
        # # text_features = self.model.encode_text(self.clip_txt)
        # text_features = self.model.encode_text(txt)
        # text_features2 = text_features/text_features.norm(dim=-1, keepdim=True)
        # similarity = (100.0 * image_features2 @ text_features2.T).softmax(dim=-1)
        # return similarity
    def loss(self,pred,label):
        return self.loss_func(pred,label)
        # return torch.sum((pred-label)**2)

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

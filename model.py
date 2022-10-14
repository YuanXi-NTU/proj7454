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

class ClipToken(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model,_=clip.load('ViT-B/32', args.device,jit=False)
        # self.model,_=clip.load('RN50x64', args.device,jit=False)
        self.args=args
        convert_weights(self.model)
        self.loss_func=torch.nn.BCELoss()
        self.tt=nn.Sequential(
            nn.Linear(1024,2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048,2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048,50),
            nn.Softmax()
        )
    def forward(self,img,txt):
        image_features = self.model.encode_image(img)
        image_features2 = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity=self.tt(image_features2 )
        # similarity=torch.bmm(image_features2.unsqueeze(1),txt.transpose(1,2)).softmax(dim=-1).squeeze(0)
        return similarity
    def loss(self,pred,label):
        return self.loss_func(pred,label)


class SimpleClip(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.model, _ = clip.load('ViT-B/32', args.device, jit=False)
        # self.model,_=clip.load('RN50x64', args.device,jit=False)
        # self.args = args
        # convert_weights(self.model)
        self.loss_func = torch.nn.BCELoss()
        self.mlp=nn.Sequential(
            nn.Linear(1024,2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            # nn.Linear(2048, 256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            nn.Linear(2048, 50),
            nn.Softmax()
        )
        self.func=torch.nn.Softmax()
        self.func2=torch.nn.ReLU()
    def forward(self,img):
        # with torch.no_grad():
        #     image_features = self.model.encode_image(img)
        #     image_features2 = image_features / image_features.norm(dim=-1, keepdim=True)
        # logit=self.mlp(image_features2)
        logit=self.mlp(img)
        return logit
    def loss(self,pred,label):
        return self.loss_func(pred,label)
class ClipModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model,_=clip.load('ViT-B/32', args.device,jit=False)
        # self.model,_=clip.load('RN50x64', args.device,jit=False)
        self.args=args
        convert_weights(self.model)
        # self.clip_txt=torch.cat([clip.tokenize(f"an action of {c}") for c in ori_classes]).to(args.device)
        # self.clip_txt=torch.cat([clip.tokenize(f"a person {c} something") for c in simp_classes]).cuda()#to(args.device)
        # self.clip_txt=torch.cat([clip.tokenize(f"{c} something") for c in simp_classes]).to(args.device)
        self.loss_func=torch.nn.BCELoss()
    def forward(self,img,txt):
        image_features = self.model.encode_image(img)
        image_features2 = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity=torch.bmm(image_features2.unsqueeze(1),txt.transpose(1,2)).softmax(dim=-1).squeeze(0)
        '''
        similarity=torch.zeros(txt.shape[0],txt.shape[1]).cuda()#to(self.args.device)
        for i in range(txt.shape[0]):
            # with torch.no_grad():
            image_features = self.model.encode_image(img[i].unsqueeze(0))
            image_features2 = image_features / image_features.norm(dim=-1, keepdim=True)

            # text_features = self.model.encode_text(self.clip_txt)
            # text_features = self.model.encode_text(txt[i])
            # text_features2 = text_features / text_features.norm(dim=-1, keepdim=True)

            # similarity[i] += (100.0 * image_features2 @ txt.T).softmax(dim=-1).squeeze(0)
            similarity[i] += (100.0 * image_features2 @ txt.T).softmax(dim=-1).squeeze(0)

            # similarity[i] += (100.0 * image_features2 @ text_features2.T).softmax(dim=-1).squeeze(0)
        '''
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

class ClipModel_gen_token(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.model,_=clip.load('RN50', args.device,jit=False)
        # self.model,_=clip.load('RN50x64', args.device,jit=False)
        self.model,_=clip.load('ViT-B/32', args.device,jit=False)

        self.args=args
        convert_weights(self.model)
        self.loss_func=torch.nn.BCELoss()
    def forward(self,img,txt):
        similarity=torch.zeros(txt.shape[0],txt.shape[1]).cuda()#to(self.args.device)
        for i in range(txt.shape[0]):
            with torch.no_grad():
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

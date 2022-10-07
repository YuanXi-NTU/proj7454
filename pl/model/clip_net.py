import torch
from torch import nn
# from torchvision.models import resnet50
import clip
from misc import ori_classes, classes, simp_classes
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
class ClipNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.model,_=clip.load('ViT-B/32', args.device,jit=False)
        self.model,_=clip.load('ViT-B/32',jit=False)
        self.args=args
        convert_weights(self.model)
        # self.clip_txt=torch.cat([clip.tokenize(f"an action of {c}") for c in ori_classes]).to(args.device)
        self.clip_txt=torch.cat([clip.tokenize(f"a person {c} something") for c in simp_classes])#.to(args.device)
        # self.clip_txt=torch.cat([clip.tokenize(f"{c} something") for c in simp_classes]).to(args.device)
        self.loss_function=torch.nn.MSELoss()
    def forward(self,img,txt):
        # import ipdb
        # ipdb.set_trace()
        similarity = torch.zeros(txt.shape[0], txt.shape[1]).cuda()
        # similarity = similarity.to(img.deivce).cuda()  # .to(self.args.device)
        for i in range(txt.shape[0]):
            with torch.no_grad():
                image_features = self.model.encode_image(img[i].unsqueeze(0))
                image_features2 = image_features / image_features.norm(dim=-1, keepdim=True)
            self.clip_txt = self.clip_txt.to(image_features2.device)

            # text_features = self.model.encode_text(self.clip_txt)
            text_features = self.model.encode_text(txt[i])
            text_features2 = text_features / text_features.norm(dim=-1, keepdim=True)
            # print(image_features2.device,text_features2.device)
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
        return self.loss_function(pred,label)
        # return torch.sum((pred-label)**2)
    def training_step(self, batch, batch_idx):
        img, labels,txt= batch
        # img, labels, filename = batch
        out = self(img,txt)
        loss = self.loss_function(out, labels)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, labels,txt= batch
        # img, labels, filename = batch
        out = self(img,txt)
        loss = self.loss_function(out, labels)
        label_digit = labels.argmax(axis=1)
        out_digit = out.argmax(axis=1)

        correct_num = sum(label_digit == out_digit).cpu().item()

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', correct_num/len(out_digit),
                 on_step=False, on_epoch=True, prog_bar=True)

        return (correct_num, len(out_digit))



# deprecated
# class ClipNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.classes=['over','in front of','beside','on','in','attached to'
#                     ,'hanging from','on the back of','falling off','going down',
#                       'painted on,walking on','running on','crossing','standing on',
#                       'lying on','sitting on','flying over','jumping over','jumping from',
#                       'wearing','holding','carrying','looking at','guiding','kissing',
#                       'eating','drinking','feeding','biting','catching','picking',
#                       'playing with','chasing','climbing','cleaning','playing',
#                       'touching','pushing','pulling','opening,cooking','talking to',
#                       'throwing','slicing,driving','riding','parked on','driving on',
#                 'about to hit','kicking','swinging,entering','exiting','enclosing','leaning on']
#         '''
#
#         body = [common.conv3x3(in_channel, hid, 3),
#                 nn.ReLU()]
#         for _ in range(layer_num-1):
#             body.append(common.conv3x3(hid, hid, 3))
#             body.append(nn.ReLU())
#
#         self.body = nn.Sequential(*body)
#         self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(hid * 6 * 6, 2048),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(2048, 2048),
#             nn.ReLU(inplace=True),
#             nn.Linear(2048, out_channel),
#             nn.Sigmoid()
#         )
#         '''
#         self.model,self.pre=clip.load('ViT-B/32')
#     def forward(self, x):
#         image_input=self.pre(x).unsqueeze(0)
#         txt_input=torch.cat([clip.tokenize(f"a photo of a {c}" for c in self.classes)])
#
#         with torch.no_grad():
#             img_feat=self.model.encode_image(image_input)
#             txt_feat=self.model.encode_text(txt_input)
#         img_feat/=img_feat.norm(dim=-1,keepdim=True)
#         txt_feat/=txt_feat.norm(dim=-1,keepdim=True)
#
#         sim=(100.0*img_feat@txt_feat.T).softmax(dim=-1)
#         val,idx=sim[0].topk(3)
#
#         return val

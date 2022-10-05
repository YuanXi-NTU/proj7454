import torch
from torch import nn
# from torchvision.models import resnet50
import clip

class ClipNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.classes=['over','in front of','beside','on','in','attached to'
                    ,'hanging from','on the back of','falling off','going down',
                      'painted on,walking on','running on','crossing','standing on',
                      'lying on','sitting on','flying over','jumping over','jumping from',
                      'wearing','holding','carrying','looking at','guiding','kissing',
                      'eating','drinking','feeding','biting','catching','picking',
                      'playing with','chasing','climbing','cleaning','playing',
                      'touching','pushing','pulling','opening,cooking','talking to',
                      'throwing','slicing,driving','riding','parked on','driving on',
                'about to hit','kicking','swinging,entering','exiting','enclosing','leaning on']
        '''
        body = [common.conv3x3(in_channel, hid, 3),
                nn.ReLU()]
        for _ in range(layer_num-1):
            body.append(common.conv3x3(hid, hid, 3))
            body.append(nn.ReLU())

        self.body = nn.Sequential(*body)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hid * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, out_channel),
            nn.Sigmoid()
        )
        '''
        self.model,self.pre=clip.load('ViT-B/32')
    def forward(self, x):
        image_input=self.pre(x).unsqueeze(0)
        txt_input=torch.cat([clip.tokenize(f"a photo of a {c}" for c in self.classes)])

        with torch.no_grad():
            img_feat=self.model.encode_image(image_input)
            txt_feat=self.model.encode_text(txt_input)
        img_feat/=img_feat.norm(dim=-1,keepdim=True)
        txt_feat/=txt_feat.norm(dim=-1,keepdim=True)

        sim=(100.0*img_feat@txt_feat.T).softmax(dim=-1)
        val,idx=sim[0].topk(3)

        import ipdb
        ipdb.set_trace()
        return val

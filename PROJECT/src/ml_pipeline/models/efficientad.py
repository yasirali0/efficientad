import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from ..ml_utils import imagenet_norm_batch
from .pdn import PDN_M, PDN_S
from .encoder_decoder import EncConv, DecConv



class Teacher(nn.Module):
    def __init__(self, size, with_bn=False, channel_size=384, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if size == 'M':
            self.pdn = PDN_M(last_kernel_size=channel_size, with_bn=with_bn)
        elif size == 'S':
            self.pdn = PDN_S(last_kernel_size=channel_size, with_bn=with_bn)
        # self.pdn.apply(weights_init)

    def forward(self, x):
        x = imagenet_norm_batch(x) #Comments on Algorithm 3: We use the image normalization of the pretrained models of torchvision [44].
        x = self.pdn(x)
        return x
    


class Student(nn.Module):
    def __init__(self, size, with_bn=False, channel_size=768, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if size == 'M':
            self.pdn = PDN_M(last_kernel_size=channel_size, with_bn=with_bn) #The student network has the same architecture, but 768 kernels instead of 384 in the Conv-5 and Conv-6 layers.
        elif size == 'S':
            self.pdn = PDN_S(last_kernel_size=channel_size, with_bn=with_bn) #The student network has the same architecture, but 768 kernels instead of 384 in the Conv-4 layer
        # self.pdn.apply(weights_init)

    def forward(self, x):
        x = imagenet_norm_batch(x) #Comments on Algorithm 3: We use the image normalization of the pretrained models of torchvision [44].
        pdn_out = self.pdn(x)
        return pdn_out



class AutoEncoder(nn.Module):
    def __init__(self, is_bn=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = EncConv()
        self.decoder = DecConv(is_bn=is_bn)

    def forward(self, x):
        x = imagenet_norm_batch(x) #Comments on Algorithm 3: We use the image normalization of the pretrained models of torchvision [44].
        x = self.encoder(x)
        x = self.decoder(x)
        return x



if __name__ == '__main__':
    from torchsummary import summary
    import torch
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoEncoder()
    model = model.to('cuda')
    # summary(model, (3, 256, 256))
    logger.debug(summary(model, (3, 256, 256)))
import tqdm
import torch


def imagenet_norm_batch(x: torch.Tensor) -> torch.Tensor:
    """Normalize batch of images with ImageNet mean and std.

    Args:
        x (torch.Tensor): Input batch.

    Returns:
        torch.Tensor: Normalized batch using the ImageNet mean and std.
    """
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].to(x.device)
    x_norm = (x - mean) / (std + 1e-11)
    
    return x_norm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


@torch.no_grad()
def global_channel_normalize(dataloader, pretrain_model, normalize_iter=1000, channel_size=384, resize=256):

    input_data = torch.randn(1, 3, resize, resize).cuda()
    temp_tensor = pretrain_model(input_data)
    x = torch.zeros((normalize_iter, channel_size, *temp_tensor.shape[2:]))
    num = 0
    for item in tqdm.tqdm(dataloader, desc='Global Channel Normalize'):
        if num >= normalize_iter:
            break
        ldist = item['image'].cuda()
        y = pretrain_model(ldist).detach().cpu()
        yb = y.shape[0]
        x[num:num+yb, :, :, :] = y[:, :, :, :]
        num += yb
    channel_mean = x[:num, :, :, :].mean(dim=(0, 2, 3), keepdim=True).cuda()
    channel_std = x[:num, :, :, :].std(dim=(0, 2, 3), keepdim=True).cuda()
    
    return channel_mean, channel_std
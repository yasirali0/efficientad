# https://github.com/rximg/EfficientAD/blob/main/distillation_training.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from loguru import logger

from ..models.wide_resnet import wide_resnet101_2
from ..models.efficientad import Teacher
from ...data_pipeline.datasets import ImageNetDataset, infinite_dataloader
from ..ml_utils import global_channel_normalize




class DistillationTraining(object):

    def __init__(self, imagenet_dir, channel_size, batch_size, save_path, normalize_iter, train_iter=10000, resize=512, model_size='S', 
                wide_resnet_101_arch="Wide_ResNet101_2_Weights.IMAGENET1K_V2", print_freq=25, with_bn=False) -> None:
        self.channel_size = channel_size
        self.mean = torch.empty(channel_size)
        self.std = torch.empty(channel_size)
        self.save_path = save_path
        self.imagenet_dir = imagenet_dir
        self.train_iter = train_iter
        self.model_size = model_size
        self.batch_size = batch_size
        self.normalize_iter = normalize_iter
        self.wide_resnet_101_arch = wide_resnet_101_arch
        self.print_freq = print_freq
        self.with_bn = with_bn
        self.resize = resize
        self.data_transforms = transforms.Compose([
                        transforms.Resize((resize, resize),),
                        transforms.RandomGrayscale(p=0.1), #6: Convert Idist to gray scale with a probability of 0.1 and 18: Convert Idist to gray scale with a probability of 0.1
                        transforms.ToTensor(),
                        ])
        
    def train(self):
        # load pretrain model
        self.load_pretrain()

        # load imagenet dataset
        imagenet_dataset = ImageNetDataset(self.imagenet_dir, self.data_transforms)
        dataloader = DataLoader(imagenet_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        dataloader = infinite_dataloader(dataloader)

        # initialize teacher model
        teacher = Teacher(self.model_size)
        teacher = teacher.cuda()

        self.mean, self.std = global_channel_normalize(dataloader, self.pretrain, self.normalize_iter, self.channel_size, self.resize)
        
        # optimizer and scheduler
        optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(0.95 * self.train_iter), gamma=0.1)
        
        best_loss = 1000
        loss_accum = 0

        logger.debug('start train iter:{}'.format(self.train_iter))
        for iteration in range(self.train_iter):

            batch_sample = next(dataloader).cuda()
            teacher.train()
            optimizer.zero_grad()
            loss = self.compute_mse_loss(teacher, batch_sample)
            loss.backward()
            optimizer.step()
            loss_accum += loss.item()
            scheduler.step()

            if (iteration+1) % self.print_freq == 0 and iteration > 100:
                loss_mean = loss_accum / self.print_freq
                logger.debug('iter:{}, loss:{:.4f}'.format(iteration, loss_mean))
                if loss_mean < best_loss or best_loss == 1000:
                    best_loss = loss_mean
                    
                    # save teacher
                    logger.debug('save best teacher at loss {}'.format(best_loss))
                    teacher.eval()
                    torch.save(teacher.state_dict(), '{}/best_teacher.pth'.format(self.save_path))
                loss_accum = 0

            # save teacher
            teacher.eval()
            torch.save(teacher.state_dict(), '{}/last_teacher.pth'.format(self.save_path))


    def load_pretrain(self):
        self.pretrain = wide_resnet101_2(self.wide_resnet_101_arch, pretrained=True)
        # self.pretrain.load_state_dict(torch.load('pretrained_model.pth'))
        self.pretrain.eval()
        self.pretrain = self.pretrain.cuda()
        # print(summary(self.pretrain, (3, 512, 512)))
        # logger.debug(summary(self.pretrain, (3, 512, 512))
    
    
    def compute_mse_loss(self,teacher,ldist):
        with torch.no_grad():
            y = self.pretrain(ldist) # torch.Size([8, 384, 64, 64])
            y = (y - self.mean) / self.std
        ldistresize = F.interpolate(ldist, size=(256, 256), mode='bilinear', align_corners=False)
        y0 = teacher(ldistresize)
        loss = F.mse_loss(y,y0)
        
        return loss

    
        
        

if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    imagenet_dir = os.path.join(os.getenv('DATA_DIR'), 'ImageNet')
    channel_size = 384
    model_size = 'S'
    save_path = os.path.join(os.getenv('MODEL_DIR'), model_size)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    distillation_training = DistillationTraining(
        imagenet_dir, channel_size, 16, save_path,
        normalize_iter=1000,
        model_size=model_size,
        train_iter=60000,
        wide_resnet_101_arch="Wide_ResNet101_2_Weights.IMAGENET1K_V2",
        )
    
    distillation_training.train()
import numpy as np
import torch
from torch import nn
from collections import OrderedDict
import albumentations as albu


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def remove_redundant_keys(state_dict: OrderedDict):
    # remove DataParallel wrapping
    if 'module' in list(state_dict.keys())[0]:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                # str.replace() can't be used because of unintended key removal (e.g. se-module)
                new_state_dict[k[7:]] = v
    else:
        new_state_dict = state_dict

    return new_state_dict


def save_checkpoint(path, model, epoch, optimizer=None, save_arch=False, params=None):
    attributes = {
        'epoch': epoch,
        'state_dict': remove_redundant_keys(model.state_dict()),
    }
    if optimizer is not None:
        attributes['optimizer'] = optimizer.state_dict()
    if save_arch:
        attributes['arch'] = model
    if params is not None:
        attributes['params'] = params

    try:
        torch.save(attributes, path)
    except TypeError:
        if 'arch' in attributes:
            print('Model architecture will be ignored because the architecture includes non-pickable objects.')
            del attributes['arch']
            torch.save(attributes, path)


def load_checkpoint(path, model=None, optimizer=None, params=False, epoch=False):
    resume = torch.load(path)
    rets = dict()

    if model is not None:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(remove_redundant_keys(resume['state_dict']))
        else:
            model.load_state_dict(remove_redundant_keys(resume['state_dict']))

        rets['model'] = model

    if optimizer is not None:
        optimizer.load_state_dict(resume['optimizer'])
        rets['optimizer'] = optimizer
    if params:
        rets['params'] = resume['params']
    if epoch:
        rets['epoch'] = resume['epoch']

    return rets


class TestTimeAugment(object):
    def __init__(self, model, times=4):
        self.model = model
        self.times = times
        self.augment = albu.Compose([
            albu.HorizontalFlip(p=.5), 
            albu.VerticalFlip(p=.5), 
            albu.ShiftScaleRotate(p=.9)
            ])
    
    def predict(self, inputs):
        device, dtype = inputs.device, inputs.dtype
        batch_size = inputs.size(0)

        outputs = self.model(inputs)
        for _ in range(self.times):
            images = np.zeros(shape=inputs.size())
            for i in range(batch_size):
                img = inputs[i].cpu().numpy().transpose(1, 2, 0)
                img = self.augment(image=img)['image']
                images[i] = img.transpose(2, 0, 1)  

            images = torch.from_numpy(images).to(device).to(dtype)
            outputs += self.model(images)
        
        return outputs / (self.times+1)
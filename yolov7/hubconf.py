"""PyTorch Hub models

Usage:
    import torch
    model = torch.hub.load('repo', 'model')
"""

from pathlib import Path

import torch

from yolov7.models.yolo import Model
from utils.general import check_requirements, set_logging
from utils.google_utils import attempt_download
from utils.torch_utils import select_device

dependencies = ['torch', 'yaml']
check_requirements(Path(__file__).parent / 'requirements.txt', exclude=('pycocotools', 'thop'))
set_logging()


def create(name, pretrained, channels, classes, autoshape):
    """Creates a specified model

    Arguments:
        name (str): name of model, i.e. 'yolov7'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes

    Returns:
        pytorch model
    """
    try:
        cfg = list((Path(__file__).parent / 'cfg').rglob(f'{name}.yaml'))[0]                   
        model = Model(cfg, channels, classes)
        if pretrained:
            fname = f'{name}.pt'                       
            attempt_download(fname)                                 
            ckpt = torch.load(fname, map_location=torch.device('cpu'))        
            msd = model.state_dict()                    
            csd = ckpt['model'].float().state_dict()                                 
            csd = {k: v for k, v in csd.items() if msd[k].shape == v.shape}          
            model.load_state_dict(csd, strict=False)        
            if len(ckpt['model'].names) == classes:
                model.names = ckpt['model'].names                             
            if autoshape:
                model = model.autoshape()                                          
        device = select_device('0' if torch.cuda.is_available() else 'cpu')                               
        return model.to(device)

    except Exception as e:
        s = 'Cache maybe be out of date, try force_reload=True.'
        raise Exception(s) from e


def custom(path_or_model='path/to/model.pt', autoshape=True):
    """custom mode

    Arguments (3 options):
        path_or_model (str): 'path/to/model.pt'
        path_or_model (dict): torch.load('path/to/model.pt')
        path_or_model (nn.Module): torch.load('path/to/model.pt')['model']

    Returns:
        pytorch model
    """
    model = torch.load(path_or_model, map_location=torch.device('cpu')) if isinstance(path_or_model, str) else path_or_model                   
    if isinstance(model, dict):
        model = model['ema' if model.get('ema') else 'model']              

    hub_model = Model(model.yaml).to(next(model.parameters()).device)          
    hub_model.load_state_dict(model.float().state_dict())                   
    hub_model.names = model.names               
    if autoshape:
        hub_model = hub_model.autoshape()                                          
    device = select_device('0' if torch.cuda.is_available() else 'cpu')                               
    return hub_model.to(device)


def yolov7(pretrained=True, channels=3, classes=80, autoshape=True):
    return create('yolov7', pretrained, channels, classes, autoshape)


if __name__ == '__main__':
    model = custom(path_or_model='yolov7.pt')                  
                                                                                                                  

                      
    import numpy as np
    from PIL import Image

    imgs = [np.zeros((640, 480, 3))]

    results = model(imgs)                     
    results.print()
    results.save()

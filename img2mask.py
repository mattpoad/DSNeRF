import torch
import numpy as np
from tqdm import tqdm

import torchvision.transforms.functional as F

import os

from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.utils import draw_segmentation_masks
from torchvision.io import read_image
from pathlib import Path

from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from torchvision.transforms.functional import convert_image_dtype

from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(3)



def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def display_mask(mask):
    dict_col = np.array(
        [
            [0, 0, 0],
            [0, 255, 0],
            [255, 0, 0],
            [0, 0, 255],
            [255, 255, 255],
            [96, 96, 96],
            [253, 96, 96],
            [255, 255, 0],
            [237, 127, 16],
            [102, 0, 153],
        ]
    )

    dict_col = np.array(
        [
            [0, 0, 0],  # black
            [0, 255, 0],  # green
            [255, 0, 0],  # red
            [0, 0, 255],  # blue
            [0, 255, 255],  # cyan
            [255, 255, 255],  # white
            [96, 96, 96],  # grey
            [255, 255, 0],  # yellow
            [237, 127, 16],  # orange
            [102, 0, 153],  # purple
            [88, 41, 0],  # brown
            [253, 108, 158],  # pink
            [128, 0, 0],  # maroon
            [255, 0, 255],
            [255, 0, 127],
            [0, 128, 255],
            [0, 102, 51],  # 17
            [192, 192, 192],
            [128, 128, 0],
            [84, 151, 120],
            [127, 255, 0]
        ]
    )

    try:
        len(mask.shape) == 2
    except AssertionError:
        print("Mask's shape is not 2")
    mask_dis = np.zeros((mask.shape[0], mask.shape[1], 3))
    # print('mask_dis shape',mask_dis.shape)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            mask_dis[i, j, :] = dict_col[mask[i, j]]
    return mask_dis


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument("--datadir", type=str, help='image directory')
    parser.add_argument("--item", type=str, help='type of labelised object')

    
    parser.add_argument("--transform", action='store_true', help='resize the mask')
    parser.add_argument("--width", type=int, help='width of the resized mask')
    parser.add_argument("--height", type=int, help='height of the resized mask')

    parser.add_argument("--network", type=str, default=deeplabv3_resnet50, help='the segmentation network used')
    
    return parser



def save_imgs(args, imgs, filename):

    if not isinstance(imgs, list):
        imgs = [imgs]
    
    path = args.datadir + '/masks/'
    mkdir(path)
    
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        img.save(path + 'mask' + filename[i][5:])


def segmentation_mask(args):

    print(f"...Generate {args.item} masks\n\n")
    
    path = args.datadir+'/images/'
    filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    img_list = [Image.open(path+f).convert('RGB') for f in filenames]
    img_list = [transforms.Compose([transforms.PILToTensor()])(img) for img in img_list]
    
    if args.transform:
        W = args.width
        H = args.height
        p = transforms.Resize((H, W))
        batch_int = torch.stack([p(img) for img in img_list])
    else:
        batch_int = torch.stack([img for img in img_list])

    batch = convert_image_dtype(batch_int, dtype=torch.float)

    batch = torch.Tensor(batch).to(device)

    N_batch = len(batch)//3
    r_batch = len(batch)%3
    
    model = args.network(pretrained=True, progress=False).to(device)
    model = model.eval()
    normalized_batch = F.normalize(batch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    sem_classes = [
        '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

    prev_i = 0
    for i in range(N_batch, len(batch), N_batch):

        output = model(normalized_batch[prev_i:i])['out']
        
        normalized_masks = torch.nn.functional.softmax(output, dim=1)
        
        class_dim = 1
        
        boolean_masks = (normalized_masks.argmax(class_dim) == sem_class_to_idx[args.item])
        boolean_masks = [m.cpu().float() for m in boolean_masks]

        save_imgs(args, boolean_masks, filenames[prev_i:i])
        prev_i = i

    if r_batch != 0:
        output = model(normalized_batch[-r_batch:])['out']

        normalized_masks = torch.nn.functional.softmax(output, dim=1)
        
        class_dim = 1
        
        boolean_masks = (normalized_masks.argmax(class_dim) == sem_class_to_idx[args.item])
        boolean_masks = [m.cpu().float() for m in boolean_masks]

        save_imgs(args, boolean_masks, filenames[-r_batch:])

    print(f"Done, masks generated with {str(args.network).split()[1]}")
    

if __name__=='__main__':

    parser = config_parser()
    args = parser.parse_args()
    
    segmentation_mask(args)

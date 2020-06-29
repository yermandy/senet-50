import numpy as np
from PIL import Image, ImageFile
import torch, os
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from warnings import filterwarnings
from time import time
import model.senet50_256 as senet50
import argparse


parser = argparse.ArgumentParser(description='SENet-50 trained on MS-Celeb-1M and fine-tuned on VGGFace2')
parser.add_argument('--batch_size', default=90, type=int, help='Batch size')
parser.add_argument('--workers', default=8, type=int, help='Workers number')
parser.add_argument('--dataset', required=True, type=str, help='Dataset name')
parser.add_argument('--bb_file', required=True, type=str, help='Bounding box file')
parser.add_argument('--bb_scale', default=0.5, type=int, help='Bounding box scale')
parser.add_argument('--cuda', default=0, type=int, help='Cuda device')
args = parser.parse_args()

dataset_name = args.dataset
bb_scale = args.bb_scale

ImageFile.LOAD_TRUNCATED_IMAGES = True
filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

os.makedirs(f'results/{dataset_name}_{bb_scale}/')


class ToTensor(object):
    def __call__(self, pil):
        return torch.Tensor(np.array(pil).transpose(2, 0, 1))

class ListDataset(torch.utils.data.Dataset):
    def __init__(self, images, bounding_boxes, scale=bb_scale):
        self.images = images
        self.bbs = bounding_boxes
        self.scale = scale
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.Resize((224, 224)),
            ToTensor(),
            transforms.Normalize(
                mean=[131.0912, 103.8827, 91.4953],
                std=[1,1,1]
            )
        ])

    def __getitem__(self, index):
        path  = self.images[index]
        bb    = self.bbs[index]
        image = Image.open(f"images/{dataset_name}/{path}")
        image = image.convert('RGB')
        image = self.crop_face(image, bb)
        image = self.transform(image)
        return image, path

    def __len__(self):
        return len(self.images)

    def crop_face(self, img, bb):
        x1, y1, x2, y2 = bb
        w_scale = ((x2 - x1) * self.scale) / 2
        h_scale = ((y2 - y1) * self.scale) / 2
        x1 -= int(w_scale)
        y1 -= int(h_scale)
        x2 += int(w_scale)
        y2 += int(h_scale)
        return img.crop((x1, y1, x2, y2))

def model(gpu=0):
    """
    Initialize model
    
    Returns
    -------
    model.senet50_256
        Model in evaluation mode for feature extraction
    """
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    model = senet50.senet50_256(weights_path='./model/senet50_256.pth').to(device)
    model.eval()
    cudnn.benchmark = True
    return model

def save_results(features, norms):
    np.save(f"results/{dataset_name}_{bb_scale}/features.npy", features)
    np.save(f"results/{dataset_name}_{bb_scale}/norms.npy", norms)

def feature_extraction(model, faces):
    """
    Extracts features from faces
    
    Parameters
    ----------
    model : senet50_256.Senet50_256
        Model to use for feature extraction
    faces : np.array
        Paths to photos, bounding boxes of faces in photos
        (N, 5): N – number of photos to process
    """
    paths  = faces[:, 0] # array containing paths to photos
    bbs    = faces[:, 1:5].astype(np.int) # array with bounding boxes
    n_imgs = len(paths)
    params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": True
    }

    dataset  = ListDataset(paths, bbs)
    loader   = torch.utils.data.DataLoader(dataset, **params)
    device   = next(model.parameters()).device
    features = np.empty((n_imgs, 256))
    norms    = np.empty((n_imgs))
    finished = 0
    
    save_each = 200

    for i, (imgs, paths) in enumerate(loader):
        run = time()
        imgs = imgs.to(device)
        f = model(imgs)[1].detach().cpu().numpy()[:, :, 0, 0]
        start = i * args.batch_size
        finished += f.shape[0]
        norm = np.linalg.norm(f, axis=1, keepdims=True)
        f = f / norm
        features[start:finished] = f
        norms[start:finished] = norm[:, 0]
        if i % save_each == 0:
            save_results(features, norms)
        print(f"-> processed {finished}/{n_imgs} images in {time() - run:.4f} seconds")

    save_results(features, norms)

    return features, norms


if __name__ == '__main__':
    model = model(gpu=args.cuda)
    faces = np.genfromtxt(f'resources/{args.bb_file}', dtype=np.str, delimiter=',')

    print(f"Bounding box scale: {bb_scale}")
    print(f"Dataset: {dataset_name}")
    print(f"File with faces and bounding boxes: {args.bb_file}")
    feature_extraction(model, faces)
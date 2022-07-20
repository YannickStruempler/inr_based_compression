'''Implementaion of Datasets based on https://github.com/vsitzmann/siren'''
import csv
import glob
import math
import os

import matplotlib.colors as colors
import numpy as np
import scipy.ndimage
import scipy.special
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

from lib.nglod.lib.torchgp import point_sample, sample_surface, load_obj, compute_sdf


def get_shape_dataset(name, data_root):
    if name == 'armadillo':
        shapeglob = glob.glob(os.path.join(data_root, name, 'armadillo_normalized.obj'))
        data_list = zip([name], shapeglob)
    elif name == 'stanford':
        shapeglob = glob.glob(os.path.join(data_root, name, "*_normalized.obj"))
        shape_names = [shape_path.split('/')[-1].split('.')[0] for shape_path in shapeglob]
        data_list = zip(shape_names, shapeglob)
    elif name == 'ShapeNet50':
        shapeglob = glob.glob(os.path.join(data_root, name, '*/*/*/model_normalized.obj'))
        shape_names = ["_".join(shape_path.split('/')[-4:-2]) for shape_path in shapeglob]
        data_list = zip(shape_names, shapeglob)
    else:
        raise ValueError('Unknown dataset name')

    return data_list


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)


def grads2img(gradients):
    mG = gradients.detach().squeeze(0).permute(-2, -1, -3).cpu()

    # assumes mG is [row,cols,2]
    nRows = mG.shape[0]
    nCols = mG.shape[1]
    mGr = mG[:, :, 0]
    mGc = mG[:, :, 1]
    mGa = np.arctan2(mGc, mGr)
    mGm = np.hypot(mGc, mGr)
    mGhsv = np.zeros((nRows, nCols, 3), dtype=np.float32)
    mGhsv[:, :, 0] = (mGa + math.pi) / (2. * math.pi)
    mGhsv[:, :, 1] = 1.

    nPerMin = np.percentile(mGm, 5)
    nPerMax = np.percentile(mGm, 95)
    mGm = (mGm - nPerMin) / (nPerMax - nPerMin)
    mGm = np.clip(mGm, 0, 1)

    mGhsv[:, :, 2] = mGm
    mGrgb = colors.hsv_to_rgb(mGhsv)
    return torch.from_numpy(mGrgb).permute(2, 0, 1)


def rescale_img(x, mode='scale', perc=None, tmax=1.0, tmin=0.0):
    if (mode == 'scale'):
        if perc is None:
            xmax = torch.max(x)
            xmin = torch.min(x)
        else:
            xmin = np.percentile(x.detach().cpu().numpy(), perc)
            xmax = np.percentile(x.detach().cpu().numpy(), 100 - perc)
            x = torch.clamp(x, xmin, xmax)
        if xmin == xmax:
            return 0.5 * torch.ones_like(x) * (tmax - tmin) + tmin
        x = ((x - xmin) / (xmax - xmin)) * (tmax - tmin) + tmin
    elif (mode == 'clamp'):
        x = torch.clamp(x, 0, 1)
    return x


class ImageFile(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.img = Image.open(filename)
        self.img_channels = len(self.img.mode)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.img


class DIV2K(Dataset):
    def __init__(self, split, data_root, downsampled=True, max_len=None):
        super().__init__()
        assert split in ['train', 'val'], "Unknown split"

        self.root = os.path.join(data_root, 'DIV2K')
        self.img_channels = 3
        self.fnames = []
        self.file_type = '.png'
        self.size = (768, 512)

        if split == 'train':
            for i in range(0, 800):
                self.fnames.append("DIV2K_train_HR/{:04d}.png".format(i + 1))
        elif split == 'val':
            for i in range(800, 900):
                self.fnames.append("DIV2K_valid_HR/{:04d}.png".format(i + 1))
        self.downsampled = downsampled

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.fnames[idx])
        img = Image.open(path)
        if self.downsampled:
            width, height = img.size  # Get dimensions
            if height > width: img = img.rotate(90, expand=1)
            img.thumbnail(self.size, Image.ANTIALIAS)
        return img


class CelebA(Dataset):
    def __init__(self, split, data_root, downsampled=False, max_len=None):
        # SIZE (178 x 218)
        super().__init__()
        assert split in ['train', 'test', 'val'], "Unknown split"

        self.root = os.path.join(data_root, 'CelebA', 'img_align_celeba_png')
        self.img_channels = 3
        self.fnames = []
        self.file_type = '.png'
        self.size = (178, 218)

        with open(os.path.join(data_root, 'CelebA', 'list_eval_partition.csv'), newline='') as csvfile:
            rowreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            i = 0
            for row in rowreader:
                if max_len and i >= max_len: break
                if split == 'train' and row[1] == '0':
                    self.fnames.append(row[0].split('.')[0])
                elif split == 'val' and row[1] == '1':
                    self.fnames.append(row[0].split('.')[0])
                    i += 1
                elif split == 'test' and row[1] == '2':
                    self.fnames.append(row[0].split('.')[0])

        self.downsampled = downsampled

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.fnames[idx] + self.file_type)
        img = Image.open(path)
        if self.downsampled:
            width, height = img.size  # Get dimensions

            s = min(width, height)
            left = (width - s) / 2
            top = (height - s) / 2
            right = (width + s) / 2
            bottom = (height + s) / 2
            img = img.crop((left, top, right, bottom))
            img = img.resize((32, 32))

        return img


class Implicit2DWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, sidelength=None, compute_diff=None):

        if isinstance(sidelength, int):
            sidelength = (sidelength, sidelength)
        self.sidelength = sidelength

        self.transform = Compose([
            Resize(sidelength),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

        self.compute_diff = compute_diff
        self.dataset = dataset
        self.mgrid = get_mgrid(sidelength)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.transform(self.dataset[idx])

        if self.compute_diff == 'gradients':
            img *= 1e1
            gradx = scipy.ndimage.sobel(img.numpy(), axis=1).squeeze(0)[..., None]
            grady = scipy.ndimage.sobel(img.numpy(), axis=2).squeeze(0)[..., None]
        elif self.compute_diff == 'laplacian':
            img *= 1e4
            laplace = scipy.ndimage.laplace(img.numpy()).squeeze(0)[..., None]
        elif self.compute_diff == 'all':
            gradx = scipy.ndimage.sobel(img.numpy(), axis=1).squeeze(0)[..., None]
            grady = scipy.ndimage.sobel(img.numpy(), axis=2).squeeze(0)[..., None]
            laplace = scipy.ndimage.laplace(img.numpy()).squeeze(0)[..., None]

        img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

        in_dict = {'idx': idx, 'coords': self.mgrid}
        gt_dict = {'img': img}

        if self.compute_diff == 'gradients':
            gradients = torch.cat((torch.from_numpy(gradx).reshape(-1, 1),
                                   torch.from_numpy(grady).reshape(-1, 1)),
                                  dim=-1)
            gt_dict.update({'gradients': gradients})

        elif self.compute_diff == 'laplacian':
            gt_dict.update({'laplace': torch.from_numpy(laplace).view(-1, 1)})

        elif self.compute_diff == 'all':
            gradients = torch.cat((torch.from_numpy(gradx).reshape(-1, 1),
                                   torch.from_numpy(grady).reshape(-1, 1)),
                                  dim=-1)
            gt_dict.update({'gradients': gradients})
            gt_dict.update({'laplace': torch.from_numpy(laplace).view(-1, 1)})

        return in_dict, gt_dict

    def get_item_small(self, idx):
        img = self.transform(self.dataset[idx])
        spatial_img = img.clone()
        img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

        gt_dict = {'img': img}

        return spatial_img, img, gt_dict


def setparam(args, param, paramstr):
    argsparam = getattr(args, paramstr, None)
    if param is not None or argsparam is None:
        return param
    else:
        return argsparam


def save_obj(fname, vertices, faces):
    """Save to Wavefront .OBJ.

    Args:
        fname (str): filename
        vertices (torch.Tensor): [N_vertices, 3]
        vertices (torch.Tensor): [N_faces, 3]
    """

    assert fname.endswith('.obj'), 'Filename must end with .obj'
    with open(fname, 'w') as f:
        for vert in vertices:
            f.write('v %f %f %f\n' % tuple(vert))
        for face in faces:
            f.write('f %d %d %d\n' % tuple(face + 1))


class MeshDataset(Dataset):
    """Base class for single mesh datasets."""

    def __init__(self,
                 args=None,
                 dataset_path=None,
                 raw_obj_path=None,
                 sample_mode=None,
                 get_normals=None,
                 seed=None,
                 num_samples=None,
                 trim=None,
                 sample_tex=None
                 ):
        self.args = args
        self.dataset_path = setparam(args, dataset_path, 'dataset_path')
        self.raw_obj_path = setparam(args, raw_obj_path, 'raw_obj_path')
        self.sample_mode = setparam(args, sample_mode, 'sample_mode')
        self.get_normals = setparam(args, get_normals, 'get_normals')
        self.num_samples = setparam(args, num_samples, 'num_samples')
        self.trim = setparam(args, trim, 'trim')
        self.sample_tex = setparam(args, sample_tex, 'sample_tex')

        # Possibly remove... or fix trim obj
        # if self.raw_obj_path is not None and not os.path.exists(self.dataset_path):
        #    _, _, self.mesh = trim_obj_to_file(self.raw_obj_path, self.dataset_path)
        # elif not os.path.exists(self.dataset_path):
        #    assert False and "Data does not exist and raw obj file not specified"
        # else:

        if self.sample_tex:
            out = load_obj(self.dataset_path, load_materials=True)
            self.V, self.F, self.texv, self.texf, self.mats = out
        else:
            self.V, self.F = load_obj(self.dataset_path)

        # self.V, self.F = normalize(self.V, self.F)
        # save_obj(dataset_path.split('.')[0] + '_normalized.obj', self.V, self.F)
        self.mesh = self.V[self.F]
        self.resample()

    def resample(self):
        """Resample SDF samples."""

        self.nrm = None
        if self.get_normals:
            self.pts, self.nrm = sample_surface(self.V, self.F, self.num_samples * 5)
            self.nrm = self.nrm.cpu()
        else:
            self.pts = point_sample(self.V, self.F, self.sample_mode, self.num_samples)

        self.d = compute_sdf(self.V.cuda(), self.F.cuda(), self.pts.cuda())

        self.d = self.d[..., None]
        self.d = self.d.cpu()
        self.pts = self.pts.cpu()

    def __getitem__(self, idx: int):
        """Retrieve point sample."""
        if self.get_normals:
            return {'idx': idx, 'coords': self.pts[idx]}, {'dist': self.d[idx], 'normal': self.nrm[idx]}
        elif self.sample_tex:
            return {'idx': idx, 'coords': self.pts[idx]}, {'dist': self.d[idx], 'rgb': self.rgb[idx]}
        else:
            return {'idx': idx, 'coords': self.pts[idx]}, {'dist': self.d[idx]}

    def __len__(self):
        """Return length of dataset (number of _samples_)."""

        return self.pts.size()[0]

    def num_shapes(self):
        """Return length of dataset (number of _mesh models_)."""

        return 1

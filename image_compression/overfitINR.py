"""Script for overfitting an INR starting from a random intialization (basic approach)."""
import glob
import json
import os
import statistics
from functools import partial

import PIL
import numpy as np
import torch
import yaml
from absl import app
from absl import flags
from torch.utils.data import DataLoader

import dataio
import losses
import modules
import training
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: %s' % device)
# Define Flags.
flags.DEFINE_string('data_root',
                    'data',
                    'Root directory of data.')
flags.DEFINE_string('exp_root',
                    'exp',
                    'Root directory of experiments.')
flags.DEFINE_enum('dataset', 'KODAK',
                  ['KODAK', 'CelebA100', 'KODAKcropped'],
                  'Dataset used during training.')
flags.DEFINE_integer('epochs',
                     10000,
                     'Maximum number of epochs.',
                     lower_bound=1)
flags.DEFINE_float('lr',
                   0.0001,
                   'Learning rate used during training.',
                   lower_bound=0.0)
flags.DEFINE_float('l1_reg',
                   0.0,
                   'L1 weight regularization.',
                   lower_bound=0.0)

flags.DEFINE_enum('activation',
                  'sine',
                  ['sine', 'relu'],
                  'Activation Function.')

flags.DEFINE_enum('encoding', 'nerf', ['mlp', 'nerf', 'positional', 'gauss'],
                  'Input encoding type used: '
                  ' mlp = basic mlp with no input encoding'
                  ' nerf = our positional encoding derived from NeRF (Mildenhall[2020]),'
                  ' positional = positional encoding suggested by Tancik[2020],'
                  ' gaussian = Gaussian encoding suggested by Tancik[2020]')

flags.DEFINE_integer('hidden_dims',
                     64,
                     'Hidden dimension of fully-connected neural network.',
                     lower_bound=1)
flags.DEFINE_integer('hidden_layers',
                     3,
                     'Number of hidden layer of ' +
                     'fully-connected neural network.',
                     lower_bound=1)
flags.DEFINE_integer('downscaling_factor',
                     1,
                     'Factor by which in input is downsampled',
                     lower_bound=1)
flags.DEFINE_integer('ff_dims',
                     16,
                     'Number of fourier feature frequencies for input encoding at different scales')
flags.DEFINE_integer('patience',
                     500,
                     'patience of lr schedule',
                     lower_bound=1)
flags.DEFINE_float('encoding_scale',
                   1.4,
                   'Standard deviation of the encoder')
flags.DEFINE_integer('epochs_til_ckpt',
                     1000,
                     'Number of epochs until checkpoint is saved.')
flags.DEFINE_integer('steps_til_summary',
                     1000,
                     'Number of steps until tensorboard summary is saved.')

FLAGS = flags.FLAGS


def main(_):
    imglob = glob.glob(os.path.join(FLAGS.data_root, FLAGS.dataset, '*'))
    mses = {}
    psnrs = {}
    ssims = {}
    experiment_folder = utils.get_base_overfitting_experiment_folder(FLAGS)
    # save FLAGS to yml
    yaml.dump(FLAGS.flag_values_dict(), open(os.path.join(experiment_folder, 'FLAGS.yml'), 'w'))

    for i, im in enumerate(imglob):
        print('Image: ' + str(i))
        image_name = im.split('/')[-1].split('.')[0]
        img_dataset = dataio.ImageFile(im)
        img = PIL.Image.open(im)
        image_resolution = (img.size[1] // FLAGS.downscaling_factor, img.size[0] // FLAGS.downscaling_factor)
        root_path = os.path.join(experiment_folder, image_name)
        if os.path.exists(os.path.join(root_path, 'checkpoints', 'model_final.pth')):
            print("Skipping ", root_path)
            continue

        coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=image_resolution)
        dataloader = DataLoader(coord_dataset, shuffle=True, pin_memory=True,
                                num_workers=0)

        # Define the model.
        model = modules.INRNet(type=FLAGS.activation, mode=FLAGS.encoding, sidelength=image_resolution,
                               out_features=img_dataset.img_channels, hidden_features=FLAGS.hidden_dims,
                               num_hidden_layers=FLAGS.hidden_layers, encoding_scale=FLAGS.encoding_scale,
                               ff_dims=FLAGS.ff_dims)

        model.to(device)
        root_path = os.path.join(experiment_folder, image_name)

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Params ", params)

        # Define the loss
        loss_fn = partial(losses.image_mse, None)

        summary_fn = partial(utils.write_image_summary, image_resolution)
        l1_loss_fn = losses.model_l1

        training.train(model=model, train_dataloader=dataloader, epochs=FLAGS.epochs, lr=FLAGS.lr,
                       steps_til_summary=FLAGS.steps_til_summary, epochs_til_checkpoint=FLAGS.epochs_til_ckpt,
                       model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, l1_reg=FLAGS.l1_reg,
                       l1_loss_fn=l1_loss_fn, patience=FLAGS.patience)

        model.eval()
        mse, ssim, psnr = utils.check_metrics_full(dataloader, model, image_resolution)
        mses[image_name] = mse
        psnrs[image_name] = psnr
        ssims[image_name] = ssim
    # Collect and average metrics over all images in the dataset
    metrics = {'mse': mses, 'psnr': psnrs, 'ssim': ssims, 'avg_mse': statistics.mean(mses.values()),
               'avg_psnr': statistics.mean(psnrs.values()), 'avg_ssim': statistics.mean(ssims.values())
               }

    with open(os.path.join(experiment_folder, 'result.json'), 'w') as fp:
        json.dump(metrics, fp)


if __name__ == '__main__':
    app.run(main)

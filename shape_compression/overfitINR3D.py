"""Training script."""

import json
import os
from functools import partial

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
from losses import sdf_mse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: %s' % device)
# Define Flags.
flags.DEFINE_string('data_root',
                    'data',
                    'Root directory of data.')
flags.DEFINE_string('exp_root',
                    'exp',
                    'Root directory of experiments.')
flags.DEFINE_enum('dataset', 'armadillo',
                  ['ShapeNet50', 'armadillo', 'stanford'],
                  'Dataset used during training.')
flags.DEFINE_integer('batch_size',
                     10000,
                     'Batch size used during training.',
                     lower_bound=1)
flags.DEFINE_integer('epochs',
                     250,
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
                     32,
                     'Hidden dimension of fully-connected neural network.',
                     lower_bound=1)
flags.DEFINE_integer('hidden_layers',
                     3,
                     'Number of hidden layer of ' +
                     'fully-connected neural network.',
                     lower_bound=1)
flags.DEFINE_integer('ff_dims',
                     16,
                     'Number of fourier feature frequencies for input encoding at different scales')
flags.DEFINE_integer('patience',
                     500,
                     'patience of lr schedule',
                     lower_bound=1)
flags.DEFINE_integer('samples_per_shape',
                     100000,
                     'Number of surface samples per shape used for overfitting ',
                     lower_bound=1)
flags.DEFINE_integer('epochs_til_ckpt',
                     10,
                     'Number of epochs until checkpoint is saved.')
flags.DEFINE_integer('steps_til_summary',
                     1000,
                     'Number of steps until tensorboard summary is saved.')
flags.DEFINE_float('encoding_scale',
                   1.4,
                   'Standard deviation of the encoder')
FLAGS = flags.FLAGS


def main(_):
    experiment_folder = utils.get_base_overfitting_experiment_folder(FLAGS)

    # save FLAGS to yml
    yaml.dump(FLAGS.flag_values_dict(), open(os.path.join(experiment_folder, 'FLAGS.yml'), 'w'))

    data_list = dataio.get_shape_dataset(FLAGS.dataset, FLAGS.data_root)
    for i, (shape_name, shape_path) in enumerate(data_list):
        mesh_dataset = dataio.MeshDataset(dataset_path=shape_path, num_samples=FLAGS.samples_per_shape,
                                          sample_mode=['rand', 'near', 'near', 'trace', 'trace'])

        dataloader = DataLoader(mesh_dataset, shuffle=True, batch_size=FLAGS.batch_size, pin_memory=True, num_workers=8)

        # Define the model.
        model = modules.INRNet(type=FLAGS.activation, mode=FLAGS.encoding,
                               out_features=1, hidden_features=FLAGS.hidden_dims,
                               num_hidden_layers=FLAGS.hidden_layers, encoding_scale=FLAGS.encoding_scale,
                               ff_dims=FLAGS.ff_dims, in_features=3)
        model.to(device)
        root_path = os.path.join(experiment_folder, shape_name)

        # Define the loss
        loss_fn = sdf_mse
        summary_fn = partial(utils.write_sdf_summary, shape_path)

        l1_loss_fn = losses.model_l1
        training.train(model=model, train_dataloader=dataloader, epochs=FLAGS.epochs, lr=FLAGS.lr,
                       steps_til_summary=FLAGS.steps_til_summary, epochs_til_checkpoint=FLAGS.epochs_til_ckpt,
                       model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, l1_reg=FLAGS.l1_reg,
                       patience=FLAGS.patience, l1_loss_fn=l1_loss_fn)

        metrics = utils.check_metrics_sdf(shape_path, model)

        mse = utils.check_mse_sdf(mesh_dataset, utils.convert_to_nn_module(model))
        metrics = {'vol_iou': metrics[0], 'chamfer': metrics[1], 'mse': mse}
        with open(os.path.join(root_path, 'result.json'), 'w') as fp:
            json.dump(metrics, fp)


if __name__ == '__main__':
    app.run(main)

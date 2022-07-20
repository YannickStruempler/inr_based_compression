import copy
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

import utils

sys.path.append("siren/torchmeta")
sys.path.append("siren")
import torch
from Quantizer import SDFQuantizer

# @title Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: %s' % device)
import torch
import os
import dataio, modules
from absl import app
from absl import flags
import glob
from torch.utils.data import DataLoader
from utils import check_metrics_sdf, check_mse_sdf
from quantize_utils import convert_to_nn_module


def mse_func(a, b):
    return np.mean((np.array(a, dtype='float32') - np.array(b, dtype='float32')) ** 2)


flags.DEFINE_string('data_root',
                    'data',
                    'Root directory of data.')
flags.DEFINE_string('exp_root',
                    'exp',
                    'Root directory of experiments.')
flags.DEFINE_string('exp_glob',
                    '*',
                    'regular expression to match experiment name')
flags.DEFINE_enum('dataset', 'armadillo',
                  ['ShapeNet50', 'armadillo', 'stanford'],
                  'Dataset used during retraining.')

flags.DEFINE_integer('retrain_epochs',
                     1000,
                     'Maximum number of epochs during retraining.',
                     lower_bound=1)
flags.DEFINE_float('retrain_lr',
                   1e-06,
                   'Learning rate used during retraining.',
                   lower_bound=0.0)
flags.DEFINE_float('l1_reg',
                   0.0,
                   'L1 weight regularization used during retraining.',
                   lower_bound=0.0)
flags.DEFINE_float('entropy_reg',
                   0.0,
                   'entropy regularization used during retraining.',
                   lower_bound=0.0)
flags.DEFINE_integer('bitwidth',
                     8,
                     'bitwidth used for Quantization',
                     lower_bound=1)
flags.DEFINE_bool('adaround',
                  True,
                  'use adative rounding post quanitzatition')
flags.DEFINE_bool('retrain',
                  True,
                  'use retraining post quanitzatition')
flags.DEFINE_bool('skip_existing',
                  True,
                  'skip_existing_conifigs')
flags.DEFINE_float('adaround_reg', 0.0001, 'regularizing parameter for adaround')
flags.DEFINE_integer('adaround_iterations', 2000, 'Number of adaround iterations')
flags.DEFINE_enum('code', 'brotli',
                  ['brotli', 'arithmetic'],
                  'Algorithm for lossless compression of the final bytestream')
FLAGS = flags.FLAGS


class AimetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (self.dataset[idx][0]['coords'].unsqueeze(0), self.dataset[idx][1]['img'])


def get_quant_config_name():
    name = 'bw' + str(FLAGS.bitwidth)

    if FLAGS.adaround:
        name = '_'.join(
            [name, 'adaround_iter', str(FLAGS.adaround_iterations), 'adaround_reg', str(FLAGS.adaround_reg)])
    if FLAGS.retrain:
        name = '_'.join([name, 'retrain_epochs' + str(FLAGS.retrain_epochs), 'retrain_lr' + str(FLAGS.retrain_lr)])
    return str(FLAGS.code) + '_' + name, name


def main(_):
    data_list = dataio.get_shape_dataset(FLAGS.dataset, FLAGS.data_root)
    exp_glob = glob.glob(os.path.join(FLAGS.exp_root, FLAGS.exp_glob))
    count = 0
    for exp_folder in exp_glob:
        count += 1
        print('Experiment ' + str(count) + '/' + str(len(exp_glob)))

        TRAINING_FLAGS = yaml.safe_load(open(os.path.join(exp_folder, 'FLAGS.yml'), 'r'))
        name, model_name = get_quant_config_name()

        for shape_name, shape_path in data_list:
            if os.path.isfile(
                    os.path.join(exp_folder, shape_name, 'metrics_' + name + '.yml')) and FLAGS.skip_existing:
                print('Skipped ' + exp_folder + shape_name)

                continue
            try:
                state_dict = torch.load(os.path.join(exp_folder, shape_name + '/checkpoints/model_best_.pth'),
                                        map_location='cpu')
            except:
                print('Did not find model checkpoint -> continue.')
                continue

            mesh_dataset = dataio.MeshDataset(dataset_path=shape_path, num_samples=TRAINING_FLAGS['samples_per_shape'],
                                              sample_mode=['rand', 'near', 'near', 'trace', 'trace'])

            dataloader = DataLoader(mesh_dataset, shuffle=True, batch_size=TRAINING_FLAGS['batch_size'],
                                    pin_memory=True,
                                    num_workers=8)

            model = modules.INRNet(type=TRAINING_FLAGS['activation'], mode=TRAINING_FLAGS['encoding'],
                                   out_features=1, in_features=3,
                                   hidden_features=TRAINING_FLAGS['hidden_dims'],
                                   num_hidden_layers=TRAINING_FLAGS['hidden_layers'],
                                   encoding_scale=TRAINING_FLAGS['encoding_scale'],
                                   ff_dims=TRAINING_FLAGS['ff_dims'])

            model = model.to(device)
            model.load_state_dict(state_dict, strict=True)
            # res = check_metrics_sdf(mesh_dataset.dataset_path, model)
            res = check_mse_sdf(mesh_dataset, convert_to_nn_module(model))
            print(exp_folder, shape_name)
            print('Before Quantization MSE: ', res)
            try:
                ref_state_dict = torch.load(os.path.join(exp_folder, 'model_maml.pth'),
                                            map_location='cpu')
                ref_model = copy.deepcopy(model)
                ref_model.load_state_dict(ref_state_dict, strict=True)
            except:
                ref_model = None
            if TRAINING_FLAGS['model_type'] == 'mlp':
                if ref_model:
                    offset_model = convert_to_nn_module_with_offset(model, ref_model)
                model = convert_to_nn_module(model)
            else:
                model = convert_to_nn_module_in_place(model)
                model.use_meta = False

            if ref_model:
                quant = SDFQuantizer(offset_model, mesh_dataset, dataloader, FLAGS.bitwidth, device, exp_folder)
            else:
                quant = SDFQuantizer(model, mesh_dataset, dataloader, FLAGS.bitwidth, device, exp_folder)

            model_quantized, metrics, bytes, state_dict = quant.compress_model(
                retrain=FLAGS.retrain, epochs=FLAGS.retrain_epochs,
                lr=FLAGS.retrain_lr, ref_model=ref_model,
                adaround=FLAGS.adaround,
                adaround_iterations=FLAGS.adaround_iterations,
                adaround_reg=FLAGS.adaround_reg,
                difference_encoding='parallel',
                code=FLAGS.code)
            model.load_state_dict(state_dict, strict=True)
            mse = check_mse_sdf(mesh_dataset, model)
            metrics = check_metrics_sdf(mesh_dataset.dataset_path, model)
            vol_iou, chamfer = metrics
            print('Final metrics: Vol. IOU {}, Chamfer Distance {}'.format(vol_iou, chamfer))
            metrics_dict = {'vol_iou': vol_iou, 'chamfer': chamfer, 'mse': mse, 'bytes': bytes}
            metrics_dict = {**metrics_dict, **TRAINING_FLAGS, **FLAGS.flag_values_dict()}

            normal, rgb = utils.render_sdf(model, verbose=True)

            plt.imsave(os.path.join(exp_folder, shape_name, 'metrics_' + name + '_normal.png'), normal)
            plt.imsave(os.path.join(exp_folder, shape_name, 'metrics_' + name + '_rgb.png'), rgb)
            name, model_name = get_quant_config_name()
            yaml.dump(metrics_dict, open(os.path.join(exp_folder, shape_name, 'metrics_' + name + '.yml'), 'w'))
            torch.save(model.state_dict(),
                       os.path.join(exp_folder, shape_name, 'model_' + model_name + '.pth'))


if __name__ == '__main__':
    app.run(main)

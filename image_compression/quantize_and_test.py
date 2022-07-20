import copy
import glob
import os

import torch
import yaml
from absl import app
from absl import flags
from torch.utils.data import DataLoader

import dataio
import modules
from Quantizer import ImageQuantizer
from quantize_utils import convert_to_nn_module, convert_to_nn_module_with_offset
from utils import check_metrics, check_metrics_full

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

flags.DEFINE_string('data_root',
                    'data',
                    'Root directory of data.')
flags.DEFINE_string('exp_root',
                    'exp',
                    'Root directory of experiments.')
flags.DEFINE_string('exp_glob',
                    '*',
                    'regular expression to match experiment name')
flags.DEFINE_enum('dataset', 'KODAK',
                  ['KODAK', 'CelebA100'],
                  'Dataset used during retraining.')
flags.DEFINE_integer('retrain_epochs',
                     300,
                     'Maximum number of epochs during retraining.',
                     lower_bound=1)
flags.DEFINE_float('retrain_lr',
                   1e-06,
                   'Learning rate used during retraining.',
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
flags.DEFINE_integer('adaround_iterations', 1000, 'Number of adaround iterations')
flags.DEFINE_enum('code', 'arithmetic',
                  ['brotli', 'arithmetic'],
                  'Algorithm for lossless compression of the final bytestream')
FLAGS = flags.FLAGS


def get_quant_config_name():
    name = 'bw' + str(FLAGS.bitwidth)

    if FLAGS.adaround:
        name = '_'.join(
            [name, 'adaround_iter', str(FLAGS.adaround_iterations), 'adaround_reg', str(FLAGS.adaround_reg)])
    if FLAGS.retrain:
        name = '_'.join([name, 'retrain_epochs' + str(FLAGS.retrain_epochs), 'retrain_lr' + str(FLAGS.retrain_lr)])
    return str(FLAGS.code) + '_' + name, name


def main(_):
    imglob = glob.glob(os.path.join(FLAGS.data_root, FLAGS.dataset, '*'))

    exp_glob = glob.glob(os.path.join(FLAGS.exp_root, FLAGS.exp_glob))
    print(os.path.join(FLAGS.exp_root, FLAGS.exp_glob))
    count = 0
    print(exp_glob)
    for exp_folder in exp_glob:
        count += 1
        print('Experiment ' + str(count) + '/' + str(len(exp_glob)))

        TRAINING_FLAGS = yaml.safe_load(open(os.path.join(exp_folder, 'FLAGS.yml'), 'r'))
        name, model_name = get_quant_config_name()

        for im in imglob:
            image_name = im.split('/')[-1].split('.')[0]

            if os.path.isfile(os.path.join(exp_folder, image_name, 'metrics_' + name + '.yml')) and FLAGS.skip_existing:
                print('Skipped ' + exp_folder + image_name)
                continue

            img = dataio.ImageFile(im)
            if img.img.size[1] > img.img.size[0] and 'maml_iterations' in TRAINING_FLAGS and TRAINING_FLAGS[
                'dataset'] == 'KODAK':
                img.img = img.img.rotate(90, expand=1)

            img_dataset = img
            scale = TRAINING_FLAGS['downscaling_factor']

            image_resolution = (img.img.size[1] // scale, img.img.size[0] // scale)
            coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=image_resolution)

            dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=1, pin_memory=True,
                                    num_workers=0)

            model = modules.INRNet(type=TRAINING_FLAGS['activation'], mode=TRAINING_FLAGS['encoding'],
                                   sidelength=image_resolution,
                                   out_features=img_dataset.img_channels,
                                   hidden_features=TRAINING_FLAGS['hidden_dims'],
                                   num_hidden_layers=TRAINING_FLAGS['hidden_layers'],
                                   encoding_scale=TRAINING_FLAGS['encoding_scale'],
                                   ff_dims=TRAINING_FLAGS['ff_dims'])

            model = model.to(device)
            state_dict = torch.load(os.path.join(exp_folder, image_name + '/checkpoints/model_best_.pth'),
                                    map_location='cpu')

            model.load_state_dict(state_dict, strict=True)
            res = check_metrics_full(dataloader, model, image_resolution)
            print(exp_folder, image_name)
            print('Before Quantization: ', res)
            try:
                ref_state_dict = torch.load(os.path.join(exp_folder, 'model_maml.pth'),
                                            map_location='cpu')
                ref_model = copy.deepcopy(model)
                ref_model.load_state_dict(ref_state_dict, strict=True)
            except:
                ref_model = None
            if ref_model:
                offset_model = convert_to_nn_module_with_offset(model, ref_model)
            model = convert_to_nn_module(model)

            if ref_model:
                quant = ImageQuantizer(offset_model, coord_dataset, dataloader, FLAGS.bitwidth, device, exp_folder)
            else:
                quant = ImageQuantizer(model, coord_dataset, dataloader, FLAGS.bitwidth, device, exp_folder)

            model_quantized, metrics, bytes, state_dict = quant.compress_model(
                retrain=FLAGS.retrain, epochs=FLAGS.retrain_epochs,
                lr=FLAGS.retrain_lr, ref_model=ref_model,
                adaround=FLAGS.adaround,
                adaround_iterations=FLAGS.adaround_iterations,
                adaround_reg=FLAGS.adaround_reg,
                difference_encoding='parallel',
                code=FLAGS.code)
            model.load_state_dict(state_dict, strict=True)
            metrics = check_metrics(dataloader, model, image_resolution)
            print('Final metrics: ', metrics)
            bpp_val = bytes * 8 / (image_resolution[0] * image_resolution[1])
            mse, ssim, psnr = metrics
            metrics_dict = {'psnr': psnr.item(), 'ssim': ssim.item(), 'mse': mse.item(), 'bpp': bpp_val}
            metrics_dict = {**metrics_dict, **TRAINING_FLAGS, **FLAGS.flag_values_dict()}

            name, model_name = get_quant_config_name()
            yaml.dump(metrics_dict, open(os.path.join(exp_folder, image_name, 'metrics_' + name + '.yml'), 'w'))
            torch.save(model.state_dict(),
                       os.path.join(exp_folder, image_name, 'model_' + model_name + '.pth'))


if __name__ == '__main__':
    app.run(main)

"""
Training script to overfit INRs starting from meta-learned initializations. The implementation of MetaSiren is largely
adopted from MetaSDF (Sitzmann 2020)
"""
import copy
import glob
import json
import os
from collections import OrderedDict
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Flags.
flags.DEFINE_string('data_root',
                    'data',
                    'Root directory of data.')
flags.DEFINE_string('exp_root',
                    'exp',
                    'Root directory of experiments.')

# MAML Flags
flags.DEFINE_enum('maml_dataset', 'DIV2K',
                  ['CelebA', 'DIV2K'],
                  'Dataset used for training MAML.')

flags.DEFINE_enum('lr_type', 'per_parameter_per_step',
                  ['static', 'global', 'per_step', 'per_parameter', 'per_parameter_per_step'],
                  'Learning rate type for MAML training')

flags.DEFINE_integer('maml_epochs',
                     1,
                     'Maximum number of dataset epochs for MAML training',
                     lower_bound=1)
flags.DEFINE_integer('maml_batch_size',
                     1,
                     'Meta Batch size used during maml training.',
                     lower_bound=1)
flags.DEFINE_integer('maml_adaptation_steps',
                     3,
                     'Adaptation step during maml training.',
                     lower_bound=1)
flags.DEFINE_float('inner_lr',
                   1e-5,
                   'Learning rate used for the inner loop in during training.',
                   lower_bound=0.0)
flags.DEFINE_float('outer_lr',
                   5e-5,
                   'Learning rate used for the outer loop in MAML training.',
                   lower_bound=0.0)

# Overfitting flags
flags.DEFINE_enum('dataset', 'KODAK',
                  ['KODAK', 'CelebA100', 'KODAKcropped'],
                  'Dataset used during training.')
flags.DEFINE_integer('batch_size',
                     1,
                     'Batch size used during training.',
                     lower_bound=1)
flags.DEFINE_integer('epochs',
                     1000,
                     'Maximum number of epochs.',
                     lower_bound=1)
flags.DEFINE_float('lr',
                   0.0005,
                   'Learning rate used during training.',
                   lower_bound=0.0)
flags.DEFINE_float('l1_reg',
                   0.00001,
                   'L1 weight regularization.',
                   lower_bound=0.0)

flags.DEFINE_enum('activation',
                  'sine',
                  ['sine', 'relu'],
                  'Activation Function.')

flags.DEFINE_enum('encoding', 'nerf', ['mlp', 'nerf', 'positional', 'gauss'], 'Input encoding type used')

flags.DEFINE_integer('hidden_dims',
                     32,
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
                     None,
                     'Number of fourier feature frequencies for input encoding at different scales')

flags.DEFINE_integer('warmup',
                     100,
                     'Warmup epochs to let the Adam momentum converge',
                     lower_bound=0)

flags.DEFINE_integer('patience',
                     500,
                     'patience of lr schedule',
                     lower_bound=1)
flags.DEFINE_float('encoding_scale', 1.4, 'Standard deviation of the encoder')
flags.DEFINE_integer('epochs_til_ckpt', 1000, 'Time interval in seconds until checkpoint is saved.')
flags.DEFINE_integer('steps_til_summary', 1000, 'Time interval in seconds until tensorboard summary is saved.')
FLAGS = flags.FLAGS


def refine(maml, context_dict, steps):
    """Specializes the model"""
    x = context_dict.get('x').cuda()
    y = context_dict.get('y').cuda()

    meta_batch_size = x.shape[0]

    with torch.enable_grad():
        # First, replicate the initialization for each batch item.
        # This is the learned initialization, i.e., in the outer loop,
        # the gradients are backpropagated all the way into the
        # "meta_named_parameters" of the hypo_module.
        fast_params = OrderedDict()
        for name, param in maml.hypo_module.meta_named_parameters():
            fast_params[name] = param[None, ...].repeat((meta_batch_size,) + (1,) * len(param.shape))

        prev_loss = 1e6
        intermed_predictions = []
        for j in range(steps):
            # Using the current set of parameters, perform a forward pass with the context inputs.
            predictions = maml.hypo_module({'coords': x}, params=fast_params)

            # Compute the loss on the context labels.
            loss = maml.loss(predictions, y)
            intermed_predictions.append(predictions['model_out'].detach().cpu())

            if loss > prev_loss:
                print('inner lr too high?')

            fast_params, grads = maml._update_step(loss, fast_params, j)

            prev_loss = loss

    return fast_params, intermed_predictions


def main(_):
    imglob = glob.glob(os.path.join(FLAGS.data_root, FLAGS.dataset, '*'))
    experiment_folder = utils.get_maml_overfitting_experiment_folder(FLAGS)
    maml_folder = utils.get_maml_folder(FLAGS)

    MAML_FLAGS = yaml.load(open(os.path.join(maml_folder, 'FLAGS.yml'), 'r'))
    yaml.dump(FLAGS.flag_values_dict(), open(os.path.join(experiment_folder, 'FLAGS.yml'), 'w'))
    maml_state_dict = torch.load(os.path.join(maml_folder, 'model_maml.pth'),
                                 map_location='cpu')

    torch.save(maml_state_dict,
               os.path.join(experiment_folder, 'model_maml.pth'))

    for i, im in enumerate(imglob):
        image_name = im.split('/')[-1].split('.')[0]
        img = dataio.ImageFile(im)

        # Flip veritically oriented Kodak images to use the same orientation for all images
        if img.img.size[1] > img.img.size[0] and FLAGS.dataset == 'KODAK':
            img.img = img.img.rotate(90, expand=1)

        img_dataset = img
        image_resolution = (img.img.size[1] // FLAGS.downscaling_factor, img.img.size[0] // FLAGS.downscaling_factor)

        coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=image_resolution)
        dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=FLAGS.maml_batch_size, pin_memory=True,
                                num_workers=0)
        model = modules.INRNet(type=FLAGS.activation, mode=FLAGS.encoding, sidelength=image_resolution,
                               out_features=img_dataset.img_channels, hidden_features=FLAGS.hidden_dims,
                               num_hidden_layers=FLAGS.hidden_layers, encoding_scale=FLAGS.encoding_scale,
                               ff_dims=FLAGS.ff_dims)

        root_path = os.path.join(experiment_folder, image_name)
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        if os.path.exists(os.path.join(root_path, 'checkpoints', 'model_final.pth')):
            # result exists already -> skip
            print("Skipping ", root_path)
            continue

        meta_siren = modules.MAML(num_meta_steps=MAML_FLAGS['maml_adaptation_steps'], hypo_module=model,
                                  loss=losses.l2_loss, init_lr=MAML_FLAGS['inner_lr'],
                                  lr_type=MAML_FLAGS['lr_type']).cuda()
        state_dict = torch.load(os.path.join(maml_folder, 'maml_obj.pth'),
                                map_location='cpu')
        meta_siren.load_state_dict(state_dict, strict=True)
        meta_siren.first_order = True
        eval_model = copy.deepcopy(meta_siren.hypo_module)
        num_maml_steps = FLAGS.maml_adaptation_steps

        for i in range(1):
            for step, (model_input, gt) in enumerate(dataloader):
                sample = {'context': {'x': model_input['coords'], 'y': gt['img']},
                          'query': {'x': model_input['coords'], 'y': gt['img']}}
                sample = utils.dict_to_gpu(sample)
                context = sample['context']
                query_x = sample['query']['x'].cuda()

                # Specialize the model with the "generate_params" function.
                fast_params, intermed_predictions = refine(meta_siren, context, num_maml_steps)

                # Compute the final outputs.
                model_output = meta_siren.hypo_module({'coords': query_x}, params=fast_params)['model_out']
                model_output = {'model_out': model_output, 'intermed_predictions': intermed_predictions,
                                'fast_params': fast_params}
                loss_fn = partial(losses.image_mse, None)
                summary_fn = partial(utils.write_image_summary, image_resolution)
                fast_params_squeezed = {name: param.squeeze() for name, param in fast_params.items()}
                eval_model.load_state_dict(fast_params_squeezed)

                print('Metrics after 3 steps: ',
                      utils.compute_metrics(model_output['model_out'], sample['query']['y'],
                                            dataloader.dataset.sidelength))

                l1_loss_fn = partial(losses.model_l1_diff, meta_siren.hypo_module)

                # Resume with the normal overfitting optimization
                training.train(model=eval_model, train_dataloader=dataloader, epochs=FLAGS.epochs, lr=FLAGS.lr,
                               steps_til_summary=FLAGS.steps_til_summary, epochs_til_checkpoint=FLAGS.epochs_til_ckpt,
                               model_dir=root_path, loss_fn=loss_fn, l1_loss_fn=l1_loss_fn, summary_fn=summary_fn,
                               l1_reg=FLAGS.l1_reg, patience=FLAGS.patience, warmup=FLAGS.warmup)

                metrics = utils.check_metrics_full(dataloader, eval_model, image_resolution)
                metrics = {'mse': metrics[0], 'psnr': metrics[2], 'ssim': metrics[1]}

                with open(os.path.join(root_path, 'result.json'), 'w') as fp:
                    json.dump(metrics, fp)


if __name__ == '__main__':
    app.run(main)

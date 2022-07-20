"""Training script for Meta-Learned Initializations."""
import copy
import os

import torch
import yaml
from absl import app
from absl import flags
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import dataio
import losses
import modules
import utils

# Define Flags.
flags.DEFINE_string('data_root',
                    'data',
                    'Root directory of data.')
flags.DEFINE_string('exp_root',
                    'exp',
                    'Root directory of experiments.')

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
                     16,
                     'Number of fourier feature frequencies for input encoding at different scales, if not specified determined through nyquist frequency')
flags.DEFINE_float('encoding_scale', 1.4, 'Scale parameter for the input encoding')
flags.DEFINE_integer('steps_til_summary', 1000, 'Time interval in seconds until tensorboard summary is saved.')
FLAGS = flags.FLAGS


def main(_):
    maml_folder = utils.get_maml_folder(FLAGS)
    if FLAGS.maml_dataset == 'CelebA':
        img_dataset = dataio.CelebA('train', data_root=FLAGS.data_root)
        val_img_dataset = dataio.CelebA('val', data_root=FLAGS.data_root, max_len=100)
        image_resolution = (
            img_dataset.size[1] // FLAGS.downscaling_factor, img_dataset.size[0] // FLAGS.downscaling_factor)
        coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=image_resolution)
        val_coord_dataset = dataio.Implicit2DWrapper(val_img_dataset, sidelength=image_resolution)
        img_channels = 3
    elif FLAGS.maml_dataset == 'DIV2K':
        img_dataset = dataio.DIV2K('train', data_root=FLAGS.data_root)
        val_img_dataset = dataio.DIV2K('val', data_root=FLAGS.data_root, max_len=100)
        image_resolution = (
            img_dataset.size[1] // FLAGS.downscaling_factor, img_dataset.size[0] // FLAGS.downscaling_factor)
        coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=image_resolution)
        val_coord_dataset = dataio.Implicit2DWrapper(val_img_dataset, sidelength=image_resolution)
        img_channels = 3
    else:
        print("Unknown dataset")

    dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=FLAGS.maml_batch_size, pin_memory=True,
                            num_workers=0)
    val_dataloader = DataLoader(val_coord_dataset, shuffle=True, batch_size=FLAGS.maml_batch_size, pin_memory=True,
                                num_workers=0)

    # Define the model.
    model = modules.INRNet(type=FLAGS.activation, mode=FLAGS.encoding, sidelength=image_resolution,
                           out_features=img_channels, hidden_features=FLAGS.hidden_dims,
                           num_hidden_layers=FLAGS.hidden_layers, encoding_scale=FLAGS.encoding_scale,
                           ff_dims=FLAGS.ff_dims)

    model.cuda()

    yaml.dump(FLAGS.flag_values_dict(), open(os.path.join(maml_folder, 'FLAGS.yml'), 'w'))
    meta_siren = modules.MAML(num_meta_steps=FLAGS.maml_adaptation_steps, hypo_module=model, loss=losses.l2_loss,
                              init_lr=FLAGS.inner_lr,
                              lr_type=FLAGS.lr_type).cuda()
    steps_til_summary = FLAGS.steps_til_summary

    optim = torch.optim.Adam(lr=FLAGS.outer_lr, params=meta_siren.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=10,
                                                           threshold=0.0001,
                                                           threshold_mode='rel', cooldown=0, eps=1e-08,
                                                           verbose=True)

    best_val_loss = float("Inf")
    best_state_dict = copy.deepcopy(meta_siren.state_dict())
    steps = FLAGS.maml_adaptation_steps
    step = 0
    for i in range(FLAGS.maml_epochs):
        for model_input, gt in dataloader:
            step += 1
            sample = {'context': {'x': model_input['coords'], 'y': gt['img']},
                      'query': {'x': model_input['coords'], 'y': gt['img']}}
            sample = utils.dict_to_gpu(sample)
            model_output = meta_siren(sample)
            loss = ((model_output['model_out'] - sample['query']['y']) ** 2).mean()
            if not step % steps_til_summary:
                visualized_steps = list(range(steps)[::int(steps / 3)])
                visualized_steps.append(steps - 1)
                print("Step %d, Total loss %0.6f" % (step, loss))
                fig, axes = plt.subplots(1, 5, figsize=(30, 6))
                ax_titles = ['Learned Initialization', 'Inner step {} output'.format(str(visualized_steps[0] + 1)),
                             'Inner step {} output'.format(str(visualized_steps[1] + 1)),
                             'Inner step {} output'.format(str(visualized_steps[2] + 1)),
                             'Ground Truth']
                for i, inner_step_out in enumerate([model_output['intermed_predictions'][i] for i in visualized_steps]):
                    utils.plot_sample_image(inner_step_out, ax=axes[i], image_resolution=image_resolution)
                    axes[i].set_title(ax_titles[i], fontsize=25)
                utils.plot_sample_image(model_output['model_out'], ax=axes[-2], image_resolution=image_resolution)
                axes[-2].set_title(ax_titles[-2], fontsize=25)

                utils.plot_sample_image(sample['query']['y'], ax=axes[-1], image_resolution=image_resolution)
                axes[-1].set_title(ax_titles[-1], fontsize=25)

                plt.show(block=False)
                fast_params = model_output['fast_params']

                print('Metrics after {} steps: '.format(FLAGS.maml_adaptation_steps),
                      utils.compute_metrics(model_output['model_out'], sample['query']['y'],
                                            dataloader.dataset.sidelength))
                torch.save(model.state_dict(),
                           os.path.join(os.path.join(maml_folder, 'model_maml_step{}.pth'.format(step))))
                torch.save(meta_siren.state_dict(),
                           os.path.join(os.path.join(maml_folder, 'maml_obj_step{}.pth'.format(step))))

            optim.zero_grad()
            loss.backward()
            optim.step()

            if not step % 500:

                val_loss_sum = 0
                meta_siren.first_order = True
                with torch.no_grad():  # disable outer loop gradient, inner gradient is manually activated in MAML module
                    for val_step, (model_input, gt) in enumerate(val_dataloader):
                        sample = {'context': {'x': model_input['coords'], 'y': gt['img']},
                                  'query': {'x': model_input['coords'], 'y': gt['img']}}
                        sample = utils.dict_to_gpu(sample)
                        model_output = meta_siren(sample)
                        val_loss = ((model_output['model_out'] - sample['query']['y']) ** 2).mean().detach().cpu()
                        val_loss_sum += val_loss

                    print("validation loss: ", val_loss_sum.item())
                    if val_loss_sum < best_val_loss:
                        best_state_dict = copy.deepcopy(meta_siren.state_dict())
                        best_val_loss = val_loss_sum
                    scheduler.step(val_loss_sum)
                meta_siren.first_order = False
    meta_siren.load_state_dict(best_state_dict)
    model = meta_siren.hypo_module

    torch.save(model.state_dict(),
               os.path.join(os.path.join(maml_folder, 'model_maml.pth')))
    torch.save(meta_siren.state_dict(),
               os.path.join(os.path.join(maml_folder, 'maml_obj.pth')))


if __name__ == '__main__':
    app.run(main)

import os
import warnings
from collections import Mapping

import numpy as np
import skimage.metrics
import torch
from torch.optim.lr_scheduler import EPOCH_DEPRECATION_WARNING, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import dataio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def lin2img(tensor, sidelen=None):
    batch_size, num_samples, channels = tensor.shape
    if not sidelen:
        d = np.sqrt(num_samples).astype(int)
        sidelen = (d, d)
    return tensor.view(batch_size, sidelen[0], sidelen[1], channels).squeeze(-1)


def plot_sample_image(img_batch, ax, image_resolution):
    img = lin2img(img_batch, sidelen=image_resolution)[0].detach().cpu().numpy()
    img += 1
    img /= 2.
    img = np.clip(img, 0., 1.)
    ax.set_axis_off()
    ax.imshow(img)


def dict_to_gpu(ob):
    if isinstance(ob, Mapping):
        return {k: dict_to_gpu(v) for k, v in ob.items()}
    else:
        return ob.cuda()


def get_base_overfitting_experiment_folder(FLAGS):
    """Create string with experiment name and get number of experiment."""

    # create exp folder
    exp_name = '_'.join([
        FLAGS.dataset, str(FLAGS.downscaling_factor) + 'x',
                       'epochs' + str(FLAGS.epochs), 'lr' + str(FLAGS.lr)])

    if FLAGS.ff_dims:
        ff_dims = FLAGS.ff_dims
        exp_name = '_'.join([exp_name, "ffdims", str(ff_dims)])
    exp_name = '_'.join([
        exp_name, 'hdims' + str(FLAGS.hidden_dims),
                  'hlayer' + str(FLAGS.hidden_layers)
    ])
    exp_name = '_'.join([exp_name, str(FLAGS.encoding)])
    exp_name = '_'.join([exp_name, str(FLAGS.activation)])

    if FLAGS.l1_reg > 0.0:
        exp_name = '_'.join([exp_name, 'l1_reg' + str(FLAGS.l1_reg)])

    if FLAGS.encoding_scale > 0.0:
        exp_name = '_'.join([exp_name, 'enc_scale' + str(FLAGS.encoding_scale)])

    exp_folder = os.path.join(FLAGS.exp_root, exp_name)

    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    return exp_folder


def get_maml_overfitting_experiment_folder(FLAGS, subfolder=None):
    """Create string with experiment name and get number of experiment."""

    # create exp folder
    exp_name = '_'.join([
        FLAGS.dataset, str(FLAGS.downscaling_factor) + 'x', 'maml_batch_size' + str(FLAGS.maml_batch_size),
                       'epochs' + str(FLAGS.epochs), 'lr' + str(FLAGS.lr), 'outer_lr' + str(FLAGS.outer_lr),
                       'inner_lr' + str(FLAGS.inner_lr), 'lr_type_' + str(FLAGS.lr_type),
                       'maml_epochs' + str(FLAGS.maml_epochs),
                       'adapt_steps' + str(FLAGS.maml_adaptation_steps)])

    if FLAGS.ff_dims:
        ff_dims = FLAGS.ff_dims
        exp_name = '_'.join([exp_name, "ffdims", str(ff_dims)])
    exp_name = '_'.join([
        exp_name, 'hdims' + str(FLAGS.hidden_dims),
                  'hlayer' + str(FLAGS.hidden_layers)
    ])
    exp_name = '_'.join([exp_name, str(FLAGS.encoding)])
    exp_name = '_'.join([exp_name, str(FLAGS.activation)])

    if FLAGS.l1_reg > 0.0:
        exp_name = '_'.join([exp_name, 'l1_reg' + str(FLAGS.l1_reg)])
    if FLAGS.encoding_scale > 0.0:
        exp_name = '_'.join([exp_name, 'enc_scale' + str(FLAGS.encoding_scale)])
    if subfolder:
        exp_folder = os.path.join(FLAGS.exp_root, subfolder, exp_name)
    else:
        exp_folder = os.path.join(FLAGS.exp_root, exp_name)

    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    return exp_folder


def get_maml_folder(FLAGS):
    """Create string with experiment name and get number of experiment."""

    # create exp folder
    exp_name = '_'.join([
        'MAML', FLAGS.maml_dataset, str(FLAGS.downscaling_factor) + 'x', 'batch_size' + str(FLAGS.maml_batch_size),
                                    'epochs' + str(FLAGS.maml_epochs), 'outer_lr' + str(FLAGS.outer_lr),
                                    'inner_lr' + str(FLAGS.inner_lr), 'lr_type_' + str(FLAGS.lr_type),
                                    'adapt_steps' + str(FLAGS.maml_adaptation_steps)])

    if FLAGS.ff_dims:
        ff_dims = FLAGS.ff_dims
        exp_name = '_'.join([exp_name, "ffdims", str(ff_dims)])
    exp_name = '_'.join([
        exp_name, 'hdims' + str(FLAGS.hidden_dims),
                  'hlayer' + str(FLAGS.hidden_layers)
    ])
    exp_name = '_'.join([exp_name, str(FLAGS.encoding)])
    exp_name = '_'.join([exp_name, str(FLAGS.activation)])
    if FLAGS.encoding_scale > 0.0:
        exp_name = '_'.join([exp_name, 'enc_scale' + str(FLAGS.encoding_scale)])

    exp_folder = os.path.join(FLAGS.exp_root, 'maml', exp_name)

    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    return exp_folder


def convert_to_nn_module(net):
    out_net = torch.nn.Sequential()
    for name, module in net.named_children():
        if module.__class__.__name__ == 'BatchLinear':
            linear_module = torch.nn.Linear(
                module.in_features,
                module.out_features,
                bias=True if module.bias is not None else False)
            linear_module.weight.data = module.weight.data.clone()
            linear_module.bias.data = module.bias.data.clone()
            out_net.add_module(name, linear_module)
        elif module.__class__.__name__ == 'Sine':
            out_net.add_module(name, module)

        elif module.__class__.__name__ == 'MetaSequential':
            new_module = convert_to_nn_module(module)
            out_net.add_module(name, new_module)
        else:
            if len(list(module.named_children())):
                out_net.add_module(name, convert_to_nn_module(module))
            else:
                out_net.add_module(name, module)
    return out_net


def check_metrics_full(test_loader: DataLoader, model: torch.nn.Module, image_resolution):
    model.eval()
    with torch.no_grad():
        for step, (model_input, gt) in enumerate(test_loader):
            model_input = {key: value.cuda() for key, value in model_input.items()}
            gt = {key: value.cuda() for key, value in gt.items()}

            predictions = model(model_input)
            gt_img = dataio.lin2img(gt['img'], image_resolution)
            pred_img = dataio.lin2img(predictions['model_out'], image_resolution)
            pred_img = pred_img.detach().cpu().numpy()[0]
            gt_img = gt_img.detach().cpu().numpy()[0]
            p = pred_img.transpose(1, 2, 0)
            trgt = gt_img.transpose(1, 2, 0)
            p = (p / 2.) + 0.5
            p = np.clip(p, a_min=0., a_max=1.)

            trgt = (trgt / 2.) + 0.5
            mse = skimage.metrics.mean_squared_error(p, trgt)
            ssim = skimage.metrics.structural_similarity(p, trgt, multichannel=True, data_range=1)
            psnr = skimage.metrics.peak_signal_noise_ratio(p, trgt, data_range=1)
    return mse, ssim, psnr


def check_metrics(test_loader: DataLoader, model: torch.nn.Module, image_resolution, params=None):
    model.eval()
    with torch.no_grad():
        for step, (model_input, gt) in enumerate(test_loader):
            model_input = {key: value.cuda() for key, value in model_input.items()}
            gt = {key: value.cuda() for key, value in gt.items()}
            if params:
                predictions = model(model_input['coords'], params=params)
            else:
                predictions = model(model_input['coords'])
            gt_img = dataio.lin2img(gt['img'], image_resolution)
            pred_img = dataio.lin2img(predictions, image_resolution)
            pred_img = pred_img.detach().cpu().numpy()[0]
            gt_img = gt_img.detach().cpu().numpy()[0]
            p = pred_img.transpose(1, 2, 0)
            trgt = gt_img.transpose(1, 2, 0)
            p = (p / 2.) + 0.5
            p = np.clip(p, a_min=0., a_max=1.)

            trgt = (trgt / 2.) + 0.5
            mse = skimage.metrics.mean_squared_error(p, trgt)
            ssim = skimage.metrics.structural_similarity(p, trgt, multichannel=True, data_range=1)
            psnr = skimage.metrics.peak_signal_noise_ratio(p, trgt, data_range=1)

    return mse, ssim, psnr


def compute_metrics(model_out, gt, image_resolution):
    with torch.no_grad():
        gt_img = dataio.lin2img(gt, image_resolution)
        pred_img = dataio.lin2img(model_out, image_resolution)
        pred_img = pred_img.detach().cpu().numpy()[0]
        gt_img = gt_img.detach().cpu().numpy()[0]
        p = pred_img.transpose(1, 2, 0)
        trgt = gt_img.transpose(1, 2, 0)
        p = (p / 2.) + 0.5
        p = np.clip(p, a_min=0., a_max=1.)

        trgt = (trgt / 2.) + 0.5
        mse = skimage.metrics.mean_squared_error(p, trgt)
        ssim = skimage.metrics.structural_similarity(p, trgt, multichannel=True, data_range=1)
        psnr = skimage.metrics.peak_signal_noise_ratio(p, trgt, data_range=1)
    return mse, ssim, psnr


class ReduceLROnPlateauWithWarmup(ReduceLROnPlateau):

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, warmup_end_lr=0, warmup_steps=0, verbose=False):

        super().__init__(optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose, threshold=threshold,
                         threshold_mode=threshold_mode, cooldown=cooldown, min_lr=min_lr, eps=eps)
        self.warmup_end_lr = warmup_end_lr
        self.warmup_steps = warmup_steps
        self._set_warmup_lr(1)

    def _set_warmup_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):

            new_lr = epoch * (self.warmup_end_lr / self.warmup_steps)
            param_group['lr'] = new_lr
            if self.verbose:
                print('Epoch {:5d}: increase learning rate'
                      ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    # Override step method to include warmup
    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch

        if self.warmup_steps > 0 and epoch <= self.warmup_steps:
            self._set_warmup_lr(epoch)

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_image_summary(image_resolution, model, model_input, gt,
                        model_output, writer, total_steps, prefix='train_'):
    gt_img = dataio.lin2img(gt['img'], image_resolution)
    pred_img = dataio.lin2img(model_output['model_out'], image_resolution)
    output_vs_gt = torch.cat((gt_img, pred_img), dim=-1)
    writer.add_image(prefix + 'gt_vs_pred', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)

    pred_img = dataio.rescale_img((pred_img + 1) / 2, mode='clamp').permute(0, 2, 3, 1).squeeze(
        0).detach().cpu().numpy()

    gt_img = dataio.rescale_img((gt_img + 1) / 2, mode='clamp').permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()

    writer.add_image(prefix + 'pred_img', torch.from_numpy(pred_img).permute(2, 0, 1), global_step=total_steps)
    writer.add_image(prefix + 'gt_img', torch.from_numpy(gt_img).permute(2, 0, 1), global_step=total_steps)

    write_psnr(dataio.lin2img(model_output['model_out'], image_resolution),
               dataio.lin2img(gt['img'], image_resolution), writer, total_steps, prefix + 'img_')


def write_psnr(pred_img, gt_img, writer, iter, prefix):
    batch_size = pred_img.shape[0]

    pred_img = pred_img.detach().cpu().numpy()
    gt_img = gt_img.detach().cpu().numpy()

    psnrs, ssims = list(), list()
    for i in range(batch_size):
        p = pred_img[i].transpose(1, 2, 0)
        trgt = gt_img[i].transpose(1, 2, 0)

        p = (p / 2.) + 0.5
        p = np.clip(p, a_min=0., a_max=1.)

        trgt = (trgt / 2.) + 0.5

        ssim = skimage.metrics.structural_similarity(p, trgt, multichannel=True, data_range=1)
        psnr = skimage.metrics.peak_signal_noise_ratio(p, trgt, data_range=1)

        psnrs.append(psnr)
        ssims.append(ssim)

    writer.add_scalar(prefix + "psnr", np.mean(psnrs), iter)
    writer.add_scalar(prefix + "ssim", np.mean(ssims), iter)

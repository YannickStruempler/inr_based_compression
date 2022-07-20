import warnings
import os
import torch
from torch.optim.lr_scheduler import EPOCH_DEPRECATION_WARNING, ReduceLROnPlateau
from torch.utils.data import DataLoader

import losses
from lib.nglod.lib.renderer import Renderer
from lib.nglod.lib.validator import GeometricValidator, PointCloudValidator
from lib.nglod.lib.options import parse_options
from lib.nglod.lib.tracer import SphereTracer
import dataio
import matplotlib.pyplot as plt
import numpy as np
import skimage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_base_overfitting_experiment_folder(FLAGS):
    """Create string with experiment name and get number of experiment."""

    # create exp folder
    exp_name = '_'.join([
        FLAGS.dataset, 'epochs' + str(FLAGS.epochs), 'lr' + str(FLAGS.lr)])

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
        FLAGS.dataset, str(FLAGS.downscaling_factor) + 'x',  'maml_batch_size' + str(FLAGS.maml_batch_size),
        'epochs' + str(FLAGS.epochs), 'lr' + str(FLAGS.lr), 'outer_lr' + str(FLAGS.outer_lr),
        'inner_lr' + str(FLAGS.inner_lr), 'lr_type_' + str(FLAGS.lr_type), 'maml_epochs' + str(FLAGS.maml_epochs),
                                    'adapt_steps' + str(FLAGS.maml_adaptation_steps)])

    if FLAGS.ff_dims:
        ff_dims = FLAGS.ff_dims
        exp_name = '_'.join([exp_name, "ffdims",  str(ff_dims)])
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
        'MAML', FLAGS.maml_dataset,str(FLAGS.downscaling_factor) + 'x', 'batch_size' + str(FLAGS.maml_batch_size),
                                    'epochs' + str(FLAGS.maml_epochs), 'outer_lr' + str(FLAGS.outer_lr),
                                    'inner_lr' + str(FLAGS.inner_lr),'lr_type_' + str(FLAGS.lr_type),
                                    'adapt_steps' + str(FLAGS.maml_adaptation_steps)])

    if FLAGS.ff_dims:
        ff_dims =  FLAGS.ff_dims
        exp_name = '_'.join([exp_name, "ffdims",  str(ff_dims)])
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
            else: out_net.add_module(name, module)
    return out_net

def check_metrics_full(test_loader: DataLoader, model: torch.nn.Module, image_resolution, show=False):
    model.eval()
    with torch.no_grad():
        for step, (model_input, gt) in enumerate(test_loader):
            model_input = {key: value.cuda() for key, value in model_input.items()}
            gt = {key: value.cuda() for key, value in gt.items()}

            predictions = model(model_input)
            gt_img = dataio.lin2img(gt['img'], image_resolution)
            pred_img = dataio.lin2img(predictions['model_out'], image_resolution)


            #    pred_grad = dataio.grads2img(dataio.lin2img(img_gradient)).permute(1,2,0).squeeze().detach().cpu().numpy()
            #  pred_lapl = cv2.cvtColor(cv2.applyColorMap(dataio.to_uint8(dataio.rescale_img(
            #                           dataio.lin2img(img_laplace), perc=2).permute(0,2,3,1).squeeze(0).detach().cpu().numpy()), cmapy.cmap('RdBu')), cv2.COLOR_BGR2RGB)


            pred_img = pred_img.detach().cpu().numpy()[0]
            gt_img = gt_img.detach().cpu().numpy()[0]
            p = pred_img.transpose(1, 2, 0)
            trgt = gt_img.transpose(1, 2, 0)
            p = (p / 2.) + 0.5
            p = np.clip(p, a_min=0., a_max=1.)

            trgt = (trgt / 2.) + 0.5
            mse  = skimage.metrics.mean_squared_error(p, trgt)
            ssim = skimage.metrics.structural_similarity(p, trgt, multichannel=True, data_range=1)
            psnr = skimage.metrics.peak_signal_noise_ratio(p, trgt, data_range=1)

            # mse = skimage.measure.compare_mse(p, trgt)
            # ssim = skimage.measure.compare_ssim(p, trgt, multichannel=True, data_range=1)
            # psnr = skimage.measure.compare_psnr(p, trgt, data_range=1)
        if show:
            import matplotlib.pyplot as plt
            plt.close()
            plt.imshow(p)
            plt.show()

        # from siren.utils import write_psnr
        # write_psnr(b, a, None, None, None)
    return mse, ssim, psnr

def check_metrics(test_loader: DataLoader, model: torch.nn.Module, image_resolution, params=None, show=False):
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
            if show:
                import matplotlib.pyplot as plt
                plt.imshow(p)
                plt.show()
            # mse = skimage.measure.compare_mse(p, trgt)
            # ssim = skimage.measure.compare_ssim(p, trgt, multichannel=True, data_range=1)
            # psnr = skimage.measure.compare_psnr(p, trgt, data_range=1)

    return mse, ssim, psnr

def check_and_save(test_loader: DataLoader, model: torch.nn.Module, image_resolution, params=None, show=False, path=None):
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

            plt.imsave(path, p)
            if show:
                plt.imshow(p)
                plt.show()
            # mse = skimage.measure.compare_mse(p, trgt)
            # ssim = skimage.measure.compare_ssim(p, trgt, multichannel=True, data_range=1)
            # psnr = skimage.measure.compare_psnr(p, trgt, data_range=1)

    return mse, ssim, psnr

def compute_metrics(model_out, gt,  image_resolution):
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
        # mse = skimage.measure.compare_mse(p, trgt)
        # ssim = skimage.measure.compare_ssim(p, trgt, multichannel=True, data_range=1)
        # psnr = skimage.measure.compare_psnr(p, trgt, data_range=1)
        mse = skimage.metrics.mean_squared_error(p, trgt)
        ssim = skimage.metrics.structural_similarity(p, trgt, multichannel=True, data_range=1)
        psnr = skimage.metrics.peak_signal_noise_ratio(p, trgt, data_range=1)
    return mse, ssim, psnr

class ReduceLROnPlateauWithWarmup(ReduceLROnPlateau):

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, warmup_end_lr=0, warmup_steps=0, verbose=False):

        super().__init__(optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose, threshold=threshold, threshold_mode=threshold_mode, cooldown=cooldown, min_lr=min_lr, eps=eps)
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

    #Override step method to include warmup
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

def write_sdf_summary(dataset_path, model, model_input, gt, model_output, writer, epoch, verbose=False):
    if (epoch > 0):
        parser = parse_options(return_parser=True)
        defaults_args = parser.parse_args([])
        renderer = Renderer(SphereTracer(args=defaults_args), defaults_args)
        out = renderer.shade_images(convert_to_nn_module(model).eval(), f=defaults_args.camera_origin,
                                    t=defaults_args.camera_lookat).image().byte().numpy()
        writer.add_image(f'Depth', out.depth.transpose(2, 0, 1), epoch)
        writer.add_image(f'Hit', out.hit.transpose(2, 0, 1), epoch)
        writer.add_image(f'Normal', out.normal.transpose(2, 0, 1), epoch)
        writer.add_image(f'RGB', out.rgb.transpose(2, 0, 1), epoch)


        #compute iou
        vol_val = GeometricValidator(dataset_path, device, model)
        voliou = vol_val.validate(epoch)
        writer.add_scalar('vol_iou',voliou, epoch)


        #chamfer distance
        point_val = PointCloudValidator(dataset_path, device, convert_to_nn_module(model))
        chamfer_dist = point_val.validate(epoch)
        writer.add_scalar('chamfer_dist', chamfer_dist, epoch)

        if verbose:
            import matplotlib.pyplot as plt
            plt.imshow(out.normal)
            plt.show()
            print("Volumetric IOU", voliou)
            print("Chamfer Distance", chamfer_dist)
def render_sdf(model, verbose=False):

            parser = parse_options(return_parser=True)
            defaults_args = parser.parse_args([])
            renderer = Renderer(SphereTracer(args=defaults_args), defaults_args)
            theta = 0
            rotation_matrix_x = torch.tensor(
                [[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]], dtype=torch.float32)
            theta = -0
            rotation_matrix_y = torch.tensor(
                [[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]], dtype=torch.float32)
            theta = 3.14
            rotation_matrix_z = torch.tensor(
                [[1, 0, 0], [0, np.cos(theta), np.sin(theta)], [0, -np.sin(theta), np.cos(theta)]], dtype=torch.float32)
            rotation_matrix = rotation_matrix_x @ rotation_matrix_y @ rotation_matrix_z
            out = renderer.shade_images(model, [1.8, 1.5, 1.8], fov=45, #,
                                        t=[0, -0.0, 0],
                                        mm=rotation_matrix).image().byte().numpy()

            # out = renderer.shade_images(f=[-2.5, -5, -2.5],
            #                             t=[0, 0, 0], fv=45).image().byte().numpy()
            # normal = out.normal.transpose(2, 0, 1)
            # rgb = out.rgb.transpose(2, 0, 1)

            if verbose:
                import matplotlib.pyplot as plt
                plt.imshow(out.normal)
                plt.show()
            return out.normal, out.rgb


def check_metrics_sdf(dataset_path, model, convert=True):
    if convert:
        nn_module_model = convert_to_nn_module(model)
    else:
        nn_module_model = model
    # compute iou
    vol_val = GeometricValidator(dataset_path, device, nn_module_model)
    voliou = vol_val.validate(0, is_meta_module=False)

    # chamfer distance
    point_val = PointCloudValidator(dataset_path, device, nn_module_model)
    chamfer_dist = point_val.validate(0)
    return voliou, chamfer_dist

def check_mse_sdf(meshdataset, model):
    model.eval()
    with torch.no_grad():
        model_input = meshdataset.pts.cuda()
        gt = {'dist': meshdataset.d.cuda()}
        predictions = model(model_input)
        model_ouput = {'model_out': predictions}
        mse = losses.sdf_mse(model_ouput, gt)['sdf_loss'].item()

    return mse
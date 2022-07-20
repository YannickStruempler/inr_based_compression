import copy
import io
import os
from functools import partial

import numpy as np
import torch
from brotli import brotli
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from lib.adaround import adaround_utils
from lib.adaround.adaround_weight import AdaroundParameters, Adaround
from lib.aimet_torch.qc_quantize_op import QcPostTrainingWrapper
from lib.aimet_torch.quantsim import QuantizationSimModel
from lib.aimet_torch.save_utils import SaveUtils
from lib.arithmetic_coding.arithmeticcompress import AE
from losses import sdf_mse
from modules import Sine, ImageDownsampling, PosEncodingNeRF, FourierFeatureEncodingPositional, \
    FourierFeatureEncodingGaussian, RefLinear
from quantize_utils import AimetDatasetSDF


class Quantizer():
    """Base Quantizer class  interfacing to AIMET as the quantization backbone"""

    def __init__(self, model, dataset, dataloader, bitwidth, device, experiment_path):
        self.model = model
        self.dataset = dataset
        self.dataloader = dataloader
        self.bitwidth = bitwidth
        self.device = device
        self.sim = None
        self.experiment_path = experiment_path

    def _quantize_model(self):
        '''Initialize the quantization simulation'''
        self.sim = self._get_quant_sim()
        self.sim.model(self.dummy_in)

    def evaluate(self, model: torch.nn.Module, iterations: int = None, use_cuda: bool = True):
        """Dummy function required by the AIMET API"""
        return None

    def _get_quant_sim(self):
        '''Build the quantization simulation only applying quantization to parameters not input & outputs'''
        self.dummy_in = ((torch.rand((2, self.input_shape[-1])).unsqueeze(0) * 2) - 1).to(self.device)
        sim = QuantizationSimModel(self.model, default_param_bw=self.bitwidth,
                                   default_output_bw=31, dummy_input=self.dummy_in, in_place=False)

        self._exclude_layers(sim)
        i = 0
        for name, mod in sim.model.named_modules():
            if isinstance(mod, QcPostTrainingWrapper):
                # Disable input and output Quantization, we only want to quantize the weights
                mod.output_quantizer.enabled = False
                mod.input_quantizer.enabled = False
                weight_quantizer = mod.param_quantizers['weight']
                bias_quantizer = mod.param_quantizers['bias']
                # we use symmetric quantization encodings to align the weight distribution of multiple layers
                weight_quantizer.use_symmetric_encodings = True
                bias_quantizer.use_symmetric_encodings = True
                if torch.count_nonzero(mod._module_to_wrap.bias.data):
                    mod.param_quantizers['bias'].enabled = True

        return sim

    def _exclude_layers(self, sim):
        '''exclude layers from quantization'''
        excl_layers = []
        for mod in sim.model.modules():
            if isinstance(mod, QcPostTrainingWrapper) and isinstance(mod._module_to_wrap, self.modules_to_exclude):
                excl_layers.append(mod)
            # we do not want our offset layers to be quantized, hence we exclude them here
            if isinstance(mod, RefLinear):
                excl_layers.append(mod.linear_offset)

        sim.exclude_layers_from_quantization(excl_layers)
        return sim

    def _apply_quantization(self, code):
        '''extract quantized weights and compress with entropy coding'''
        quantized_dict = {}
        state_dict = {}
        for name, module in self.sim.model.named_modules():
            if isinstance(module, QcPostTrainingWrapper) and isinstance(module._module_to_wrap, torch.nn.Linear):
                weight_quantizer = module.param_quantizers['weight']
                bias_quantizer = module.param_quantizers['bias']
                weight_quantizer.enabled = True
                bias_quantizer.enabled = True
                wrapped_linear = module._module_to_wrap
                weight = wrapped_linear.weight
                bias = wrapped_linear.bias

                # We quantize and dequantize to get the floating point weights with quantization noise
                state_dict[name + '.weight'] = weight_quantizer.quantize_dequantize(weight,
                                                                                    weight_quantizer.round_mode).cpu().detach()
                # Assert that the quantization worked correctly, there can only be 2 ** bitwidth different weights
                assert (len(torch.unique(state_dict[name + '.weight'])) <= 2 ** weight_quantizer.bitwidth)
                state_dict[name + '.bias'] = bias_quantizer.quantize_dequantize(bias,
                                                                                bias_quantizer.round_mode).cpu().detach()
                assert (len(torch.unique(state_dict[name + '.bias'])) <= 2 ** bias_quantizer.bitwidth)
                quantized_weight = weight_quantizer.quantize(weight,
                                                             weight_quantizer.round_mode).cpu().detach().numpy()

                # Get the integer quantized weights and store them in the quantized_dict dictionary
                quantized_bias = bias_quantizer.quantize(bias,
                                                         bias_quantizer.round_mode).cpu().detach().numpy()
                quantized_dict[name] = {'weight': {'data': quantized_weight, 'encoding': weight_quantizer.encoding},
                                        'bias': {'data': quantized_bias, 'encoding': bias_quantizer.encoding}}
        # Based on the bitwidth used, select the smallest possible uint type and create flattened weight vector
        weights_np = []
        step_sizes = []
        for l in quantized_dict.values():
            w = l['weight']['data']
            b = l['bias']['data']
            Q = l['weight']['encoding'].bw
            if Q < 9:
                tpe = 'uint8'
            elif Q < 17:
                tpe = 'uint16'
            else:
                tpe = 'uint32'
            w = w.astype(tpe).flatten()
            weights_np.append(w)
            # to be able to reconstruct the floating point weights, we need to store the step sizes as well
            step_sizes.append(l['weight']['encoding'].delta)
            if l['bias']['encoding']:
                Q = l['bias']['encoding'].bw
                if Q < 9:
                    tpe = 'uint8'
                elif Q < 17:
                    tpe = 'uint16'
                else:
                    tpe = 'uint32'
                b = b.astype(tpe).flatten()
                weights_np.append(b)
                step_sizes.append(l['bias']['encoding'].delta)
        weights_np = np.concatenate(weights_np)
        step_sizes = np.array(step_sizes)
        # compress with brotli
        if code == 'brotli':
            comp = brotli.compress(weights_np.tobytes())
            step_size_comp = brotli.compress(step_sizes.tobytes())
            bytes = len(comp) + len(step_size_comp)
        # compress with arithmetic coding
        elif code == 'arithmetic':
            out = io.BytesIO()
            ae = AE()
            bytes, frequencies = ae.compress_bytes(io.BytesIO(weights_np.tobytes()), out, bits=self.bitwidth,
                                                   write_freq=True)
            step_size_comp = brotli.compress(step_sizes.tobytes())
            bytes += len(step_size_comp)
        SaveUtils.remove_quantization_wrappers(self.sim.model)
        return bytes, state_dict

    def _apply_adaround(self, adaround_reg=0.01, adaround_iterations=500, offset=True):
        '''apply AdaRound to the quantized model weights'''
        params = AdaroundParameters(data_loader=self.aimet_dataloader, num_batches=len(self.aimet_dataloader),
                                    default_num_iterations=adaround_iterations,
                                    default_reg_param=adaround_reg, default_beta_range=(20, 2))

        # Compute only param encodings
        Adaround._compute_param_encodings(self.sim)

        # Get the module - activation function pair using ConnectedGraph
        module_act_func_pair = adaround_utils.get_module_act_func_pair(self.model, self.dummy_in, offset=offset)

        Adaround._adaround_model(self.model, self.sim, module_act_func_pair, params, self.dummy_in, offset=offset,
                                 working_dir=os.path.join(self.experiment_path, 'bitwidth' + str(self.bitwidth),
                                                          'tmp/adaround/'))

        # Update every module (AdaroundSupportedModules) weight with Adarounded weight (Soft rounding)
        Adaround._update_modules_with_adarounded_weights(self.sim)

        filename_prefix = '../lib/adaround'
        # Export quantization encodings to JSON-formatted file
        Adaround._export_encodings_to_json(self.experiment_path, 'bitwidth' + str(self.bitwidth) + filename_prefix,
                                           self.sim)
        SaveUtils.remove_quantization_wrappers(self.sim.model)
        adarounded_model = self.sim.model
        self.model = adarounded_model
        sim = self._get_quant_sim()
        sim.set_and_freeze_param_encodings(encoding_path=os.path.join(self.experiment_path, 'bitwidth' + str(
            self.bitwidth) + filename_prefix + '.encodings'))
        return sim

    def compress_model(self, retrain=True, epochs=300, ref_model=None,
                       adaround=False, lr=0.0000001, adaround_reg=0.01, adaround_iterations=500,
                       difference_encoding='same', code='brotli'):
        '''Run a model through the compression pipeline'''
        self.sim = self._get_quant_sim()

        if ref_model is not None and difference_encoding == 'parallel':
            print("Compressing model with meta-learned initialization")
            res = self.check_metrics()
            print('After Quantization: ', res)
            if adaround:
                self.sim = self._apply_adaround(adaround_reg=adaround_reg, adaround_iterations=adaround_iterations)
                res = self.check_metrics()
                print('After Adaround: ', res)
            if retrain:
                loss_fn = partial(losses.image_mse, None)
                retrained_model = self.retrain(epochs=epochs,
                                               lr=lr)
                self.sim.compute_encodings(forward_pass_callback=self.evaluate, forward_pass_callback_args=1)
                res = self.check_metrics()
                self.model = retrained_model

                print('After retraining: ', res)
            bytes, state_dict = self._apply_quantization(code)
            new_dict = {}
            for name in state_dict.keys():
                new_name = name.replace('linear.', '')
                new_dict[new_name] = state_dict[name] + ref_model.state_dict()[new_name].cpu()
            state_dict = new_dict


        else:
            print("Compressing model with standard initialization")
            res = self.check_metrics()
            print('After Quantization: ', res)
            if adaround:
                self.sim = self._apply_adaround(adaround_reg=adaround_reg, adaround_iterations=adaround_iterations,
                                                offset=False)
                res = self.check_metrics()
                print('After Adaround: ', res)
            if retrain:
                retrained_model = self.retrain(epochs=epochs,
                                               lr=lr)

                res = self.check_metrics()
                self.sim.compute_encodings(forward_pass_callback=self.evaluate, forward_pass_callback_args=1)
                print('After retraining: ', res)
            bytes, state_dict = self._apply_quantization(code)
        if "positional_encoding.B" in self.model.state_dict():
            state_dict["positional_encoding.B"] = self.model.state_dict()["positional_encoding.B"]
        return self.model, res, bytes, state_dict


class SDFQuantizer(Quantizer):
    '''Quantizer class for 3D shapes represented as SDFs'''

    def __init__(self, model, dataset, dataloader, bitwidth, device, exp_path):
        super().__init__(model, dataset, dataloader, bitwidth, device, exp_path)
        self.input_shape = (3,)

        self.dataloader = dataloader
        self.modules_to_exclude = (
            Sine, ImageDownsampling, PosEncodingNeRF, FourierFeatureEncodingPositional, FourierFeatureEncodingGaussian,
            torch.nn.ReLU)
        self.aimet_dataloader = DataLoader(AimetDatasetSDF(dataset), shuffle=False, batch_size=len(dataset),
                                           pin_memory=True,
                                           num_workers=8)

    def evaluate(self, model: torch.nn.Module, iterations: int = None, use_cuda: bool = True):
        return self.check_metrics()

    def check_metrics(self):
        return utils.check_mse_sdf(self.dataset, self.sim.model)

    def retrain(self, epochs, lr):
        optim = torch.optim.Adam(lr=lr, params=self.sim.model.parameters())
        best_score = self.check_metrics()
        best_state_dict = copy.deepcopy(self.sim.model.state_dict())
        use_amp = True
        q_wrapper_list = []
        for name, mod in self.sim.model.named_modules():
            if isinstance(mod, QcPostTrainingWrapper):
                q_wrapper_list.append(mod)
        loss_fn = sdf_mse
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        with tqdm(total=len(self.dataloader) * epochs) as pbar:
            for epoch in range(epochs):
                self.sim.model.train()
                for step, (model_input, gt) in enumerate(self.dataloader):
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        model_input = {key: value.cuda() for key, value in model_input.items()}
                        gt = {key: value.cuda() for key, value in gt.items()}
                        model_output = {}
                        model_output['model_out'] = self.sim.model(model_input['coords'])
                        losses = loss_fn(model_output, gt)
                        train_loss = 0.
                        for loss_name, loss in losses.items():
                            single_loss = loss.mean()
                            train_loss += single_loss
                        optim.zero_grad()
                        scaler.scale(train_loss).backward()
                        scaler.step(optim)
                        scaler.update()
                        pbar.update(1)

                with torch.no_grad():
                    score = self.check_metrics()
                # keep highest scoring model
                if score < best_score:
                    best_state_dict = copy.deepcopy(self.sim.model.state_dict())
                    best_score = score
                    print("best score", best_score)

                # if model is diverging, try to recover from previous best state dict
                if score > 10 * best_score:
                    print("recover")
                    self.sim.model.load_state_dict(best_state_dict, strict=True)
        # load highest scoring model
        self.sim.model.load_state_dict(best_state_dict, strict=True)
        return self.sim.model

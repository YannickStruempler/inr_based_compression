
from typing import Tuple, Union, List, Dict
import torch

# Import AIMET specific modules
# from aimet_common.utils import AimetLogger
# from aimet_torch.meta.connectedgraph import ConnectedGraph
# from aimet_torch.utils import create_rand_tensors_given_shapes, get_device

from image_compression.modules import RefLinear, Sine

ActivationTypes = (torch.nn.ReLU6, torch.nn.ReLU, torch.nn.PReLU, torch.nn.RReLU, torch.nn.LeakyReLU,
                       torch.nn.Sigmoid, torch.nn.LogSigmoid, torch.nn.Softmin, torch.nn.Softmax, torch.nn.LogSoftmax,
                       torch.nn.Tanh, torch.nn.Hardtanh, Sine)

def get_module_act_func_pair(model: torch.nn.Module, model_input: Union[Tuple[torch.Tensor], List[torch.Tensor]], offset: bool = False) -> \
        Dict[torch.nn.Module, Union[torch.nn.Module, None]]:
    """
    For given model, returns dictionary of module to immediate following activation function else maps
    module to None.

    Activation functions should be defined as nn.Modules in model and not as functional in the forward pass.

    :param model: Pytorch model
    :param model_input:  Model input, Can be a list/tuple of input tensor(s)
    :return: Dictionary of module to activation function
    """
    module_act_func_pair = {}
    for module in model.modules():
        if isinstance(module, torch.nn.Sequential):
            if module[0].__class__.__name__ == 'RefLinear' and offset:
                if len(module) > 1 and isinstance(module[1], ActivationTypes):
                    module_act_func_pair[module[0].linear] = module[1]
                else:
                    module_act_func_pair[module[0].linear] = None
            if isinstance(module[0], torch.nn.Linear) and not offset:
                if len(module) > 1 and isinstance(module[1], ActivationTypes):
                    module_act_func_pair[module[0]] = module[1]
                else:
                    module_act_func_pair[module[0]] = None
    return module_act_func_pair

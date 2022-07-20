import torch

from modules import RefLinear


def convert_to_nn_module(net):
    """
    Converts a the input module from the torchmeta MetaModule classes to torch.nn.Module.
    This is necessary because AIMET expects a torch.nn.Module for quantization.
    """
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


def convert_to_nn_module_with_offset(net, ref_net):
    """
    Converts a the input module from the torchmeta MetaModule classes to torch.nn.Module.
    This is necessary because AIMET expects a torch.nn.Module for quantization.
    This function also replaces the linear layers by RefLinear layers, which use the weight from net
    and the weight offset form ref_net.
    """
    out_net = torch.nn.Sequential()
    for (name, module), (ref_name, ref_module) in zip(net.named_children(), ref_net.named_children()):
        if module.__class__.__name__ == 'BatchLinear':
            # Create a RefLinear layer that contains the weight from the module and the offset from the ref_module
            linear_module = RefLinear(module.in_features, module.out_features, weight=module.weight.data.clone(),
                                      weight_offset=ref_module.weight.data.clone(), bias=module.bias.data.clone(),
                                      bias_offset=ref_module.bias.data.clone())
            out_net.add_module(name, linear_module)
        elif module.__class__.__name__ == 'Sine':
            out_net.add_module(name, module)

        elif module.__class__.__name__ == 'MetaSequential':
            new_module = convert_to_nn_module_with_offset(module, ref_module)
            out_net.add_module(name, new_module)
        else:
            if len(list(module.named_children())):
                out_net.add_module(name, convert_to_nn_module_with_offset(module, ref_module))
            else:
                out_net.add_module(name, module)
    return out_net


class AimetDataset(torch.utils.data.Dataset):
    """Dataset wrapper for AIMET"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (self.dataset[idx][0]['coords'].unsqueeze(0), self.dataset[idx][1]['img'])

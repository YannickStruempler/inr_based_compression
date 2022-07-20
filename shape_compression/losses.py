'''Loss functions.'''


def sdf_mse(model_output, gt):
    "Computes the MSE between the SDF output (model_output) and the ground truth distances (gt['dist'])"
    return {'sdf_loss': ((model_output['model_out'] - gt['dist']) ** 2).mean()}


def l2_loss(prediction, gt):
    "Computes the L2 loss between the prediction and gt"
    return ((prediction['model_out'] - gt) ** 2).mean()


def model_l1(model, l1_lambda):
    "Computes the L1 norm of the models parameters and weights it with l1_lambda"
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return {'l1_loss': l1_lambda * l1_norm}


def model_l2(model, l2_lambda):
    "Computes the L2 norm of the models parameters and weights it with l2_lambda"
    l2_norm = sum((p ** 2).sum() for p in model.parameters())
    return {'l2_loss': l2_lambda * l2_norm}


def model_l1_diff(ref_model, model, l1_lambda):
    "Computes the L1 norm of the parameter difference between a models and a ref_model and weights it with l1_lambda"
    l1_norm = sum((p - ref_p).abs().sum() for (p, ref_p) in zip(model.parameters(), ref_model.parameters()))
    return {'l1_loss': l1_lambda * l1_norm}


def model_l1_dictdiff(ref_model_dict, model_dict, l1_lambda):
    "Computes the L1 norm of the parameter difference between two model state dicts weights it with l1_lambda"
    l1_norm = sum(
        (p.squeeze() - ref_p.squeeze()).abs().sum() for (p, ref_p) in zip(ref_model_dict.values(), model_dict.values()))
    return {'l1_loss': l1_lambda * l1_norm}


def model_l2_diff(ref_model, model, l1_lambda):
    "Computes the L1 norm of the parameter difference between a models and a ref_model and weights it with l1_lambda"
    l1_norm = sum(((p - ref_p) ** 2).sum() for (p, ref_p) in zip(model.parameters(), ref_model.parameters()))
    return {'l1_loss': l1_lambda * l1_norm}

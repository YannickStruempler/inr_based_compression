from shape_compression import dataio
from torch.utils.data import DataLoader
from ..options import parse_options
from ..tracer import SphereTracer
from pytorch3d.loss.chamfer import chamfer_distance


class PointCloudValidator(object):


    def __init__(self, dataset_path, device, net):
        self.device = device
        self.net = net
        self.num_samples = 100000
        self.dataset_path = dataset_path
        self.set_dataset()


    def set_dataset(self):

        # Same as training since we're overfitting
        self.val_dataset = dataio.MeshDataset(args=None, num_samples=self.num_samples, dataset_path=self.dataset_path, sample_mode=['trace'])
        self.val_data_loader = DataLoader(self.val_dataset,
                                          batch_size=self.num_samples,
                                          shuffle=False, pin_memory=True, num_workers=4)

    def validate(self, epoch):
        """Geometric validation; sample surface points."""
        default_args =  parse_options(return_parser = True).parse_args([])
        tracer = SphereTracer(args=default_args)
        res = tracer.sample_surface(self.num_samples, self.net)#.cpu()

        # Uniform points metrics
        for n_iter, (model_input, gt) in enumerate(self.val_data_loader):
            model_input = {key: value.to(self.device) for key, value in model_input.items()}
            pts = model_input['coords']
            d = float(chamfer_distance(pts.unsqueeze(0), res.unsqueeze(0))[0].detach().cpu())
            print('Chamfer Distance: ', d)

        return d
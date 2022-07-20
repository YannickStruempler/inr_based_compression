  # The MIT License (MIT)
#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from torch.utils.data import DataLoader
from shape_compression import dataio

from .metrics import *

class GeometricValidator(object):
    """Geometric validation; sample 3D points for distance/occupancy metrics."""

    def __init__(self, dataset_path, device, net):
        self.device = device
        self.net = net
        self.num_samples = 100000
        self.dataset_path = dataset_path
        self.set_dataset()

    
    def set_dataset(self):
        """Two datasets; 1) samples uniformly for volumetric IoU, 2) samples surfaces only."""

        # Same as training since we're overfitting
        self.val_volumetric_dataset = dataio.MeshDataset(args=None, num_samples=self.num_samples, dataset_path=self.dataset_path, sample_mode=['rand'])
        self.val_volumetric_data_loader = DataLoader(self.val_volumetric_dataset,
                                          batch_size=self.num_samples,
                                          shuffle=False, pin_memory=True, num_workers=4)


    def validate(self, epoch, is_meta_module=True):
        """Geometric validation; sample surface points."""

        
        # Uniform points metrics
        for n_iter, (model_input, gt) in enumerate(self.val_volumetric_data_loader):
            model_input = {key: value.to(self.device) for key, value in model_input.items()}
            gt = {key: value.to(self.device) for key, value in gt.items()}
            if not is_meta_module:
                model_input= model_input['coords']
            # Volumetric IoU
            pred = self.net(model_input)
            if is_meta_module:
                pred = pred['model_out']
            vol_iou =  float(compute_iou(gt['dist'], pred))


        return vol_iou


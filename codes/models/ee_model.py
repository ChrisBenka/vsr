from collections import OrderedDict

import torch
import torch.optim as optim

from .base_model import BaseModel
from .optim import define_criterion, define_lr_schedule, AlphaScheduler
from utils import base_utils, net_utils
from .edge_enhancer import EdgeEnhancement,EdgeExtraction,display_edges,display_frame


class EEOnly(BaseModel):
    """ A model wrapper for objective video super-resolution
    """

    def __init__(self, opt):
        super(EEOnly, self).__init__(opt)

        # define network
        self.sched_G = None
        self.optim_G = None
        self.ee_crit = None
        self.pp_crit = None
        self.shed_Alpha = None

        # set edge enhance params
        self.patch_size = opt['model']['edge_enhancer']['patch_size']
        self.enhancement_type = opt['model']['edge_enhancer']['type']
        self.alpha = 10
        self.total_loss = 0
        self.curr_iter = 0

        self.set_networks()

        # config training
        if self.is_train:
            self.set_criterions()
            self.set_optimizers()
            self.set_lr_schedules()

    def set_networks(self):
        # define edge enhancer
        self.edge_enhancer = EdgeEnhancement(edge_enhancement_type=self.enhancement_type)

        self.edge_enhancer = self.model_to_device(self.edge_enhancer, device="cuda:0")
        # load edge enhancer
        load_path_ee = self.opt['model']['edge_enhancer'].get('load_path')

        if load_path_ee is not None:
            self.load_network(self.edge_enhancer, load_path_ee)
            base_utils.log_info(f'Load edge enhancer from: {load_path_ee}')

    def set_criterions(self):
        # charbonier criterion
        self.ee_crit = define_criterion(
            self.opt['train'].get('edgeenhanced_crit'))

    def set_optimizers(self):
        self.optim_ee = optim.Adam(
            self.edge_enhancer.parameters(),
            lr=self.opt['train']['edge_enhancer']['lr'],
            weight_decay=self.opt['train']['edge_enhancer'].get('weight_decay', 0),
            betas=self.opt['train']['edge_enhancer'].get('betas', (0.9, 0.999)))

    def set_lr_schedules(self):
        self.sched_G = define_lr_schedule(
            self.opt['train']['edge_enhancer'].get('lr_schedule'), self.optim_ee)

    def train(self):
        """
        Train to match edges....
        """
        self.curr_iter += 1
        # === initialize === #
        self.edge_enhancer.train()
        self.optim_ee.zero_grad()


        # === forward net_ee === #
        self.log_dict = OrderedDict()

        # === edge enhance === #
        _, enhanced_edges = self.edge_enhancer(self.lr_data.squeeze(0).cuda())

        # === edge enhance loss  ===#
        # get edges from GT data using edge extraction
        gt_edges = laplacian_filter_functional(self.gt_data.squeeze(0))

        ee_loss = self.ee_crit(enhanced_edges,gt_edges)
        self.total_loss += ee_loss.item()
        self.log_dict["ee_charbonier_loss"] = (self.total_loss/self.curr_iter)
        # optimize
        ee_loss.backward()
        self.optim_ee.step()

    def infer(self):
        """ Infer the sr_base enhanceed edges sequence
            :return: np.ndarray sequence in type [uint8] and shape [thwc]
        """
        # infer
        self.edge_enhancer.eval()

        # enhance edges in frames
        _, enhanced_frames = self.edge_enhancer(self.lr_data)

        return enhanced_frames

    def save(self, current_iter):
        self.save_network(self.edge_enhancer, "E", current_iter)

from collections import OrderedDict

import torch
import torch.optim as optim

from .base_model import BaseModel
from .networks import define_generator
from .optim import define_criterion, define_lr_schedule
from utils import base_utils, net_utils, AlphaScheduler
from .edge_enhancer import EdgeEnhancement

class VSRModel(BaseModel):
    """ A model wrapper for objective video super-resolution
    """

    def __init__(self, opt):
        super(VSRModel, self).__init__(opt)

        # define network
        self.sched_G = None
        self.optim_G = None
        self.ee_crit = None
        self.pp_crit = None
        self.shed_Alpha = None
        self.set_networks()

        # set edge enhance params
        self.patch_size = opt['edge']['patch_size']
        self.edge_enhancer = EdgeEnhancement(patch_size=self.patch_size)
        self.alpha = 10
        # config training
        if self.is_train:
            self.set_criterions()
            self.set_optimizers()
            self.set_lr_schedules()

    def set_networks(self):
        # define generator
        self.net_G = define_generator(self.opt)
        self.net_G = self.model_to_device(self.net_G)
        base_utils.log_info('Edge Enhanced Generator: {}\n{}'.format(
            self.opt['model']['generator']['name'], self.net_G.__str__()))

        # load generator
        load_path_G = self.opt['model']['generator'].get('load_path')
        if load_path_G is not None:
            self.load_network(self.net_G, load_path_G)
            base_utils.log_info(f'Load generator from: {load_path_G}')

        # load edge enhancer
        load_path_ee = self.opt['model']['edge_enhancer'].get('load_path')
        if load_path_G is not None:
            self.load_network(self.edge_enhancer, load_path_ee)
            base_utils.log_info(f'Load generator from: {load_path_ee}')

    def set_criterions(self):
        # ping pong critertion
        self.pp_crit = define_criterion(self.opt['train'].get('pingpong_crit'))

        # edge enhanced criterion
        self.ee_crit = define_criterion(
            self.opt['train'].get('edgeenhanced_crit'))

    def set_optimizers(self):
        self.optim_G = optim.Adam(
            [{'params': self.net_G.parameters()},
             {'params': self.edge_enhancer.parameters()}
             ],
            lr=self.opt['train']['generator']['lr'],
            weight_decay=self.opt['train']['generator'].get('weight_decay', 0),
            betas=self.opt['train']['generator'].get('betas', (0.9, 0.999)))

    def set_lr_schedules(self):
        self.sched_G = define_lr_schedule(
            self.opt['train']['generator'].get('lr_schedule'), self.optim_G)

        self.shed_Alpha = AlphaScheduler(initial=10, decay_rate=1.1, decay_step=160)

    def train(self):
        # === get data === #
        lr_data, gt_data = self.lr_data, self.gt_data

        # === initialize === #
        self.net_G.train()
        self.edge_enhancer.train()
        self.optim_G.zero_grad()

        # === forward net_G === #
        loss_G = 0
        self.log_dict = OrderedDict()

        net_G_output_dict = self.net_G(self.lr_data)
        hr_data_base = net_G_output_dict['hr_data']

        # === edge enhance === #
        hr_data = self.edge_enhancer(hr_data_base)

        # === edge enhance loss  ===#
        ee_loss = self.ee_crit(hr_data_base,hr_data,gt_data,self.alpha)
        self.log_dict['ee_content_loss'] = ee_loss.item()
        loss_G += ee_loss

        # === ping pong loss === #
        tempo_extent = self.opt['train']['tempo_extent']

        hr_data_fw = hr_data[:, :tempo_extent - 1, ...]  # -------->|
        hr_data_bw = hr_data[:, tempo_extent:, ...].flip(1)  # <--------|
        pp_w = self.opt['train']['pingpong_crit'].get('weight', 1)
        loss_pp_G = pp_w * self.pp_crit(hr_data_fw, hr_data_bw)
        self.log_dict['pp_loss'] = loss_pp_G.item()
        
        loss_G += loss_pp_G

        # === other logging info ###
        self.log_dict['g_loss_total'] = loss_G.item()

        # === optimize net_G === #

        # optimize
        loss_G.backward()
        self.optim_G.step()

    def infer(self):
        """ Infer the `lr_data` sequence

            :return: np.ndarray sequence in type [uint8] and shape [thwc]
        """

        lr_data = self.lr_data

        # temporal padding
        lr_data, n_pad_front = self.pad_sequence(lr_data)

        # infer
        self.net_G.eval()
        self.edge_enhancer.eval()

        hr_seq = self.net_G(lr_data, self.device)
        # enhance edges in frames
        hr_seq = self.edge_enhancer(hr_seq)
        hr_seq = hr_seq[n_pad_front:]

        return hr_seq

    def save(self, current_iter):
        self.save_network(self.net_G, 'G', current_iter)
        self.save_network(self.edge_enhancer,"E",current_iter)

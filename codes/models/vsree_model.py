from collections import OrderedDict

import torch
import torch.optim as optim

from .base_model import BaseModel
from .networks import define_generator
from .optim import define_criterion, define_lr_schedule, AlphaScheduler
from utils import base_utils, net_utils
from .edge_enhancer import EdgeEnhancement, EdgeExtraction,EdgeDetector,display_frame,display_edges
import cv2

class VSREEModel(BaseModel):
    """ A model wrapper for objective video super-resolution
    """

    def __init__(self, opt):
        super(VSREEModel, self).__init__(opt)

        # define network
        self.sched_G = None
        self.optim_G = None
        self.ee_crit = None
        self.pp_crit = None
        self.shed_Alpha = None

        # set edge enhance params
        self.enhancement_type = opt['model']['edge_enhancer']['type']
        self.alpha = 10

        self.set_networks()
        self.current_iter = 0

        # config training
        if self.is_train:
            self.set_criterions()
            self.set_optimizers()
            self.set_lr_schedules()

    def set_networks(self):
        # define generator
        self.net_G = define_generator(self.opt)
        self.net_G = self.model_to_device(self.net_G, device="cuda:0")
        base_utils.log_info('Edge Enhanced Generator: {}\n{}'.format(
            self.opt['model']['generator']['name'], self.net_G.__str__()))

        # load generator
        load_path_G = self.opt['model']['generator'].get('load_path')
        if load_path_G is not None:
            self.load_network(self.net_G, load_path_G)
            base_utils.log_info(f'Load generator from: {load_path_G}')

        self.edge_enhancer = EdgeEnhancement(edge_enhancement_type=self.enhancement_type)

        self.edge_enhancer = self.model_to_device(self.edge_enhancer, device="cuda:0")
        # load edge enhancer
        load_path_ee = self.opt['model']['edge_enhancer'].get('load_path')

        if load_path_ee is not None:
            self.load_network(self.edge_enhancer, load_path_ee)
            base_utils.log_info(f'Load edge enhancer from: {load_path_ee}')

    def set_criterions(self):
        # ping pong critertion
        self.pp_crit = define_criterion(self.opt['train'].get('pingpong_crit'))

        # edge enhanced criterion
        self.ee_crit = define_criterion(
            self.opt['train'].get('edgeenhanced_crit'))

        # pixel criterion
        self.pix_crit = define_criterion(
            self.opt['train'].get('pixel_crit'))

    def set_optimizers(self):
        # self.optim_G = optim.Adam(
        #     self.net_G.parameters(),
        #     lr=self.opt['train']['generator']['lr'],
        #     weight_decay=self.opt['train']['generator'].get('weight_decay', 0),
        #     betas=self.opt['train']['generator'].get('betas', (0.9, 0.999)))
        self.optim_ee = optim.Adam(
            self.edge_enhancer.parameters(),
            lr=self.opt['train']['edge_enhancer']['lr'],
            weight_decay=self.opt['train']['edge_enhancer'].get('weight_decay', 0),
            betas=self.opt['train']['generator'].get('betas', (0.9, 0.999)))

    def set_lr_schedules(self):
        self.sched_G = define_lr_schedule(
            self.opt['train']['edge_enhancer'].get('lr_schedule'), self.optim_G)

        self.shed_Alpha = AlphaScheduler(initial=10, decay_rate=1.1, decay_step=25000)

    def train(self):
        # === get data === #
        lr_data, gt_data = self.lr_data, self.gt_data
        self.current_iter += 1

        # match_color = self.current_iter % 1000 == 0

        # === initialize === #
        self.edge_enhancer.train()
        self.optim_ee.zero_grad()

        # === forward net_G === #
        loss_G = 0
        self.log_dict = OrderedDict()

        # === edge enhance === #
        hr_data, edges = self.edge_enhancer(self.lr_data.squeeze(0).cuda())

        # pixel loss on image
        loss_pix_G_base = self.pix_crit(lr_data.squeeze(0).cuda().detach(), self.gt_data.squeeze(0).detach())
        loss_pix_E_base = self.pix_crit(hr_data.detach(), self.gt_data.squeeze(0).detach())

        self.log_dict['l_pix_B'] = loss_pix_G_base.item()
        self.log_dict['l_pix_ee'] = loss_pix_E_base.item()
        _,gt_edges_gray = self.edge_enhancer.edge_detector(self.gt_data.squeeze(0))

        loss_pix_edges = self.pix_crit(edges, gt_edges_gray)

        loss_G = loss_pix_edges + loss_pix_E_base

        self.log_dict['l_pix_edges'] = loss_pix_edges.item()

        # === other logging info ###
        # self.log_dict['g_loss_total'] = loss_G.item()

        # === optimize net_G === #

        # optimize
        loss_G.backward()
        self.optim_ee.step()

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
        for i in range(hr_seq.shape[0]):
            hr_seq[i,:,:,:] = cv2.normalize(hr_seq[i], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        hr_seq, _ = self.edge_enhancer(torch.tensor(hr_seq).permute(0, 3, 1, 2))
        hr_seq = hr_seq[n_pad_front:]

        return hr_seq

    def save(self, current_iter):
        # self.save_network(self.net_G, 'G', current_iter)
        self.save_network(self.edge_enhancer, "vsree_3", current_iter)

from collections import OrderedDict

import torch
import torch.optim as optim
from piqa import SSIM

from .base_model import BaseModel
from .networks import define_generator
from .optim import define_criterion, define_lr_schedule, AlphaScheduler
from utils import base_utils, net_utils
from .edge_enhancer import EdgeEnhancement, EdgeExtraction, EdgeDetector, display_frame, display_edges
import cv2
import numpy as np


class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)


# criterion = SSIMLoss().cpu()  # .cuda() if you need GPU support
criterion = torch.nn.MSELoss(reduction='mean')# .cuda() if you need GPU support
eval = SSIM()
ssim_eval = SSIM().cuda()

def normalize_tensor(x):
    return  (x - x.min()) / (x.max() - x.min())


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

        self.ssims = []
        self.edge_ssims = []
        self.alpha = 10

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

    def running_mean_last_500(self, data):
        # Convert data to PyTorch tensor

        # Extract the last 500 items
        last_500 = self.losses[-500:]

        # Compute the running mean
        running_mean = np.mean(last_500)

        return running_mean.item()  # Co

    def set_optimizers(self):
        self.optim_G = optim.Adam([
             {"params":self.net_G.parameters()},
             {"params": self.edge_enhancer.parameters()}
        ]   ,
             lr=self.opt['train']['generator']['lr'],
             weight_decay=self.opt['train']['generator'].get('weight_decay', 0),
             betas=self.opt['train']['generator'].get('betas', (0.9, 0.999)))
        # self.optim_ee = optim.Adam(
        #     self.edge_enhancer.parameters(),
        #     lr=self.opt['train']['edge_enhancer']['lr'],
        #     weight_decay=self.opt['train']['edge_enhancer'].get('weight_decay', 0),
        #     betas=self.opt['train']['generator'].get('betas', (0.9, 0.999)))

    def set_lr_schedules(self):
        self.sched_G = define_lr_schedule(
            self.opt['train']['generator'].get('lr_schedule'), self.optim_G)

        self.shed_Alpha = AlphaScheduler(initial=10, decay_rate=1.1, decay_step=20000)

    def train(self):
        # === initialize === #
        self.net_G.train()
        self.edge_enhancer.train()
        self.optim_G.zero_grad()
        self.log_dict = OrderedDict()

        lr_data, gt_data = self.lr_data, self.gt_data

        n, t, c, lr_h, lr_w = lr_data.size()
        _, _, _, gt_h, gt_w = gt_data.size()


        # === forward net_G === #
        net_G_output_dict = self.net_G(lr_data)
        hr_data_base = net_G_output_dict['hr_data'].cuda()
        hr_data_final, edges = self.edge_enhancer(hr_data_base.squeeze(0).cuda())
        edges = edges.cpu()
        final_ssim_loss = criterion(hr_data_final.cuda() * 255,gt_data.squeeze(0).cuda())
        base_ssim_loss =  criterion(normalize_tensor(hr_data_base.squeeze(0).cuda()) * 255,gt_data.squeeze(0).cuda())
        self.log_dict['final_mse'] = final_ssim_loss.item()
        self.log_dict['base_mse'] =  base_ssim_loss.item()
        edges_gt = self.edge_enhancer.edge_detector(gt_data.squeeze(0))[0]
        edge_loss =  criterion(edges.cuda() * 255,edges_gt.cuda() * 255)
        self.log_dict['edge_loss'] = edge_loss.item()


        # # generate bicubic upsampled data
        # upsample_fn = self.get_bare_model(self.net_G).upsample_func
        # bi_data = upsample_fn(
        #     lr_data.view(n * t, c, lr_h, lr_w)).view(n, t, c, gt_h, gt_w)

        # # augment data for pingpong criterion
        # # i.e., (0,city,2,...,t-2,t-city) -> (0,city,2,...,t-2,t-city,t-2,...,2,city,0)
        # if self.pp_crit is not None:
        #     lr_rev = lr_data.flip(1)[:, 1:, ...]
        #     gt_rev = gt_data.flip(1)[:, 1:, ...]
        #     bi_rev = bi_data.flip(1)[:, 1:, ...]

        #     lr_data = torch.cat([lr_data, lr_rev], dim=1)
        #     gt_data = torch.cat([gt_data, gt_rev], dim=1)
        #     bi_data = torch.cat([bi_data, bi_rev], dim=1)

 

        # # calculate losses
        # loss_G = 0

        # # ping-pong (pp) loss
        # if self.pp_crit is not None:
        #     tempo_extent = self.opt['train']['tempo_extent']
        #     hr_data_fw = hr_data_final[:, :tempo_extent - 1, ...]      # -------->|
        #     hr_data_bw = hr_data_final[:, tempo_extent:, ...].flip(1)  # <--------|

        #     pp_w = self.opt['train']['pingpong_crit'].get('weight', 1)
        #     loss_pp_G = pp_w * self.pp_crit(hr_data_fw * 255, hr_data_bw * 255)
            # self.log_dict['l_pp_G'] = loss_pp_G.item()

        alpha = self.shed_Alpha.step()
        # self.log_dict['alpha']  = alpha
        content_loss =  final_ssim_loss + base_ssim_loss + edge_loss
        loss_G = content_loss 
        self.log_dict['total_loss_G'] = loss_G.item()

        # update net_G
        loss_G.backward()
        self.optim_G.step()
    
    def infer(self):
        """ Infer the `lr_data` sequence

            :return: np.ndarray sequence in type [uint8] and shape [thwc]
        """
        lr_data = self.lr_data

        # temporal padding
        # lr_data, n_pad_front = self.pad_sequence(lr_data)

        # infer
        self.net_G.eval()
        self.edge_enhancer.eval()

        hr_seq_base = self.net_G(lr_data, self.device)
        # enhance edges in frames
        hr_seq_final, _ = self.edge_enhancer(torch.tensor(hr_seq_base).permute(0,-1,1,2).cuda())
        # returns normalized between 0 and 1, lets fix that. 
        return  hr_seq_final

    def save(self, current_iter):
        self.save_network(self.net_G, 'G', current_iter)
        self.save_network(self.edge_enhancer, "E", current_iter)

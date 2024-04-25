
from __future__ import absolute_import

import sys
import numpy as np
import torch
from torch import nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
from .base_model import BaseModel
from scipy.ndimage import zoom
import fractions
import functools
import skimage.transform
from tqdm import tqdm

from IPython import embed

from . import networks_basic as networks

import numpy as np
from PIL import Image
import inspect
import re
import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom
from skimage.metrics import structural_similarity as compare_ssim
import torch
from IPython import embed
import cv2
from datetime import datetime

def datetime_str():
    now = datetime.now()
    return '%04d-%02d-%02d-%02d-%02d-%02d'%(now.year,now.month,now.day,now.hour,now.minute,now.second)

def read_text_file(in_path):
    fid = open(in_path,'r')

    vals = []
    cur_line = fid.readline()
    while(cur_line!=''):
        vals.append(float(cur_line))
        cur_line = fid.readline()

    fid.close()
    return np.array(vals)

def bootstrap(in_vec,num_samples=100,bootfunc=np.mean):
    from astropy import stats
    return stats.bootstrap(np.array(in_vec),bootnum=num_samples,bootfunc=bootfunc)

def rand_flip(input1,input2):
    if(np.random.binomial(1,.5)==1):
        return (input1,input2)
    else:
        return (input2,input1)

def l2(p0, p1, range=255.):
    return .5*np.mean((p0 / range - p1 / range)**2)

def psnr(p0, p1, peak=255.):
    return 10*np.log10(peak**2/np.mean((1.*p0-1.*p1)**2))

def dssim(p0, p1, range=255.):
    # embed()
    return (1 - compare_ssim(p0, p1, data_range=range, multichannel=True)) / 2.

def rgb2lab(in_img,mean_cent=False):
    from skimage import color
    img_lab = color.rgb2lab(in_img)
    if(mean_cent):
        img_lab[:,:,0] = img_lab[:,:,0]-50
    return img_lab

def normalize_blob(in_feat,eps=1e-10):
    norm_factor = np.sqrt(np.sum(in_feat**2,axis=1,keepdims=True))
    return in_feat/(norm_factor+eps)

def cos_sim_blob(in0,in1):
    in0_norm = normalize_blob(in0)
    in1_norm = normalize_blob(in1)
    (N,C,X,Y) = in0_norm.shape

    return np.mean(np.mean(np.sum(in0_norm*in1_norm,axis=1),axis=1),axis=1)

def normalize_tensor(in_feat,eps=1e-10):
    # norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1)).view(in_feat.size()[0],1,in_feat.size()[2],in_feat.size()[3]).repeat(1,in_feat.size()[1],1,1)
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1)).view(in_feat.size()[0],1,in_feat.size()[2],in_feat.size()[3])
    return in_feat/(norm_factor.expand_as(in_feat)+eps)

def cos_sim(in0,in1):
    in0_norm = normalize_tensor(in0)
    in1_norm = normalize_tensor(in1)
    N = in0.size()[0]
    X = in0.size()[2]
    Y = in0.size()[3]

    return torch.mean(torch.mean(torch.sum(in0_norm*in1_norm,dim=1).view(N,1,X,Y),dim=2).view(N,1,1,Y),dim=3).view(N)

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the conve

def tensor2np(tensor_obj):
    # change dimension of a tensor object into a numpy array
    return tensor_obj[0].cpu().float().numpy().transpose((1,2,0))

def np2tensor(np_obj):
     # change dimenion of np array into tensor array
    return torch.Tensor(np_obj[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

def tensor2tensorlab(image_tensor,to_norm=True,mc_only=False):
    # image tensor to lab tensor
    from skimage import color

    img = tensor2im(image_tensor)
    # print('img_rgb',img.flatten())
    img_lab = color.rgb2lab(img)
    # print('img_lab',img_lab.flatten())
    if(mc_only):
        img_lab[:,:,0] = img_lab[:,:,0]-50
    if(to_norm and not mc_only):
        img_lab[:,:,0] = img_lab[:,:,0]-50
        img_lab = img_lab/100.

    return np2tensor(img_lab)

def tensorlab2tensor(lab_tensor,return_inbnd=False):
    from skimage import color
    import warnings
    warnings.filterwarnings("ignore")

    lab = tensor2np(lab_tensor)*100.
    lab[:,:,0] = lab[:,:,0]+50
    # print('lab',lab)

    rgb_back = 255.*np.clip(color.lab2rgb(lab.astype('float')),0,1)
    # print('rgb',rgb_back)
    if(return_inbnd):
        # convert back to lab, see if we match
        lab_back = color.rgb2lab(rgb_back.astype('uint8'))
        # print('lab_back',lab_back)
        # print('lab==lab_back',np.isclose(lab_back,lab,atol=1.))
        # print('lab-lab_back',np.abs(lab-lab_back))
        mask = 1.*np.isclose(lab_back,lab,atol=2.)
        mask = np2tensor(np.prod(mask,axis=2)[:,:,np.newaxis])
        return (im2tensor(rgb_back),mask)
    else:
        return im2tensor(rgb_back)

def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=255./2.):
# def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=1.):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + cent) * factor
    return image_numpy.astype(imtype)

def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
# def im2tensor(image, imtype=np.uint8, cent=1., factor=1.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

def tensor2vec(vector_tensor):
    return vector_tensor.data.cpu().numpy()[:, :, 0, 0]

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)

def grab_patch(img_in, P, yy, xx):
    return img_in[yy:yy+P,xx:xx+P,:]

def load_image(path):
    if(path[-3:] == 'dng'):
        import rawpy
        with rawpy.imread(path) as raw:
            img = raw.postprocess()
        # img = plt.imread(path)
    elif(path[-3:]=='bmp' or path[-3:]=='jpg' or path[-3:]=='png'):
        import cv2
        return cv2.imread(path)[:,:,::-1]
    else:
        img = (255*plt.imread(path)[:,:,:3]).astype('uint8')

    return img


def resize_image(img, max_size=256):
    [Y, X] = img.shape[:2]

    # resize
    max_dim = max([Y, X])
    zoom_factor = 1. * max_size / max_dim
    img = zoom(img, [zoom_factor, zoom_factor, 1])

    return img

def resize_image_zoom(img, zoom_factor=1., order=3):
    if(zoom_factor==1):
        return img
    else:
        return zoom(img, [zoom_factor, zoom_factor, 1], order=order)

def save_image(image_numpy, image_path, ):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def prep_display_image(img, dtype='uint8'):
    if(dtype == 'uint8'):
        return np.clip(img, 0, 255).astype('uint8')
    else:
        return np.clip(img, 0, 1.)


def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [
        e for e in dir(object) if isinstance(
            getattr(
                object,
                e),
            collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print("\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]))


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print(
            'mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' %
            (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def rgb2lab(input):
    from skimage import color
    return color.rgb2lab(input / 255.)


def montage(
    imgs,
    PAD=5,
    RATIO=16 / 9.,
    EXTRA_PAD=(
        False,
        False),
        MM=-1,
        NN=-1,
        primeDir=0,
        verbose=False,
        returnGridPos=False,
        backClr=np.array(
            (0,
             0,
             0))):
    # INPUTS
    #   imgs        YxXxMxN or YxXxN
    #   PAD         scalar              number of pixels in between
    #   RATIO       scalar              target ratio of cols/rows
    #   MM          scalar              # rows, if specified, overrides RATIO
    #   NN          scalar              # columns, if specified, overrides RATIO
    #   primeDir    scalar              0 for top-to-bottom, 1 for left-to-right
    # OUTPUTS
    #   mont_imgs   MM*Y x NN*X x M     big image with everything montaged
    # def montage(imgs, PAD=5, RATIO=16/9., MM=-1, NN=-1, primeDir=0,
    # verbose=False, forceFloat=False):
    if(imgs.ndim == 3):
        toExp = True
        imgs = imgs[:, :, np.newaxis, :]
    else:
        toExp = False

    Y = imgs.shape[0]
    X = imgs.shape[1]
    M = imgs.shape[2]
    N = imgs.shape[3]

    PADS = np.array((PAD))
    if(PADS.flatten().size == 1):
        PADY = PADS
        PADX = PADS
    else:
        PADY = PADS[0]
        PADX = PADS[1]

    if(MM == -1 and NN == -1):
        NN = np.ceil(np.sqrt(1.0 * N * RATIO))
        MM = np.ceil(1.0 * N / NN)
        NN = np.ceil(1.0 * N / MM)
    elif(MM == -1):
        MM = np.ceil(1.0 * N / NN)
    elif(NN == -1):
        NN = np.ceil(1.0 * N / MM)

    if(primeDir == 0):  # write top-to-bottom
        [grid_mm, grid_nn] = np.meshgrid(
            np.arange(MM, dtype='uint'), np.arange(NN, dtype='uint'))
    elif(primeDir == 1):  # write left-to-right
        [grid_nn, grid_mm] = np.meshgrid(
            np.arange(NN, dtype='uint'), np.arange(MM, dtype='uint'))

    grid_mm = np.uint(grid_mm.flatten()[0:N])
    grid_nn = np.uint(grid_nn.flatten()[0:N])

    EXTRA_PADY = EXTRA_PAD[0] * PADY
    EXTRA_PADX = EXTRA_PAD[0] * PADX

    # mont_imgs = np.zeros(((Y+PAD)*MM-PAD, (X+PAD)*NN-PAD, M), dtype=use_dtype)
    mont_imgs = np.zeros(
        (np.uint(
            (Y + PADY) * MM - PADY + EXTRA_PADY),
            np.uint(
            (X + PADX) * NN - PADX + EXTRA_PADX),
            M),
        dtype=imgs.dtype)
    mont_imgs = mont_imgs + \
        backClr.flatten()[np.newaxis, np.newaxis, :].astype(mont_imgs.dtype)

    for ii in np.random.permutation(N):
        # print imgs[:,:,:,ii].shape
        # mont_imgs[grid_mm[ii]*(Y+PAD):(grid_mm[ii]*(Y+PAD)+Y), grid_nn[ii]*(X+PAD):(grid_nn[ii]*(X+PAD)+X),:]
        mont_imgs[np.uint(grid_mm[ii] *
                          (Y +
                           PADY)):np.uint((grid_mm[ii] *
                                           (Y +
                                            PADY) +
                                           Y)), np.uint(grid_nn[ii] *
                                                        (X +
                                                         PADX)):np.uint((grid_nn[ii] *
                                                                         (X +
                                                                          PADX) +
                                                                         X)), :] = imgs[:, :, :, ii]

    if(M == 1):
        imgs = imgs.reshape(imgs.shape[0], imgs.shape[1], imgs.shape[3])

    if(toExp):
        mont_imgs = mont_imgs[:, :, 0]

    if(returnGridPos):
        # return (mont_imgs,np.concatenate((grid_mm[:,:,np.newaxis]*(Y+PAD),
        # grid_nn[:,:,np.newaxis]*(X+PAD)),axis=2))
        return (mont_imgs, np.concatenate(
            (grid_mm[:, np.newaxis] * (Y + PADY), grid_nn[:, np.newaxis] * (X + PADX)), axis=1))
        # return (mont_imgs, (grid_mm,grid_nn))
    else:
        return mont_imgs

class zeroClipper(object):
    def __init__(self, frequency=1):
        self.frequency = frequency

    def __call__(self, module):
        embed()
        if hasattr(module, 'weight'):
            # module.weight.data = torch.max(module.weight.data, 0)
            module.weight.data = torch.max(module.weight.data, 0) + 100

def flatten_nested_list(nested_list):
    # only works for list of list
    accum = []
    for sublist in nested_list:
        for item in sublist:
            accum.append(item)
    return accum

def read_file(in_path,list_lines=False):
    agg_str = ''
    f = open(in_path,'r')
    cur_line = f.readline()
    while(cur_line!=''):
        agg_str+=cur_line
        cur_line = f.readline()
    f.close()
    if(list_lines==False):
        return agg_str.replace('\n','')
    else:
        line_list = agg_str.split('\n')
        ret_list = []
        for item in line_list:
            if(item!=''):
                ret_list.append(item)
        return ret_list

def read_csv_file_as_text(in_path):
    agg_str = []
    f = open(in_path,'r')
    cur_line = f.readline()
    while(cur_line!=''):
        agg_str.append(cur_line)
        cur_line = f.readline()
    f.close()
    return agg_str

def random_swap(obj0,obj1):
    if(np.random.rand() < .5):
        return (obj0,obj1,0)
    else:
        return (obj1,obj0,1)

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


class DistModel(BaseModel):
    def name(self):
        return self.model_name

    def initialize(self, model='net-lin', net='alex', colorspace='Lab', pnet_rand=False, pnet_tune=False, model_path=None,
            use_gpu=True, printNet=False, spatial=False,
            is_train=False, lr=.0001, beta1=0.5, version='0.1', gpu_ids=[0]):
        '''
        INPUTS
            model - ['net-lin'] for linearly calibrated network
                    ['net'] for off-the-shelf network
                    ['L2'] for L2 distance in Lab colorspace
                    ['SSIM'] for ssim in RGB colorspace
            net - ['squeeze','alex','vgg']
            model_path - if None, will look in weights/[NET_NAME].pth
            colorspace - ['Lab','RGB'] colorspace to use for L2 and SSIM
            use_gpu - bool - whether or not to use a GPU
            printNet - bool - whether or not to print network architecture out
            spatial - bool - whether to output an array containing varying distances across spatial dimensions
            is_train - bool - [True] for training mode
            lr - float - initial learning rate
            beta1 - float - initial momentum term for adam
            version - 0.1 for latest, 0.0 was original (with a bug)
            gpu_ids - int array - [0] by default, gpus to use
        '''
        BaseModel.initialize(self, use_gpu=use_gpu, gpu_ids=gpu_ids)

        self.model = model
        self.net = net
        self.is_train = is_train
        self.spatial = spatial
        self.gpu_ids = gpu_ids
        self.model_name = '%s [%s]'%(model,net)

        if(self.model == 'net-lin'): # pretrained net + linear layer
            self.net = networks.PNetLin(pnet_rand=pnet_rand, pnet_tune=pnet_tune, pnet_type=net,
                use_dropout=True, spatial=spatial, version=version, lpips=True)
            #kw = {}
            #if not use_gpu:
            #    kw['map_location'] = 'cpu'
            if(model_path is None):
                import inspect
                model_path = os.path.abspath(os.path.join(inspect.getfile(self.initialize), '..', 'weights/v%s/%s.pth'%(version,net)))

            if(not is_train):
                # self.net.load_state_dict(torch.load(model_path, **kw), strict=False)
                state_dict = torch.load(
                    model_path, map_location=lambda storage, loc: storage)
                self.net.load_state_dict(state_dict, strict=False)

        elif(self.model=='net'): # pretrained network
            self.net = networks.PNetLin(pnet_rand=pnet_rand, pnet_type=net, lpips=False)
        elif(self.model in ['L2','l2']):
            self.net = networks.L2(use_gpu=use_gpu,colorspace=colorspace) # not really a network, only for testing
            self.model_name = 'L2'
        elif(self.model in ['DSSIM','dssim','SSIM','ssim']):
            self.net = networks.DSSIM(use_gpu=use_gpu,colorspace=colorspace)
            self.model_name = 'SSIM'
        else:
            raise ValueError("Model [%s] not recognized." % self.model)

        self.parameters = list(self.net.parameters())

        if self.is_train: # training mode
            # extra network on top to go from distances (d0,d1) => predicted human judgment (h*)
            self.rankLoss = networks.BCERankingLoss()
            self.parameters += list(self.rankLoss.net.parameters())
            self.lr = lr
            self.old_lr = lr
            self.optimizer_net = torch.optim.Adam(self.parameters, lr=lr, betas=(beta1, 0.999))
        else: # test mode
            self.net.eval()

        if(use_gpu):
            self.net.to(gpu_ids[0])
            #self.net = torch.nn.DataParallel(self.net, device_ids=gpu_ids)
            if(self.is_train):
                self.rankLoss = self.rankLoss.to(device=gpu_ids[0]) # just put this on GPU0

        if(printNet):
            print('---------- Networks initialized -------------')
            networks.print_network(self.net)
            print('-----------------------------------------------')

    def forward(self, in0, in1, retPerLayer=False):
        ''' Function computes the distance between image patches in0 and in1
        INPUTS
            in0, in1 - torch.Tensor object of shape Nx3xXxY - image patch scaled to [-1,1]
        OUTPUT
            computed distances between in0 and in1
        '''

        return self.net.forward(in0, in1, retPerLayer=retPerLayer)

    # ***** TRAINING FUNCTIONS *****
    def optimize_parameters(self):
        self.forward_train()
        self.optimizer_net.zero_grad()
        self.backward_train()
        self.optimizer_net.step()
        self.clamp_weights()

    def clamp_weights(self):
        for module in self.net.modules():
            if(hasattr(module, 'weight') and module.kernel_size==(1,1)):
                module.weight.data = torch.clamp(module.weight.data,min=0)

    def set_input(self, data):
        self.input_ref = data['ref']
        self.input_p0 = data['p0']
        self.input_p1 = data['p1']
        self.input_judge = data['judge']

        if(self.use_gpu):
            self.input_ref = self.input_ref.to(device=self.gpu_ids[0])
            self.input_p0 = self.input_p0.to(device=self.gpu_ids[0])
            self.input_p1 = self.input_p1.to(device=self.gpu_ids[0])
            self.input_judge = self.input_judge.to(device=self.gpu_ids[0])

        self.var_ref = Variable(self.input_ref,requires_grad=True)
        self.var_p0 = Variable(self.input_p0,requires_grad=True)
        self.var_p1 = Variable(self.input_p1,requires_grad=True)

    def forward_train(self): # run forward pass
        # print(self.net.module.scaling_layer.shift)
        # print(torch.norm(self.net.module.net.slice1[0].weight).item(), torch.norm(self.net.module.lin0.model[1].weight).item())

        self.d0 = self.forward(self.var_ref, self.var_p0)
        self.d1 = self.forward(self.var_ref, self.var_p1)
        self.acc_r = self.compute_accuracy(self.d0,self.d1,self.input_judge)

        self.var_judge = Variable(1.*self.input_judge).view(self.d0.size())

        self.loss_total = self.rankLoss.forward(self.d0, self.d1, self.var_judge*2.-1.)

        return self.loss_total

    def backward_train(self):
        torch.mean(self.loss_total).backward()

    def compute_accuracy(self,d0,d1,judge):
        ''' d0, d1 are Variables, judge is a Tensor '''
        d1_lt_d0 = (d1<d0).cpu().data.numpy().flatten()
        judge_per = judge.cpu().numpy().flatten()
        return d1_lt_d0*judge_per + (1-d1_lt_d0)*(1-judge_per)

    def get_current_errors(self):
        retDict = OrderedDict([('loss_total', self.loss_total.data.cpu().numpy()),
                            ('acc_r', self.acc_r)])

        for key in retDict.keys():
            retDict[key] = np.mean(retDict[key])

        return retDict

    def get_current_visuals(self):
        zoom_factor = 256/self.var_ref.data.size()[2]

        ref_img = util.tensor2im(self.var_ref.data)
        p0_img = util.tensor2im(self.var_p0.data)
        p1_img = util.tensor2im(self.var_p1.data)

        ref_img_vis = zoom(ref_img,[zoom_factor, zoom_factor, 1],order=0)
        p0_img_vis = zoom(p0_img,[zoom_factor, zoom_factor, 1],order=0)
        p1_img_vis = zoom(p1_img,[zoom_factor, zoom_factor, 1],order=0)

        return OrderedDict([('ref', ref_img_vis),
                            ('p0', p0_img_vis),
                            ('p1', p1_img_vis)])

    def save(self, path, label):
        if(self.use_gpu):
            self.save_network(self.net.module, path, '', label)
        else:
            self.save_network(self.net, path, '', label)
        self.save_network(self.rankLoss.net, path, 'rank', label)

    def update_learning_rate(self,nepoch_decay):
        lrd = self.lr / nepoch_decay
        lr = self.old_lr - lrd

        for param_group in self.optimizer_net.param_groups:
            param_group['lr'] = lr

        print('update lr [%s] decay: %f -> %f' % (type,self.old_lr, lr))
        self.old_lr = lr

def score_2afc_dataset(data_loader, func, name=''):
    ''' Function computes Two Alternative Forced Choice (2AFC) score using
        distance function 'func' in dataset 'data_loader'
    INPUTS
        data_loader - CustomDatasetDataLoader object - contains a TwoAFCDataset inside
        func - callable distance function - calling d=func(in0,in1) should take 2
            pytorch tensors with shape Nx3xXxY, and return numpy array of length N
    OUTPUTS
        [0] - 2AFC score in [0,1], fraction of time func agrees with human evaluators
        [1] - dictionary with following elements
            d0s,d1s - N arrays containing distances between reference patch to perturbed patches
            gts - N array in [0,1], preferred patch selected by human evaluators
                (closer to "0" for left patch p0, "1" for right patch p1,
                "0.6" means 60pct people preferred right patch, 40pct preferred left)
            scores - N array in [0,1], corresponding to what percentage function agreed with humans
    CONSTS
        N - number of test triplets in data_loader
    '''

    d0s = []
    d1s = []
    gts = []

    for data in tqdm(data_loader.load_data(), desc=name):
        d0s+=func(data['ref'],data['p0']).data.cpu().numpy().flatten().tolist()
        d1s+=func(data['ref'],data['p1']).data.cpu().numpy().flatten().tolist()
        gts+=data['judge'].cpu().numpy().flatten().tolist()

    d0s = np.array(d0s)
    d1s = np.array(d1s)
    gts = np.array(gts)
    scores = (d0s<d1s)*(1.-gts) + (d1s<d0s)*gts + (d1s==d0s)*.5

    return(np.mean(scores), dict(d0s=d0s,d1s=d1s,gts=gts,scores=scores))

def score_jnd_dataset(data_loader, func, name=''):
    ''' Function computes JND score using distance function 'func' in dataset 'data_loader'
    INPUTS
        data_loader - CustomDatasetDataLoader object - contains a JNDDataset inside
        func - callable distance function - calling d=func(in0,in1) should take 2
            pytorch tensors with shape Nx3xXxY, and return pytorch array of length N
    OUTPUTS
        [0] - JND score in [0,1], mAP score (area under precision-recall curve)
        [1] - dictionary with following elements
            ds - N array containing distances between two patches shown to human evaluator
            sames - N array containing fraction of people who thought the two patches were identical
    CONSTS
        N - number of test triplets in data_loader
    '''

    ds = []
    gts = []

    for data in tqdm(data_loader.load_data(), desc=name):
        ds+=func(data['p0'],data['p1']).data.cpu().numpy().tolist()
        gts+=data['same'].cpu().numpy().flatten().tolist()

    sames = np.array(gts)
    ds = np.array(ds)

    sorted_inds = np.argsort(ds)
    ds_sorted = ds[sorted_inds]
    sames_sorted = sames[sorted_inds]

    TPs = np.cumsum(sames_sorted)
    FPs = np.cumsum(1-sames_sorted)
    FNs = np.sum(sames_sorted)-TPs

    precs = TPs/(TPs+FPs)
    recs = TPs/(TPs+FNs)
    score = voc_ap(recs,precs)

    return(score, dict(ds=ds,sames=sames))
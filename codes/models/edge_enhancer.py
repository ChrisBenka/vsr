import functools
import operator
import time
from enum import Enum

import cv2
import numpy as np
import torch
from PIL import Image


### SRC/INSPIRATION : https://github.com/kuijiang94/EEGAN/blob/master/src/EEGANx4.py

def float32_to_uint8(inputs):
    """ Convert np.float32 array to np.uint8

        Parameters:
            :param input: np.float32, (NT)CHW, [0, city]
            :return: np.uint8, (NT)CHW, [0, 255]
    """
    return np.uint8(np.clip(np.round(inputs * 255), 0, 255))


class EdgeExtraction(Enum):
    LAPLACIAN = 1
    SOBEL = 2
    BOTH = 3


from torch import nn
import torch.nn.functional as F
#https://github.com/DCurro/CannyEdgePytorch/blob/master/net_canny.py
def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_2D = gaussian_2D / (2 * np.pi *sigma **2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D
    #https://github.com/DCurro/CannyEdgePytorch/blob/master/net_canny.py

def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D
    #https://github.com/DCurro/CannyEdgePytorch/blob/master/net_canny.py

def get_thin_kernels(start=0, end=360, step=45):
    k_thin = 3  # actual size of the directional kernel
    # increase for a while to avoid interpolation when rotating
    k_increased = k_thin + 2

    # get 0° angle directional kernel
    thin_kernel_0 = np.zeros((k_increased, k_increased))
    thin_kernel_0[k_increased // 2, k_increased // 2] = 1
    thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

    # rotate the 0° angle directional kernel to get the other ones
    thin_kernels = []
    for angle in range(start, end, step):
        (h, w) = thin_kernel_0.shape
        # get the center to not rotate around the (0, 0) coord point
        center = (w // 2, h // 2)
        # apply rotation
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        kernel_angle_increased = cv2.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)

        # get the k=3 kerne
        kernel_angle = kernel_angle_increased[1:-1, 1:-1]
        is_diag = (abs(kernel_angle) == 1)  # because of the interpolation
        kernel_angle = kernel_angle * is_diag  # because of the interpolation
        thin_kernels.append(kernel_angle)
    return thin_kernels

class CannyFilter(nn.Module):
    #https://github.com/DCurro/CannyEdgePytorch/blob/master/net_canny.py
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 use_cuda=True):
        super(CannyFilter, self).__init__()
        # device
        self.device = 'cuda' if use_cuda else 'cpu'

        # gaussian

        gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=k_gaussian,
                                         padding=k_gaussian // 2,
                                         bias=False)
        for param in self.gaussian_filter.parameters():
            param.requires_grad = False

        self.gaussian_filter.weight[:] = torch.from_numpy(gaussian_2D)

        # sobel

        sobel_2D = get_sobel_kernel(k_sobel)
        self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        for param in self.sobel_filter_x.parameters():
            param.requires_grad = False
        self.sobel_filter_x.weight[:] = torch.from_numpy(sobel_2D)


        self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        for param in self.sobel_filter_y.parameters():
            param.requires_grad = False

        self.sobel_filter_y.weight[:] = torch.from_numpy(sobel_2D.T)


        # thin

        thin_kernels = get_thin_kernels()
        directional_kernels = np.stack(thin_kernels)

        self.directional_filter = nn.Conv2d(in_channels=1,
                                            out_channels=8,
                                            kernel_size=thin_kernels[0].shape,
                                            padding=thin_kernels[0].shape[-1] // 2,
                                            bias=False)
        for param in self.directional_filter.parameters():
            param.requires_grad = False
        self.directional_filter.weight[:, 0] = torch.from_numpy(directional_kernels)

        # hysteresis

        hysteresis = np.ones((3, 3)) + 0.25
        self.hysteresis = nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False)
        for param in self.hysteresis.parameters():
            param.requires_grad = False
        self.hysteresis.weight[:] = torch.from_numpy(hysteresis)


    def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=False):
        # set the setps tensors
        B, C, H, W = img.shape
        blurred = torch.zeros((B, C, H, W)).to(self.device)
        grad_x = torch.zeros((B, 1, H, W)).to(self.device)
        grad_y = torch.zeros((B, 1, H, W)).to(self.device)
        grad_magnitude = torch.zeros((B, 1, H, W)).to(self.device)
        grad_orientation = torch.zeros((B, 1, H, W)).to(self.device)

        # gaussian

        for c in range(C):
            blurred[:, c:c+1] = self.gaussian_filter(img[:, c:c+1])

            grad_x = grad_x + self.sobel_filter_x(blurred[:, c:c+1])
            grad_y = grad_y + self.sobel_filter_y(blurred[:, c:c+1])

        # thick edges

        grad_x, grad_y = grad_x / C, grad_y / C
        grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5
        grad_orientation = torch.atan(grad_y / grad_x)
        grad_orientation = grad_orientation * (360 / np.pi) + 180 # convert to degree
        grad_orientation = torch.round(grad_orientation / 45) * 45  # keep a split by 45

        # thin edges

        directional = self.directional_filter(grad_magnitude)
        # get indices of positive and negative directions
        positive_idx = (grad_orientation / 45) % 8
        negative_idx = ((grad_orientation / 45) + 4) % 8
        thin_edges = grad_magnitude.clone()
        # non maximum suppression direction by direction
        for pos_i in range(4):
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = (positive_idx == pos_i) * 1
            is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
            pos_directional = directional[:, pos_i]
            neg_directional = directional[:, neg_i]
            selected_direction = torch.stack([pos_directional, neg_directional])

            # get the local maximum pixels for the angle
            is_max = selected_direction.min(dim=0)[0] > 0.0
            is_max = torch.unsqueeze(is_max, dim=1)

            # apply non maximum suppression
            to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
            thin_edges[to_remove] = 0.0

        # thresholds

        if low_threshold is not None:
            low = thin_edges > low_threshold

            if high_threshold is not None:
                high = thin_edges > high_threshold
                # get black/gray/white only
                thin_edges = low * 0.5 + high * 0.5

                if hysteresis:
                    # get weaks and check if they are high or not
                    weak = (thin_edges == 0.5) * 1
                    weak_is_high = (self.hysteresis(thin_edges) > 1) * weak
                    thin_edges = high * 1 + weak_is_high * 1
            else:
                thin_edges = low * 1


        return  thin_edges


class EdgeDetector(torch.nn.Module):
    # https://github.com/zhaoyuzhi/PyTorch-Sobel/blob/main/pytorch-sobel.py

    def __init__(self, type="LAPLACIAN"):
        super(EdgeDetector, self).__init__()
        print(f"using {type}")
        laplacian_kernel_detail = [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]

        ]
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]

        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        laplacian_kernel_detail = torch.FloatTensor(laplacian_kernel_detail).unsqueeze(0).unsqueeze(0)

        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)
        self.laplac = nn.Parameter(data=laplacian_kernel_detail, requires_grad=False)
        self.canny = CannyFilter()
        self.type = type

    def normalize(self, x):
        x_min = torch.min(x)
        x_max = torch.max(x)
        return torch.div(x - x_min, x_max - x_min)

    def get_gray(self, x):
        '''
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):
        if x.shape[1] == 3:
            x = self.get_gray(x)
        if self.type == "BOTH":
            x_v = F.conv2d(x, self.weight_v, padding=1)
            x_h = F.conv2d(x, self.weight_h, padding=1)
            sobel = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2))
            x = F.conv2d(sobel, self.laplac, padding=1)
        elif self.type == "LAPLACIAN":
            x = F.conv2d(x, self.laplac, padding=1)
        elif self.type == "CannyFilter":
            x = self.canny(x)
        else:
            x_v = F.conv2d(x, self.weight_v, padding=1)
            x_h = F.conv2d(x, self.weight_h, padding=1)
            x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)
        x_norm = self.normalize(x)
        return x_norm.repeat(1, 3, 1, 1), x_norm


class SubBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.leaky = nn.LeakyReLU(negative_slope=.2)
        self.dconv_1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dconv_2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dconv_3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.dconv_4 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=1, stride=1)
        self.dconv_5 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=1, stride=1)
        self.dconv_6 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=1)

    def forward(self, x_1, x_2, x_3):
        im_h = x_1.shape[2]
        im_w = x_1.shape[3]

        bz = x_1.shape[0]

        a1 = self.leaky(self.dconv_1(x_1, output_size=torch.Size([bz, 64, im_h, im_w])))
        b1 = self.leaky(self.dconv_2(x_2, output_size=torch.Size([bz, 64, im_h, im_w])))
        c1 = self.leaky(self.dconv_3(x_3, output_size=torch.Size([bz, 64, im_h, im_w])))

        sum = torch.concat([a1, b1, c1], dim=1)

        x_1 = self.leaky(self.dconv_4(torch.concat([sum, x_1], dim=1)))
        x_2 = self.leaky(self.dconv_5(torch.concat([sum, x_2], dim=1)))
        x_3 = self.leaky(self.dconv_6(torch.concat([sum, x_3], dim=1)))

        return x_1, x_2, x_3


class Block(nn.Module):
    def __init__(self, num_sublocks=3):
        super().__init__()
        self.leaky = nn.LeakyReLU(negative_slope=.2)
        self.sub_blocks = nn.ModuleList()
        for i in range(num_sublocks):
            self.sub_blocks.append(
                SubBlock()
            )
        self.conv_block_out = nn.ConvTranspose2d(in_channels=192, out_channels=64, kernel_size=3, padding=1)

    def forward(self, x_1, x_2, x_3):
        for i in range(len(self.sub_blocks)):
            x_1, x_2, x_3 = self.sub_blocks[i](x_1, x_2, x_3)
        return self.leaky(self.conv_block_out(torch.concat([x_1, x_2, x_3], dim=1)))


class Dense(nn.Module):
    def __init__(self, num_blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(Block())

    def forward(self, x_frames):
        x = x_frames.clone()
        for i in range(len(self.blocks)):
            x_1 = x_2 = x_3 = x.clone()
            x += self.blocks[i](x_1, x_2, x_3)
        return x


# noise mask
class Mask(nn.Module):
    def __init__(self):
        super().__init__()
        self.leaky = nn.LeakyReLU(negative_slope=.2)
        self.dconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.dconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.dconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        bz = input.shape[0]

        im_h = input.shape[2]
        im_w = input.shape[3]

        x = self.leaky(self.dconv1(input, output_size=torch.Size([bz, 64, im_h, im_w])))
        x = self.leaky(self.dconv2(x, output_size=torch.Size([bz, 128, im_h, im_w])))
        x = self.leaky(self.dconv3(x, output_size=torch.Size([bz, 256, im_h, im_w])))
        x = self.sigmoid(x)
        return x


class EdgeEnhancement(nn.Module):
    def __init__(self, edge_enhancement_type="LAPLACIAN"):
        super(EdgeEnhancement, self).__init__()
        self.edge_detector = EdgeDetector(edge_enhancement_type)

        self.leaky = nn.LeakyReLU(negative_slope=.2)

        self.encoder_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1,padding='same')
        self.encoder_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,padding='same')
        self.encoder_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2,padding=1)
        self.encoder_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,padding='same')
        self.encoder_5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2,padding=1)
        self.encoder_6 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1,padding='same')

        # intermediate conv layer.
        self.conv_tranpose_interm = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            self.leaky
        )

        self.dense = Dense()
        # noise mask
        self.mask = Mask()

        self.conv_tranpose_interm_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            self.leaky
        )

        # upscaling ...
        self.upscale_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            self.leaky
        )
        # upscaling ...
        self.upscale_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            self.leaky
        )

        self.conv_final = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)

    def encoder(self, frames):
        x = self.leaky(self.encoder_1(frames))
        # (city,64,520,720)
        x = self.leaky(self.encoder_2(x))
        # (city,64,520,720)
        x = self.leaky(self.encoder_3(x))
        # (city,128,259,359)
        x = self.leaky(self.encoder_4(x))
        x = self.leaky(self.encoder_5(x))
        x = self.leaky(self.encoder_6(x))
        return x

    def forward(self, lr_data, device=None):
        if self.training:
            hr_frames, enhanced_edges = self.forward_sequence(lr_data)
        else:
            hr_frames, enhanced_edges = self.infer_sequence(lr_data, device)

        return hr_frames, enhanced_edges

    def infer_sequence(self, lr_data, device="cuda:0"):
        """
            Parameters:
                :param lr_data: torch.FloatTensor in shape tchw
                :param device: torch.device

                :return hr_seq: uint8 np.ndarray in shape tchw
        """

        # set params
        tot_frm, c, h, w = lr_data.size()

        # forward
        hr_seq = []
        edge_seq = []

        with torch.no_grad():
            for i in range(tot_frm):
                lr_curr = lr_data[i: i + 1, ...].to(device)
                hr_curr, enhanced_edge = self.forward_sequence(lr_curr.cuda().float())
                hr_frm = hr_curr.squeeze(0).cpu().numpy()
                enhanced_edge = enhanced_edge.squeeze(0).cpu().numpy()
                hr_seq.append(float32_to_uint8(hr_frm))
                edge_seq.append(enhanced_edge)

        return np.stack(hr_seq).transpose(0, 2, 3, 1), \
            np.stack(edge_seq).transpose(0, 2, 3, 1)  # thwc

    def forward_sequence(self, frames):
        # extract edges using LAPLACIAN/SOBEL/BOTH
        #[0,1]
        frames = self.edge_detector.normalize(frames)
        extracted_edges_repeat, edges_gray = self.edge_detector(frames)
        # feature learning
        residual_x = self.encoder(extracted_edges_repeat)
        # DENSE BLOCK + CON
        x_decoder = self.dense(residual_x)
        x_decoder = self.conv_tranpose_interm(x_decoder)
        # MASK
        x_mask = self.mask(residual_x)
        # maths
        x_frame = x_mask * x_decoder + x_decoder
        # x_frame = x_mask
        x_frame = self.conv_tranpose_interm_2(x_frame)
        # upscale
        x_frame = self.upscale_1(x_frame)
        x_frame = self.upscale_2(x_frame)
        enhanced_edges = self.conv_final(x_frame)

        enhanced_edges = self.edge_detector.normalize(enhanced_edges - extracted_edges_repeat)
        x_super_r = self.edge_detector.normalize(enhanced_edges + frames)
        return x_super_r, enhanced_edges


"""
FUNCTIONS USEFUL FOR DEBUGGING / VIEWING OUTPUTS WHILE TRAINING
I DID NOT WRITE THESE....
"""
def display_frame(frame):
    image_array = np.squeeze(frame.clone().detach().cpu().numpy())
    image_array = np.transpose(image_array, (1, 2, 0))
    image_array = (image_array * 255).astype(np.uint8)
    cv2.imwrite('/root/vsr/mine.jpg', cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))


def display_edges(edges):
    image_array = edges.clone().detach().cpu().numpy()
    image_array = np.transpose(image_array, (1, 2, 0))
    image_array = (image_array * 255).astype(np.uint8)
    cv2.imwrite('/root/vsr/my_edges_laplacian_base.jpg', image_array)


if __name__ == '__main__':
    # torch.cuda.empty_cache()
    # edge_enhancer = EdgeEnhancement(patch_size=32,edge_enhancement_type= "LAPLACIAN").cuda()
    # x = torch.rand(city, 3, 720, 1280).cuda()
    # out = edge_enhancer(x)
    # print(out[0].shape)
    # opt = {}
    # opt["scale"] = 4
    # opt["device"] = "cuda"
    # opt["dist"] = "false"
    # opt["is_train"] = "false"
    model = EdgeEnhancement(edge_enhancement_type="CannyFilter")

    # Move the model to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters())

    opt.zero_grad()
    model.train()

    criterion = torch.nn.MSELoss(reduction='mean').cuda()
    in_ = torch.rand(5, 3, 96, 96).to(device)
    # with torch.autograd.set_detect_anomaly(True):
    start_time = time.time()

    out, edges = model(in_)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time: forward", execution_time, "seconds")

    trgt = torch.rand(5, 3, 96, 96).to(device)
    loss = criterion(edges, trgt)
    print(loss.item())

    start_time = time.time()
    loss.backward()
    opt.step()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time: backprop", execution_time, "seconds")

import torch
from torch import nn
from enum import Enum
import torch.nn.functional as F
import numpy as np
import functools, operator
import unfoldNd
import time
import cv2

def float32_to_uint8(inputs):
    """ Convert np.float32 array to np.uint8

        Parameters:
            :param input: np.float32, (NT)CHW, [0, 1]
            :return: np.uint8, (NT)CHW, [0, 255]
    """
    return np.uint8(np.clip(np.round(inputs * 255), 0, 255))


class EdgeExtraction(Enum):
    LAPLACIAN = 1
    SOBEL = 2
    BOTH = 3

from torch import nn
import torch.nn.functional as F
class EdgeDetector(torch.nn.Module):
#https://github.com/zhaoyuzhi/PyTorch-Sobel/blob/main/pytorch-sobel.py

    def __init__(self,type="LAPLACIAN"):
        super(EdgeDetector, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        laplacian_kernel_detail = [
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]

                                 ]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        laplacian_kernel_detail = torch.FloatTensor(laplacian_kernel_detail).unsqueeze(0).unsqueeze(0)

        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)
        self.laplac = nn.Parameter(data=laplacian_kernel_detail, requires_grad=False)
        self.type = type
    def normalize(self,x):
        x_min = torch.min(x)
        x_max = torch.max(x)
        return torch.div(x - x_min, x_max - x_min)
    def get_gray(self,x):
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
            x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)
            x = x + F.conv2d(x,self.laplac,padding=1)
        elif self.type == "LAPLACIAN":
            x = F.conv2d(x,self.laplac,padding=1)
        else:
            x_v = F.conv2d(x, self.weight_v, padding=1)
            x_h = F.conv2d(x, self.weight_h, padding=1)
            x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)
        x_norm = self.normalize(x)
        return x_norm.repeat(1,3,1,1),x_norm

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
            x += self.blocks[i](x_1,x_2,x_3)
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


class Conv2dSamePadding(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(functools.reduce(operator.__add__,
                                                         [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in
                                                          self.kernel_size[::-1]]))

    def forward(self, input):
        return self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)


class EdgeEnhancement(nn.Module):
    def __init__(self, edge_enhancement_type="LAPLACIAN"):
        super(EdgeEnhancement, self).__init__()
        self.edge_detector = EdgeDetector(edge_enhancement_type)

        self.leaky = nn.LeakyReLU(negative_slope=.2)

        self.encoder_1 = Conv2dSamePadding(in_channels=3, out_channels=64, kernel_size=3, stride=1)
        self.encoder_2 = Conv2dSamePadding(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.encoder_3 = Conv2dSamePadding(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.encoder_4 = Conv2dSamePadding(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.encoder_5 = Conv2dSamePadding(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.encoder_6 = Conv2dSamePadding(in_channels=256, out_channels=64, kernel_size=3, stride=1)

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

        self.conv_final = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)

    def encoder(self, frames):
        x = self.leaky(self.encoder_1(frames))
        # (1,64,520,720)
        x = self.leaky(self.encoder_2(x))
        # (1,64,520,720)
        x = self.leaky(self.encoder_3(x))
        # (1,128,259,359)
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
        extracted_edges_repeat,edges_gray = self.edge_detector(frames)
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

        enhaced_edges = self.edge_detector.normalize(enhanced_edges)
        enhaced_edges = enhaced_edges - edges_gray
        x_super_r = self.edge_detector.normalize(enhaced_edges.repeat(1,3,1,1) + frames)
        # super resolute images, edges
        return x_super_r, enhaced_edges


def display_frame(frame):
    # Reshape the array to (96, 96)
    image_array = np.squeeze(frame.clone().detach().cpu().numpy())  # Remove the singleton dimension
    image_array = np.transpose(image_array, (1, 2, 0))  # Transpose to (96, 96, 1)

    # Convert to uint8 (assuming values are in the range 0-1)
    image_array = (image_array * 255).astype(np.uint8)

    # Save the array as a grayscale image using OpenCV# Save the array as an RGB image using OpenCV
    cv2.imwrite('/root/vsr/rgb_image_ee.jpg', cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    print("Image saved successfully.")

def display_edges(edges):
    # Reshape the array to (96, 96)
    # ee_module = EdgeDetector()
    # edges = ee_module(frame)
    image_array = edges.clone().detach().cpu().numpy() # Remove the singleton dimension
    image_array = np.transpose(image_array, (1, 2, 0))  # Transpose to (96, 96, 1)

    # Convert to uint8 (assuming values are in the range 0-1)
    image_array = (image_array * 255).astype(np.uint8)

    # Save the array as a grayscale image using OpenCV# Save the array as an RGB image using OpenCV
    cv2.imwrite('/root/vsr/grayscale_image.jpg',image_array)
    print("Image saved successfully.")

if __name__ == '__main__':
    # torch.cuda.empty_cache()
    # edge_enhancer = EdgeEnhancement(patch_size=32,edge_enhancement_type= "LAPLACIAN").cuda()
    # x = torch.rand(1, 3, 720, 1280).cuda()
    # out = edge_enhancer(x)
    # print(out[0].shape)
    # opt = {}
    # opt["scale"] = 4
    # opt["device"] = "cuda"
    # opt["dist"] = "false"
    # opt["is_train"] = "false"
    model = EdgeEnhancement()


    # Move the model to GPU
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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

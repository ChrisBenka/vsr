import torch
from torch import nn
from enum import Enum
import torch.nn.functional as F
import numpy as np
import functools, operator


class EdgeExtraction(Enum):
    LAPLACIAN = 1
    SOBEL = 2
    BOTH = 3


def edge_extractor(frames,device,edge_extraction_type=EdgeExtraction.LAPLACIAN):
    laplacian = torch.tensor(
        [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ]
    ).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    sobel_np = np.array(
        [
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ]
    )
    if edge_extraction_type == EdgeExtraction.LAPLACIAN:
        res = F.conv2d(frames, laplacian.to(device), stride=1, padding=1).repeat(1, 3, 1, 1)
        return res
    elif edge_extraction_type == EdgeExtraction.SOBEL:
        sobel_x_filter = torch.tensor(sobel_np).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        sobel_y_filter = torch.tensor(sobel_np.T).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        sobel_x_edges = F.conv2d(frames, sobel_x_filter, stride=1, padding=1)
        sobel_y_edges = F.conv2d(frames, sobel_y_filter, stride=1, padding=1)
        res = torch.sqrt(sobel_x_edges ** 2 + sobel_y_edges ** 2).repeat(1, 3, 1, 1)
    else:
        laplacian = F.conv2d(frames, laplacian.cpu(), stride=1, padding=1)
        sobel_x_filter = torch.tensor(sobel_np).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        sobel_y_filter = torch.tensor(sobel_np.T).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        sobel_x_edges = F.conv2d(frames, sobel_x_filter, stride=1, padding=1)
        sobel_y_edges = F.conv2d(frames, sobel_y_filter, stride=1, padding=1)
        return (torch.sqrt(sobel_x_edges ** 2 + sobel_y_edges ** 2) + laplacian).repeat(1, 3, 1, 1)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.leaky = nn.LeakyReLU(negative_slope=.2)
        self.dconv_1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dconv_2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dconv_3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.dconv_4 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=1, stride=1)
        self.dconv_5 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=1, stride=1)
        self.dconv_6 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=1)

        self.conv_out = nn.ConvTranspose2d(in_channels=192, out_channels=64, kernel_size=3, padding=1)

    def forward(self, input):
        x_1 = x_2 = x_3 = input.clone()
        bz = x_1.shape[0]
        img_h = x_1.shape[2]
        img_w = x_1.shape[3]

        a1 = self.leaky(self.dconv_1(x_1, output_size=torch.Size([bz, 64, img_h, img_w])))
        b1 = self.leaky(self.dconv_2(x_2, output_size=torch.Size([bz, 64, img_h, img_w])))
        c1 = self.leaky(self.dconv_3(x_3, output_size=torch.Size([bz, 64, img_h, img_w])))

        sum = torch.concat([a1, b1, c1], dim=1)

        x_1 = self.leaky(self.dconv_4(torch.concat([sum, a1], dim=1)))
        x_2 = self.leaky(self.dconv_5(torch.concat([sum, b1], dim=1)))
        x_3 = self.leaky(self.dconv_6(torch.concat([sum, c1], dim=1)))

        return self.leaky(self.conv_out(torch.concat([x_1, x_2, x_3], dim=1)))


class Dense(nn.Module):
    def __init__(self, num_blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(Block())

    def forward(self, x_frames):
        x = x_frames.clone()
        for i in range(len(self.blocks)):
            x += self.blocks[i](x)
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
    def __init__(self, edge_enhancement_type=EdgeExtraction.LAPLACIAN):
        super(EdgeEnhancement, self).__init__()

        if edge_enhancement_type == "LAPLACIAN":
            self.edge_enhancement_type = EdgeExtraction.LAPLACIAN
        elif edge_enhancement_type == "SOBEL":
            self.edge_enhancement_type == EdgeExtraction.SOBEL
        else:
            self.edge_enhancement_type == EdgeExtraction.BOTH


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
        self.mask = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            self.leaky,
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            self.leaky,
            nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

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
        # (1,64,520,720)
        x = self.leaky(self.encoder_2(x))
        # (1,64,520,720)
        x = self.leaky(self.encoder_3(x))
        # (1,128,259,359)
        x = self.leaky(self.encoder_4(x))
        x = self.leaky(self.encoder_5(x))
        x = self.leaky(self.encoder_6(x))
        return x

    def forward(self, frames):
        device = next(self.parameters()).device
        # extract edges using LAPLACIAN/SOBEL/BOTH
        extracted_edges = edge_extractor(frames,device,edge_extraction_type=self.edge_enhancement_type)
        # feature learning
        residual_x = self.encoder(extracted_edges)
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

        x_frame = self.conv_final(x_frame)

        x_super_r = x_frame + frames - extracted_edges
        # super resolute images, edges
        return x_super_r, frames - extracted_edges


if __name__ == '__main__':
    edge_enhancer = EdgeEnhancement().cuda()
    x_sr, edges = edge_enhancer(torch.randn(1, 3, 576, 720).cuda())
    print(x_sr.shape)

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


class Patchify(nn.Module):
    "src:https://mrinath.medium.com/vit-part-1-patchify-images-using-pytorch-unfold-716cd4fd4ef6"
    def __init__(self, patch_size=56):
        super().__init__()
        self.p = patch_size
        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x -> B c h w
        bs, c, h, w = x.shape

        x = self.unfold(x)
        # x -> B (c*p*p) L

        # Reshaping into the shape we want
        a = x.view(bs, c, self.p, self.p, -1).permute(0, 4, 1, 2, 3)
        # a -> ( B no.of patches c p p )
        return a

class UnPatchify(nn.Module):
    "src:https://mrinath.medium.com/vit-part-1-patchify-images-using-pytorch-unfold-716cd4fd4ef6"
    def __init__(self, patch_size, desired_shape=None):
        super().__init__()
        self.patch_size = patch_size
        self.fold = torch.nn.Fold(output_size=desired_shape,
                                  kernel_size=self.patch_size,
                                  stride=self.patch_size)

    def forward(self, x):
        # x -> B no.of patches c p p
        bs, num_patches, c, p, _ = x.shape

        # Reshaping into the shape we want for folding
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(bs, c * p * p, num_patches)

        # Folding to reconstruct the image
        reconstructed_image = self.fold(x)
        return reconstructed_image

class Block(nn.Module):
    def __init__(self,patch_size=32):
        super().__init__()
        self.leaky = nn.LeakyReLU(negative_slope=.2)
        self.dconv_1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dconv_2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dconv_3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.dconv_4 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=1, stride=1)
        self.dconv_5 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=1, stride=1)
        self.dconv_6 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=1)

        self.conv_out = nn.ConvTranspose2d(in_channels=192, out_channels=64, kernel_size=3, padding=1)
        self.patch_size = patch_size
        self.patchify = Patchify(patch_size=self.patch_size)

    def forward(self, input):
        input_patches = Patchify(patch_size=self.patch_size)(input)
        input_patches_flat = input_patches.flatten(start_dim=0,end_dim=1)
        # input_patches = input.flatten(start_dim=0,end_dim=1).unfold(0, 64, 64).unfold(1, 32, 32).unfold(2, 32, 32).flatten(start_dim=0,end_dim=2)
        # input_patches = input_patches.fold(output_size=)


        x_1 = x_2 = x_3 = input_patches_flat.clone()
        bz = x_1.shape[0]

        a1 = self.leaky(self.dconv_1(x_1, output_size=torch.Size([bz, 64, self.patch_size, self.patch_size])))
        b1 = self.leaky(self.dconv_2(x_2, output_size=torch.Size([bz, 64, self.patch_size, self.patch_size])))
        c1 = self.leaky(self.dconv_3(x_3, output_size=torch.Size([bz, 64, self.patch_size, self.patch_size])))

        sum = torch.concat([a1, b1, c1], dim=1)

        x_1 = self.leaky(self.dconv_4(torch.concat([sum, a1], dim=1)))
        x_2 = self.leaky(self.dconv_5(torch.concat([sum, b1], dim=1)))
        x_3 = self.leaky(self.dconv_6(torch.concat([sum, c1], dim=1)))

        out_patches =  self.leaky(self.conv_out(torch.concat([x_1, x_2, x_3], dim=1)))

        return UnPatchify(patch_size=self.patch_size, desired_shape=input.size()[2:])(
            out_patches.view(input_patches.shape)
        )


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


# noise mask
class Mask(nn.Module):
    def __init__(self,patch_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.leaky = nn.LeakyReLU(negative_slope=.2)
        self.dconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.dconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.dconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,input):
        input_patches = Patchify(patch_size=self.patch_size)(input)
        input_patches_flat = input_patches.flatten(start_dim=0,end_dim=1)
        bz = input_patches_flat.shape[0]
        x = self.leaky(self.dconv1(input_patches_flat, output_size=torch.Size([bz, 64, self.patch_size, self.patch_size])))
        x = self.leaky(self.dconv2(x, output_size=torch.Size([bz, 128, self.patch_size, self.patch_size])))
        x = self.leaky(self.dconv3(x, output_size=torch.Size([bz, 256, self.patch_size, self.patch_size])))
        x = self.sigmoid(x)
        return UnPatchify(patch_size=self.patch_size, desired_shape=input.size()[2:])(
            x.view(input_patches.shape[0],input_patches.shape[1],256,self.patch_size,self.patch_size)
        )
class Conv2dSamePadding(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(functools.reduce(operator.__add__,
                                                         [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in
                                                          self.kernel_size[::-1]]))

    def forward(self, input):
        return self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)


class EdgeEnhancement(nn.Module):
    def __init__(self, edge_enhancement_type=EdgeExtraction.LAPLACIAN,patch_size=32):
        super(EdgeEnhancement, self).__init__()

        if edge_enhancement_type == "LAPLACIAN":
            self.edge_enhancement_type = EdgeExtraction.LAPLACIAN
        elif edge_enhancement_type == "SOBEL":
            self.edge_enhancement_type == EdgeExtraction.SOBEL
        else:
            self.edge_enhancement_type == EdgeExtraction.BOTH

        self.patch_size = patch_size
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
        self.mask = Mask(patch_size=self.patch_size)

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
        return x_super_r


if __name__ == '__main__':
    # torch.cuda.empty_cache()
    # edge_enhancer = EdgeEnhancement(patch_size=32,edge_enhancement_type= "LAPLACIAN").cuda()
    # x = torch.rand(1, 3, 720, 1280).cuda()
    # out = edge_enhancer(x)
    # print(out[0].shape)

    model =  EdgeEnhancement(patch_size=32,edge_enhancement_type= "LAPLACIAN")

    # Check memory usage before moving to GPU
    torch.cuda.empty_cache()  # Ensure GPU memory is freed
    torch.cuda.synchronize()  # Synchronize CUDA kernel launches
    mem_before = torch.cuda.memory_allocated()

    # Move the model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Check memory usage after moving to GPU
    torch.cuda.synchronize()  # Synchronize CUDA kernel launches
    mem_after = torch.cuda.memory_allocated()

    # Calculate memory usage difference
    mem_used = mem_after - mem_before
    print(mem_before)
    print(f"Memory used by the model on GPU: {mem_used / (1024 ** 2)} MB")

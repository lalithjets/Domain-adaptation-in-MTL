import math
import torch
import torch.nn as nn

# 1D gaussian kernel 
def get_gaussian_filter_1D(kernel_sizex=3, kernel_sizey=1, sigma=2, channels=3):
    kernel_size = max(kernel_sizex, kernel_sizey)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()

    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    xy_grid = torch.sum((xy_grid[:kernel_size,:kernel_size,:] - mean)**2., dim=-1)

    # Calculate the 1-dimensional gaussian kernel
    gaussian_kernel = (1./((math.sqrt(2.*math.pi)*sigma))) * \
                        torch.exp(-1* (xy_grid[int(kernel_size/2)]) / (2*variance))

    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1)

    padding = 1 if kernel_size==3 else 2 if kernel_size == 5 else 0
    gaussian_filter = nn.Conv1d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels,
                                bias=False, padding=padding)
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False 
    return gaussian_filter

def get_laplaceOfGaussian_filter_1D(kernel_size=3, sigma=2, channels=3):
    
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1)/2.

    used_sigma = sigma
    # Calculate the 2-dimensional gaussian kernel which is
    log_kernel = (-1./(math.pi*(used_sigma**4))) \
                  * (1-(torch.sum((xy_grid[int(kernel_size/2)] - mean)**2., dim=-1) / (2*(used_sigma**2)))) \
                  * torch.exp(-torch.sum((xy_grid[int(kernel_size/2)] - mean)**2., dim=-1) / (2*(used_sigma**2)))
    
    # Make sure sum of values in gaussian kernel equals 1.
    log_kernel = log_kernel / torch.sum(log_kernel)
    log_kernel = log_kernel.view(1, 1, kernel_size)
    log_kernel = log_kernel.repeat(channels, 1, 1)

    padding = 1 if kernel_size==3 else 2 if kernel_size == 5 else 0
    log_filter = nn.Conv1d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels,
                                bias=False, padding=padding)
    log_filter.weight.data = log_kernel
    log_filter.weight.requires_grad = False
    return log_filter



#pragma once

#include <cuda_runtime.h>
#include "config.h"

__device__ __kde_float epanechnikov_2d_kde(
    __float x0, __float y0,                     // Evaluation point
    const __float* x, const __float* y,         // Input points (x0, x1, ..., xn), (y0, y1, ..., yn)
    int num_points,                             // Number of points
    __kde_float bandwidth                       // Bandwidth parameter
);

__global__ void histogram_2d(
    const float* x,          // Input x coordinates
    const float* y,          // Input y coordinates
    int num_points,          // Number of data points
    __kde_float* output,           // Output grid (grid_size x grid_size)
    int grid_size           // Grid dimension
);

__device__ __kde_float compute_mi(const __kde_float* pxy_flat, int grid_size);
__global__ void mi_kernel(const __kde_float* pxy_flat, __kde_float* output, int grid_size);
__global__ void normalize_data(__float* data, int num_points, __float min, __float max);
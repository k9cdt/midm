#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <cuda_runtime.h>

#include "config.h"

__global__ void histogram_2d(
    const float* x,          // Input x coordinates
    const float* y,          // Input y coordinates
    int num_points,          // Number of data points
    __kde_float* output,           // Output grid (grid_size x grid_size)
    int grid_size           // Grid dimension
) {
    // Calculate grid coordinates for this data point
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    // Get normalized coordinates [0,1]
    float x_norm = x[idx];
    float y_norm = y[idx];

    // Convert to grid indices
    int i = static_cast<int>(x_norm * (grid_size - 1));
    int j = static_cast<int>(y_norm * (grid_size - 1));

    // Clamp to valid range (shouldn't be needed if input is properly normalized)
    i = max(0, min(grid_size - 1, i));
    j = max(0, min(grid_size - 1, j));

    // Atomic increment of the corresponding bin
    atomicAdd(&output[i * grid_size + j], 1.0f/num_points);
}

__device__ __kde_float compute_mi(const __kde_float* pxy_flat, int grid_size) {
    // Allocate and initialize
    __kde_float* px = new __kde_float[grid_size]();  // Zero-initialized
    __kde_float* py = new __kde_float[grid_size]();
    __kde_float total_sum = 0.0;

    // Single-pass computation of marginals and total sum
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            const __kde_float val = pxy_flat[i * grid_size + j];
            px[i] += val;
            py[j] += val;
            total_sum += val;
        }
    }

    // Normalization factor
    const __kde_float norm_factor = 1.0 / total_sum;
    const __kde_float epsilon = 1e-10;  

    // Compute MI and entropy
    __kde_float mi = 0.0, entropy = 0.0;
    for (int i = 0; i < grid_size; ++i) {
        const __kde_float px_norm = px[i] * norm_factor;
        for (int j = 0; j < grid_size; ++j) {
            const __kde_float p = pxy_flat[i * grid_size + j] * norm_factor;
            if (p > epsilon) {
                const __kde_float log_p = log(p);
                entropy -= p * log_p;
                
                const __kde_float py_norm = py[j] * norm_factor;
                if (px_norm > epsilon && py_norm > epsilon) {
                    mi += p * (log_p - log(px_norm * py_norm));
                }
            }
        }
    }

    delete[] px;
    delete[] py;

    // Return normalized MI distance (clamped to [0,1])
    double mi_distance = (entropy > epsilon) ? (1.0 - mi / entropy) : 0.0;
    return fmax(0.0, fmin(1.0, mi_distance));
}

__global__ void mi_kernel(const __kde_float* pxy_flat, __kde_float* output, int grid_size) {
    if (threadIdx.x == 0) {
        *output = compute_mi(pxy_flat, grid_size);
//        printf("Mutual Information: %f\n", *output);
    }
}

__global__ void normalize_data(__float* data, int num_points, __float min, __float max){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    // Calculate scale factor (with safety check)
    __float scale = (max > min) ? 1.0f /(max - min) : 1.0f;
    
    // Normalize in place
    data[idx] = (data[idx] - min) * scale;
}
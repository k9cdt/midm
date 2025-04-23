#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>

#include <cuda_runtime.h>

#include "config.h"
#include "kde.h"
#include "kernel.h"

DistanceMatrix::DistanceMatrix(const Colvar* cv) {

    this->_n = cv->cols() - 1;
    this->_cv = cv;

    int rows = this->_cv->rows();

    // Allocate memory for the distance matrix, on host and device
    this->_mat = (__kde_float*)calloc(this->_n * this->_n, sizeof(__kde_float*));
    cudaMalloc((void**)&this->_mat_device, sizeof(__kde_float) * this->_n * this->_n);

    // Copy timeseries data to device
    this->_ts = (__float**)calloc(this->_n, sizeof(__float*));
    int nthreads = 1024;
    int nblocks = (rows + nthreads - 1) / nthreads;
    for (int i = 0; i < this->_n; i++) {
        cudaMalloc((void**)&this->_ts[i], sizeof(__float) * rows);
        cudaMemcpy(this->_ts[i], this->_cv->get_cv(i), sizeof(__float) * rows, cudaMemcpyHostToDevice);
        normalize_data<<<nblocks, nthreads>>>(this->_ts[i], rows, this->_cv->min(i), this->_cv->max(i));
    }

    printf("DistanceMatrix: Allocated %d x %d matrix\n", this->_n, this->_n);
}

DistanceMatrix::~DistanceMatrix() {
    free(this->_mat);
    cudaFree(this->_mat_device);
    for (int i = 0; i < this->_n; i++) {
        cudaFree(this->_ts[i]);
    }
}

void DistanceMatrix::compute(int bins) {

    int rows = this->_cv->rows();

    int n_comps = this->_n * (this->_n - 1) / 2;
    int count = 0;

    cudaEvent_t start, now;
    cudaEventCreate(&start);
    cudaEventCreate(&now);
    cudaEventRecord(start);

    // Allocate memory for the output array
    __kde_float* pxy;
    cudaMalloc((void**)&pxy, sizeof(__kde_float) * bins * bins);

    for (int i = 0; i < this->_n; i++) {
        for (int j = 0; j <= i; j++){

            // Fill pxy with zeros
            cudaMemset(pxy, 0, sizeof(__kde_float) * bins * bins);

            // Compute KDE
            int nthreads = 256;
            int nblocks = (rows + nthreads - 1) / nthreads;
            histogram_2d<<<nblocks, nthreads>>>(this->_ts[i], this->_ts[j], rows, pxy, bins);

            // Compute distance
            int idx = i * this->_n + j;
            mi_kernel<<<1, 1>>>(pxy, &(this->_mat_device[idx]), bins);

            count++;
            if (count % 100 == 0) {
                float elapsedTime;
                cudaEventRecord(now);
                cudaEventSynchronize(now);
                cudaEventElapsedTime(&elapsedTime, start, now);

                float it_per_s = count / (elapsedTime / 1000);
                float remainingTime = (n_comps - count) * elapsedTime / count;
                printf("DistanceMatrix: %d / %d.", count, n_comps);
                printf(" %.2f it/s.", it_per_s);
                printf(" [%dm%ds left].\n", 
                    static_cast<int>(std::round(remainingTime/60000)), 
                    static_cast<int>(std::round(remainingTime/1000)) % 60
                );
            }
        }
    }

    cudaFree(pxy);
    cudaEventDestroy(start);
    cudaEventDestroy(now);

    // Copy the distance matrix back to host
    cudaMemcpy(this->_mat, this->_mat_device, sizeof(__kde_float) * this->_n * this->_n, cudaMemcpyDeviceToHost);

    // Fill the upper triangle of the matrix
    for (int i = 0; i < this->_n; i++) {
        for (int j = i + 1; j < this->_n; j++) {
            this->_mat[i * this->_n + j] = this->_mat[j * this->_n + i];
        }
    }
}

void DistanceMatrix::save(const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file %s for writing\n", filename);
        return;
    }
    std::fwrite(this->_mat, sizeof(__kde_float), this->_n * this->_n, fp);
    fclose(fp);
    printf("DistanceMatrix: Saved distance matrix to %s\n", filename);
}

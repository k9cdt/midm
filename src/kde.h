#pragma once

#include <cuda_runtime.h>
#include "config.h"
#include "colvar.h"

class DistanceMatrix{

private:
    __kde_float* _mat;
    __kde_float* _mat_device;
    int _n;                 // number of CVs
    const Colvar* _cv;
    
    __float** _ts;

public:
    DistanceMatrix(const Colvar* cv);
    ~DistanceMatrix();
    void compute(int bins);
    void save(const char* filename);

};


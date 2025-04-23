#pragma once

#include <cuda_runtime.h>
#include <omp.h>

#include "config.h"

class Colvar{

private:

    int _rows, _cols;
    __float** _colvar;

    __float *_min, *_max;

public:
    Colvar(int argc, char** argv);
    ~Colvar();
    int rows() const { return this->_rows; }
    int cols() const { return this->_cols; }
    __float min(int i) const { return this->_min[i+1]; }
    __float max(int i) const { return this->_max[i+1]; }
    __float* get_cv(int i) const;
};
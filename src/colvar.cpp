#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#include "config.h"
#include "colvar.h"

Colvar::Colvar(int argc, char** argv) {

    this->_rows = 0;
    this->_cols = -1;
    int n_trajs = argc - 1;

    // First pass: determine total rows and consistent columns
    int* local_rows = (int*)calloc(n_trajs, sizeof(int));
    for(int i = 0; i < n_trajs; i++) {
        FILE* file = fopen(argv[i+1], "r");
        if (!file) {
            fprintf(stderr, "Error: Could not open file %s\n", argv[i+1]);
            exit(1);
        }
        printf("Reading file %s\n", argv[i+1]);

        // Count rows and columns for this file
        char line[10000];
        while (fgets(line, sizeof(line), file)) {
            if (line[0] == '\n' || line[0] == '#') continue;
            
            local_rows[i]++;
            
            if (this->_cols == -1) {
                char* token;
                char* rest = line;
                int cols = 0;
                while ((token = strtok_r(rest, " \t\n\r", &rest))) cols++;
                this->_cols = cols;
            }
        }
        fclose(file);
        this->_rows += local_rows[i];
    }

    if (this->_rows == 0) {
        fprintf(stderr, "Error: No data found in any file\n");
        exit(1);
    }
    if (this->_cols <= 0) {
        fprintf(stderr, "Error: No columns found in any file\n");
        exit(1);
    }

    // Allocate the final transposed array directly
    this->_colvar = (__float**)calloc(this->_cols, sizeof(__float*));
    for (int i = 0; i < this->_cols; i++) {
        this->_colvar[i] = (__float*)calloc(this->_rows, sizeof(__float));
    }

    // Second pass: read data directly into transposed positions
    int global_row = 0;
    for (int i = 0; i < n_trajs; i++) {
        FILE* file = fopen(argv[i+1], "r");
        if (!file) {
            fprintf(stderr, "Error: Could not open file %s\n", argv[i+1]);
            exit(1);
        }

        char line[20000];
        int local_row = 0;
        while (fgets(line, sizeof(line), file) && local_row < local_rows[i]) {
            if (line[0] == '\n' || line[0] == '#') continue;

            char* token;
            char* rest = line;
            int col = 0;
            while ((token = strtok_r(rest, " \t\n\r", &rest)) && col < this->_cols) {
                this->_colvar[col][global_row] = atof(token);
                col++;
            }
            local_row++;
            global_row++;
        }
        fclose(file);
    }

    free(local_rows);

    // Book-keeping
    this->_min = (__float*)calloc(this->_cols, sizeof(__float));
    this->_max = (__float*)calloc(this->_cols, sizeof(__float));
    for (int i = 0; i < this->_cols; i++) {
        this->_min[i] = this->_colvar[i][0];
        this->_max[i] = this->_colvar[i][0];
        for (int j = 1; j < this->_rows; j++) {
            if (this->_colvar[i][j] < this->_min[i]) this->_min[i] = this->_colvar[i][j];
            if (this->_colvar[i][j] > this->_max[i]) this->_max[i] = this->_colvar[i][j];
        }
    }
}

Colvar::~Colvar() {
    for (int i = 0; i < this->_cols; i++) {
        free(this->_colvar[i]);
    }
    free(this->_colvar);
    free(this->_min);
    free(this->_max);
}

__device__ __host__ __float* Colvar::get_cv(int i) const {
    return this->_colvar[i+1];
}
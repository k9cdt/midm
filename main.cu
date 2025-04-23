#include <cstdio>
#include <cuda_runtime.h>

#include "config.h"
#include "colvar.h"
#include "kde.h"

int main(int argc, char** argv){

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    Colvar* cv = new Colvar(argc, argv);
    printf("Loaded %d rows and %d columns from %d files.\n", cv->rows(), cv->cols(), argc-1);

    DistanceMatrix* dm = new DistanceMatrix(cv);
    dm->compute(64);
    dm->save("output.matrix");

    delete dm;
    delete cv;
}
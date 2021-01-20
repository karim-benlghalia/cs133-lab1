// Header inclusions, if any...
#include <omp.h>
#include "lib/gemm.h"
#include <cstring>
// Using declarations, if any...
#define blocking_size 64
void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ])
{
 omp_set_num_threads(32);
	int i, j, k;
    int b_i, b_j, bk;
    int i_blocks = kI/blocking_size;
    int j_blocks = kJ/blocking_size;

    #pragma omp parallel for private(i, j, k, b_i, b_j)
    
    for(i = 0; i < i_blocks; i++) {

        for(j = 0; j < j_blocks; j++) {

           float temp[blocking_size][blocking_size];
           memset(temp,0,sizeof(float)*blocking_size*blocking_size);

            for(k = 0; k < kK; k++) {
                for(b_i = 0; b_i < blocking_size; b_i++) {
                    for(b_j = 0; b_j < blocking_size; b_j++) {
                        temp[b_i][b_j] += a[i*blocking_size + b_i][k]*b[k][j*blocking_size + b_j];
                    }
                }
            }

            for(b_i = 0; b_i < blocking_size; b_i++) {
                for(b_j = 0; b_j < blocking_size; b_j++) {
                    c[i*blocking_size + b_i][j*blocking_size + b_j] = temp[b_i][b_j];
                }
            }
        }
}
}

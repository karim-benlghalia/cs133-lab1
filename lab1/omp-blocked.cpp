// Header inclusions, if any...
#include <omp.h>
#include "lib/gemm.h"
#include <cstring>
// Using declarations, if any...
 #define blocking_size 32
void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ]) {
omp_set_num_threads(32);
	int i, j, k;
    int bi, bj, bk;
    int i_blocks = kI/blocking_size;
    int j_blocks = kJ/blocking_size;
     int k_blocks = kK/blocking_size;

    #pragma omp parallel  private(i, j, k, bi, bj) 
    {

    #pragma omp for schedule(static)
    for(i = 0; i < i_blocks; i++) {


        for(k = 0; k < j_blocks; k++) {

           float temp[blocking_size][blocking_size];
           memset(temp,0,sizeof(float)*blocking_size*blocking_size);

            for(j = 0; j < kJ; j++) {
                for(bi = 0; bi < blocking_size; bi++) {
                  int calc_bloc = i*blocking_size + bi;
                    for(bk = 0; bk < blocking_size; bk++) {
                        temp[bi][bk] = temp[bi][bk] + a[ calc_bloc ][j]*b[j][k*blocking_size + bk];
                         c[calc_bloc][k*blocking_size + bk] = temp[bi][bk];
                    }
                }
            }

            // for(bi = 0; bi < blocking_size; bi++) {
            //    int calc_bloc = i*blocking_size + bi;
            //     for(bk = 0; bk < blocking_size; bk++) {
            //         c[calc_bloc][k*blocking_size + bk] = temp[bi][bk];
            //     }
            // }
        }




    }
    }

}

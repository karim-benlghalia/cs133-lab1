#include <cstring>
#include "lib/gemm.h"
#include <omp.h>

// Using declarations, if any...

void GemmParallel(const float a[kI][kK], const float b[kK][kJ],
                  float c[kI][kJ]) {
  float temp;
  int i, j, k;
  
 int nCores=omp_get_max_threads();
 omp_set_num_threads(nCores);

 // #pragma omp parallel for 
   for (int i = 0; i < kI; ++i) {
    std::memset(c[i], 0, sizeof(float) * kJ);
  }

  #pragma omp parallel for private(j,k) schedule(static)
  for (i = 0; i < kI; ++i) {
    for (k = 0; k < kK; ++k) {
      temp=a[i][k];
      for (j = 0; j < kJ; ++j) {
        c[i][j] += temp*b[k][j];
      }
    }
  }
  
}

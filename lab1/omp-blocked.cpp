// Header inclusions, if any...
#include <omp.h>
#include "lib/gemm.h"
#include <cstring>
// Using declarations, if any...
#define blocking_size 32
void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ])
{
  
  int nCores = omp_get_max_threads();
  omp_set_num_threads(nCores);
  int i, j, k;
  int bi, bj, bk;
  int i_blocks = kI / blocking_size;
  int j_blocks = kJ / blocking_size;



  for (i = 0; i < i_blocks; i++)
  {
#pragma omp parallel for private(i, j, k, bi, bj) schedule(static, 16)
    for (j = 0; j < j_blocks; j++)
    {

      float temp[blocking_size][blocking_size];
      memset(temp, 0, sizeof(float) * blocking_size * blocking_size);

      for (k = 0; k < kK; k++)
      {
        for (bi = 0; bi < blocking_size; bi++)
        {
          for (bj = 0; bj < blocking_size; bj++)
          {
            temp[bi][bj] += a[i * blocking_size + bi][k] * b[k][j * blocking_size + bj];
          }
        }
      }

      for (bi = 0; bi < blocking_size; bi++)
      {
        for (bj = 0; bj < blocking_size; bj++)
        {
          c[i * blocking_size + bi][j * blocking_size + bj] = temp[bi][bj];
        }
      }
    }
  }
}

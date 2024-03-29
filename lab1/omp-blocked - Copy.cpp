// Header inclusions, if any...
#include <omp.h>
#include "lib/gemm.h"
#include <cstring>
// Using declarations, if any...
#define blocking_size 128
void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ])
{
  
  int nCores = omp_get_max_threads();
  omp_set_num_threads(nCores);
  int i, j, k;
  int b_i, b_j, b_k;
  int Iblock = kI / blocking_size;
  int Jblock = kJ / blocking_size;
  int Kblock = kK / blocking_size;


#pragma omp parallel for private(i)
  for (int i = 0; i < kI; ++i)
  {
    std::memset(c[i], 0, sizeof(float) * kJ);
  }

#pragma omp parallel for private(i, j, k, b_i, b_j, b_k) schedule(static, 16)
  for (i = 0; i < Iblock; i++)
  {

    for (j = 0; j < Jblock; j++)
    {
      float temp[blocking_size][blocking_size];
      memset(temp, 0, sizeof(float) * blocking_size * blocking_size);

      for (k = 0; k < kK; k++)
      {
        for (b_i = 0; b_i < blocking_size; b_i++)
        {
          int indexI = i * blocking_size + b_i;
          for (b_j = 0; b_j < blocking_size; b_j++)
          {
            int indexJ = j * blocking_size + b_j;
            temp[b_i][b_j] += a[indexI][k] * b[k][indexJ ];
             
          }
        }
      }

        for (b_i = 0; b_i < blocking_size; b_i++)
      {
         int indexI = i * blocking_size + b_i;
        for (b_j = 0; b_j < blocking_size; b_j++)
        {
          int indexJ = j * blocking_size + b_j;
          c[indexI][indexJ] = temp[b_i][b_j];
        }
      }
    }
  }
}

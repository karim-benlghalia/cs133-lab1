// Header inclusions, if any...
#include <omp.h>
#include "lib/gemm.h"
#include <cstring>
// Using declarations, if any...
#define blocking_size 64
void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ])
{
  int nCores=omp_get_max_threads();
 omp_set_num_threads(nCores);

  int i, j, k, i_block, j_block, k_block;
  float temp;
  //int num;
  //int id;

#pragma omp parallel private(k_block, j_block, i, k, j)

{
 
 #pragma omp for schedule(static)
  for (i_block = 0; i_block < kI; i_block += blocking_size)
  {
    for (j_block = 0; j_block < kJ; j_block += blocking_size)
    {
      float temp_buff[blocking_size][blocking_size] = {0};

      for (k_block = 0; k_block < kK; k_block += blocking_size)
      {
        for (i = i_block; i < i_block + blocking_size; i++)
        {
          for (k = k_block; k < k_block + blocking_size; k++)
          {
            temp = a[i][k];
            for (j = j_block; j < j_block + blocking_size; j++)
            {
              int indexI = i - i_block;
              int indexJ = j - j_block;
              temp_buff[indexI][indexJ] += temp * b[k][j];
            }
          }
        }
      }
      for (i = 0; i < blocking_size; i++)
        c[i_block + i][j_block] = temp_buff[i][0];
    }
  }
}
}

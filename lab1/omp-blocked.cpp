
// Header inclusions, if any...
#include <omp.h>
#include "lib/gemm.h"
#include <cstring>
// Using declarations, if any...
#define blocking_size 64
void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ])
{
int i,j,k;
  int nCores = omp_get_max_threads();
  omp_set_num_threads(nCores);
  
  float temp_buff2[blocking_size][blocking_size] = {0};

 
 float temp;
  #pragma omp parallel for private(i, j, k, temp) collapse(2)
  for (int b_i = 0; b_i < kI; b_i += blocking_size)
  { 
    for (int b_j = 0; b_j < kJ; b_j += blocking_size)
    {

    float temp_buff[blocking_size][blocking_size] = {0};
    
      for (int b_k = 0; b_k < kK; b_k += blocking_size)
      {

       #pragma omp parallel for schedule(static)
        for (i = b_i; i < ((b_i + blocking_size)); i++)
        {
          int indexI = i - b_i;
          int indexJ=0;
          for (k = b_k; k < ((b_k + blocking_size) ); k++)
          {
            temp = a[i][k];
            for (j = b_j; j < ((b_j + blocking_size) ); j++)
            {
              indexJ = j - b_j;
              temp_buff[indexI][indexJ] += temp * b[k][j];
             // temp_buff2[indexI][indexJ] += temp * b[k][j]; 
            }

          }
        }
      }
      //memcpy ( &temp_buff2, &temp_buff, sizeof(temp_buff) );
     //&temp_buff2 = &temp_buff;
   //  #pragma omp parallel for schedule(static)
      for (int i = 0; i < blocking_size; i++)
      { int indexI = b_i + i;
        memcpy(&c[indexI][b_j], &temp_buff[i][0], sizeof(float) * blocking_size);}
    }
  }
  
}

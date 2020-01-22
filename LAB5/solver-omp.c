#include "heat.h"
#include "omp.h"

/*
 * Function to copy one matrix into another
 */

void copy_mat (double *u, double *v, unsigned sizex, unsigned sizey)
{
 
    for (int i=1; i<=sizex-2; i++)
        for (int j=1; j<=sizey-2; j++) 
            v[ i*sizey+j ] = u[ i*sizey+j ];
}

/*
 * Blocked Jacobi solver: one iteration step
 */
double relax_jacobi (double *u, double *utmp, unsigned sizex, unsigned sizey)
{
    double diff, sum=0.0;
    //int howmany=omp_get_max_threads();
    #pragma omp parallel private(diff) reduction(+:sum)
    {
    int howmany=omp_get_num_threads();
    int blockid = omp_get_thread_num();
      int i_start = lowerb(blockid, howmany, sizex);
      int i_end = upperb(blockid, howmany, sizex);
      for (int i=max(1, i_start); i<= min(sizex-2, i_end); i++) {
        for (int j=1; j<= sizey-2; j++) {
	     utmp[i*sizey+j]= 0.25 * ( u[ i*sizey     + (j-1) ]+  // left
	                               u[ i*sizey     + (j+1) ]+  // right
				       u[ (i-1)*sizey + j     ]+  // top
				       u[ (i+1)*sizey + j     ]); // bottom
	     diff = utmp[i*sizey+j] - u[i*sizey + j];
	     sum += diff * diff; 
	 }
      }
    }
    return sum;
}

/*
 * Blocked Gauss-Seidel solver: one iteration step
 */
double relax_gauss (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;
    int howmany=omp_get_max_threads();
    #pragma omp parallel for ordered(2) private(unew,diff) reduction(+:sum)	
    for (int row = 0; row < howmany; ++row) { 
		for (int col = 0; col < howmany; ++col) {      //2 fors -> un para row i otro para col
			int row_start = lowerb(row, howmany, sizex);
			int row_end = upperb(row, howmany, sizex); 
			int col_start = lowerb(col, howmany, sizey);
			int col_end = upperb(col, howmany, sizey);   
			
			#pragma omp ordered depend(sink: row-1, col)           
			for (int i=max(1, row_start); i<= min(sizex-2, row_end); i++) {     //row_start,row_end,col_start,col_end
				for (int j=max(1, col_start); j<= min(sizey-2,col_end); j++) {
				unew= 0.25 * ( u[ i*sizey	+ (j-1) ]+  // left
					u[ i*sizey	+ (j+1) ]+  // right
					u[ (i-1)*sizey	+ j     ]+  // top
					u[ (i+1)*sizey	+ j     ]); // bottom
				diff = unew - u[i*sizey+ j];
				sum += diff * diff; 
				u[i*sizey+j]=unew;
				}
			}
			#pragma omp ordered depend(source)
		}
	}
    return sum;
}

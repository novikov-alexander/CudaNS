#include "header.h"

//---------------------------------------------------------------------
// this function performs the solution of the approximate factorization
// step in the y-direction for all five matrix components
// simultaneously. The Thomas algorithm is employed to solve the
// systems for the y-lines. Boundary conditions are non-periodic
//---------------------------------------------------------------------

#undef rhs
#define rhs(x,y,z,m) rhs[x + (y) * P_SIZE + (z) * P_SIZE * P_SIZE + (m) * P_SIZE * P_SIZE * P_SIZE]

__global__ void y_solve_kernel_one(double* lhs_, double* lhsp_, double* lhsm_, int nx2, int ny2, int nz2)
{
	int m;

	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int k = threadIdx.y + blockIdx.y * blockDim.y + 1;

	//part 1
	if (k <= nz2 && i <= nx2)
    {        
        for (m = 0; m < 5; m++)
        {
            lhs_(k,i,0,m) = lhs_(k,i,ny2 + 1,m) = 0.0;
            lhsp_(k,i,0,m) = lhsp_(k,i,ny2 + 1,m) = 0.0;
            lhsm_(k,i,0,m) = lhsm_(k,i,ny2 + 1,m) = 0.0;
        }
        lhs_(k,i,0,2) = lhs_(k,i,ny2 + 1,2) = 1.0;
        lhsp_(k,i,0,2) = lhsp_(k,i,ny2 + 1,2) = 1.0;
        lhsm_(k,i,0,2) = lhsm_(k,i,ny2 + 1,2) = 1.0;       
	}
}

__global__ void y_solve_kernel_two(double* lhs_, double* lhsp_, double* lhsm_, double* rhs, double* rho_i, double* vs, double* speed, double c3c4, double dy3, double  con43, double  dy5, double c1c5, double dy1, double dtty2, double dtty1, double dymax, double c2dtty1, double comz1, double comz4, double comz5, double comz6, int nx2, int ny2, int nz2, int ny)
{
	int  m;
	double ru1, rhoq1;

	int k = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int i = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int j = threadIdx.z + blockIdx.z * blockDim.z + 1;

	//part 2
	if (k <= nz2 && j <= ny2 && i <= nx2)
    {        
        lhs_(k,i,j,0) = 0.0;

        ru1 = c3c4*rho_i(k,j - 1,i);
        rhoq1 = fmax(fmax(dy3 + con43*ru1, dy5 + c1c5*ru1), fmax(dymax + ru1, dy1));
        lhs_(k,i,j,1) = -dtty2 * vs(k,j - 1,i) - dtty1 * rhoq1;

        ru1 = c3c4*rho_i(k,j,i);
        rhoq1 = fmax(fmax(dy3 + con43*ru1, dy5 + c1c5*ru1), fmax(dymax + ru1, dy1));
        lhs_(k,i,j,2) = 1.0 + c2dtty1 * rhoq1;

        ru1 = c3c4*rho_i(k,j + 1,i);
        rhoq1 = fmax(fmax(dy3 + con43*ru1, dy5 + c1c5*ru1), fmax(dymax + ru1, dy1));
        lhs_(k,i,j,3) = dtty2 * vs(k,j + 1,i) - dtty1 * rhoq1;
        lhs_(k,i,j,4) = 0.0;

        if (j == 1)
        {
            lhs_(k,i,j,2) = lhs_(k,i,j,2) + comz5;
            lhs_(k,i,j,3) = lhs_(k,i,j,3) - comz4;
            lhs_(k,i,j,4) = lhs_(k,i,j,4) + comz1;
        }
        else if (j == 2)
        {
            lhs_(k,i,j,1) = lhs_(k,i,j,1) - comz4;
            lhs_(k,i,j,2) = lhs_(k,i,j,2) + comz6;
            lhs_(k,i,j,3) = lhs_(k,i,j,3) - comz4;
            lhs_(k,i,j,4) = lhs_(k,i,j,4) + comz1;
        }
        else if (j == ny - 3)
        {
            lhs_(k,i,j,0) = lhs_(k,i,j,0) + comz1;
            lhs_(k,i,j,1) = lhs_(k,i,j,1) - comz4;
            lhs_(k,i,j,2) = lhs_(k,i,j,2) + comz6;
            lhs_(k,i,j,3) = lhs_(k,i,j,3) - comz4;
        }
        else if (j == ny - 2)
        {
            lhs_(k,i,j,0) = lhs_(k,i,j,0) + comz1;
            lhs_(k,i,j,1) = lhs_(k,i,j,1) - comz4;
            lhs_(k,i,j,2) = lhs_(k,i,j,2) + comz5;
        }
        else
        {
            lhs_(k,i,j,0) = lhs_(k,i,j,0) + comz1;
            lhs_(k,i,j,1) = lhs_(k,i,j,1) - comz4;
            lhs_(k,i,j,2) = lhs_(k,i,j,2) + comz6;
            lhs_(k,i,j,3) = lhs_(k,i,j,3) - comz4;
            lhs_(k,i,j,4) = lhs_(k,i,j,4) + comz1;
        }

        lhsp_(k,i,j,0) = lhs_(k,i,j,0);
        lhsp_(k,i,j,1) = lhs_(k,i,j,1) - dtty2 * speed(k,j - 1,i);
        lhsp_(k,i,j,2) = lhs_(k,i,j,2);
        lhsp_(k,i,j,3) = lhs_(k,i,j,3) + dtty2 * speed(k,j + 1,i);
        lhsp_(k,i,j,4) = lhs_(k,i,j,4);

        lhsm_(k,i,j,0) = lhs_(k,i,j,0);
        lhsm_(k,i,j,1) = lhs_(k,i,j,1) + dtty2 * speed(k,j - 1,i);
        lhsm_(k,i,j,2) = lhs_(k,i,j,2);
        lhsm_(k,i,j,3) = lhs_(k,i,j,3) - dtty2 * speed(k,j + 1,i);
        lhsm_(k,i,j,4) = lhs_(k,i,j,4);
	}
}

__global__ void y_solve_kernel_three(double* lhs_, double* lhsp_, double* lhsm_, double* rhs, double* rho_i, double* vs, double* speed, double c3c4, double dy3, double  con43, double  dy5, double c1c5, double dy1, double dtty2, double dtty1, double dymax, double c2dtty1, double comz1, double comz4, double comz5, double comz6, int nx2, int ny2, int nz2, int ny)
{
	int  j1, j2, m;
	double ru1, rhoq1, fac1, fac2;

    int k = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int j;
	int i = threadIdx.y + blockIdx.y * blockDim.y + 1;

	//part 3
	if (k <= nz2 && i <= nx2)
    {        
        for (j = 1; j <= ny2; j++)
        {
            j1 = j;
            j2 = j + 1;

            fac1 = 1.0 / lhs_(k,i,j - 1,2);
            lhs_(k,i,j - 1,3) = fac1*lhs_(k,i,j - 1,3);
            lhs_(k,i,j - 1,4) = fac1*lhs_(k,i,j - 1,4);
            #pragma unroll 3
            for (m = 0; m < 3; m++)
                rhs(k,j - 1,i,m) = fac1*rhs(k,j - 1,i,m);

            lhs_(k,i,j1,2) = lhs_(k,i,j1,2) - lhs_(k,i,j1,1) * lhs_(k,i,j - 1,3);
            lhs_(k,i,j1,3) = lhs_(k,i,j1,3) - lhs_(k,i,j1,1) * lhs_(k,i,j - 1,4);
            #pragma unroll 3
            for (m = 0; m < 3; m++)
                rhs(k,j1,i,m) = rhs(k,j1,i,m) - lhs_(k,i,j1,1) * rhs(k,j - 1,i,m);

            lhs_(k,i,j2,1) = lhs_(k,i,j2,1) - lhs_(k,i,j2,0) * lhs_(k,i,j - 1,3);
            lhs_(k,i,j2,2) = lhs_(k,i,j2,2) - lhs_(k,i,j2,0) * lhs_(k,i,j - 1,4);
            #pragma unroll 3
            for (m = 0; m < 3; m++)
                rhs(k,j2,i,m) = rhs(k,j2,i,m) - lhs_(k,i,j2,0) * rhs(k,j - 1,i,m);

            if (j == ny2)
            {
                fac1 = 1.0 / lhs_(k,i,j1,2);
                lhs_(k,i,j1,3) = fac1 * lhs_(k,i,j1,3);
                lhs_(k,i,j1,4) = fac1 * lhs_(k,i,j1,4);
                #pragma unroll 3
                for (m = 0; m < 3; m++)
                    rhs(k,j1,i,m) = fac1 * rhs(k,j1,i,m);

             lhs_(k,i,j2,2) = lhs_(k,i,j2,2) - lhs_(k,i,j2,1) * lhs_(k,i,j1,3);
             lhs_(k,i,j2,3) = lhs_(k,i,j2,3) - lhs_(k,i,j2,1) * lhs_(k,i,j1,4);
             for (m = 0; m < 3; m++)
                 rhs(k,j2,i,m) = rhs(k,j2,i,m) - lhs_(k,i,j2,1) * rhs(k,j1,i,m);

                 fac2 = 1.0 / lhs_(k,i,j2,2);
                 #pragma unroll 3
                 for (m = 0; m < 3; m++)
                     rhs(k,j2,i,m) = fac2 * rhs(k,j2,i,m);
             }
            
             m = 3;
             fac1 = 1.0 / lhsp_(k,i,j - 1,2);
             lhsp_(k,i,j - 1,3) = fac1 * lhsp_(k,i,j - 1,3);
             lhsp_(k,i,j - 1,4) = fac1 * lhsp_(k,i,j - 1,4);

             rhs(k,j - 1,i,m) = fac1 * rhs(k,j - 1,i,m);
             lhsp_(k,i,j1,2) = lhsp_(k,i,j1,2) - lhsp_(k,i,j1,1) * lhsp_(k,i,j - 1,3);
             lhsp_(k,i,j1,3) = lhsp_(k,i,j1,3) - lhsp_(k,i,j1,1) * lhsp_(k,i,j - 1,4);

             rhs(k,j1,i,m) = rhs(k,j1,i,m) - lhsp_(k,i,j1,1) * rhs(k,j - 1,i,m);
             lhsp_(k,i,j2,1) = lhsp_(k,i,j2,1) - lhsp_(k,i,j2,0) * lhsp_(k,i,j - 1,3);
             lhsp_(k,i,j2,2) = lhsp_(k,i,j2,2) - lhsp_(k,i,j2,0) * lhsp_(k,i,j - 1,4);
             rhs(k,j2,i,m) = rhs(k,j2,i,m) - lhsp_(k,i,j2,0) * rhs(k,j - 1,i,m);

             m = 4;
             fac1 = 1.0 / lhsm_(k,i,j - 1,2);
             lhsm_(k,i,j - 1,3) = fac1 * lhsm_(k,i,j - 1,3);
             lhsm_(k,i,j - 1,4) = fac1 * lhsm_(k,i,j - 1,4);

             rhs(k,j - 1,i,m) = fac1 * rhs(k,j - 1,i,m);
             lhsm_(k,i,j1,2) = lhsm_(k,i,j1,2) - lhsm_(k,i,j1,1) * lhsm_(k,i,j - 1,3);
             lhsm_(k,i,j1,3) = lhsm_(k,i,j1,3) - lhsm_(k,i,j1,1) * lhsm_(k,i,j - 1,4);

             rhs(k,j1,i,m) = rhs(k,j1,i,m) - lhsm_(k,i,j1,1) * rhs(k,j - 1,i,m);
             lhsm_(k,i,j2,1) = lhsm_(k,i,j2,1) - lhsm_(k,i,j2,0) * lhsm_(k,i,j - 1,3);
             lhsm_(k,i,j2,2) = lhsm_(k,i,j2,2) - lhsm_(k,i,j2,0) * lhsm_(k,i,j - 1,4);
             rhs(k,j2,i,m) = rhs(k,j2,i,m) - lhsm_(k,i,j2,0) * rhs(k,j - 1,i,m);

             if (j == ny2)
             {
                 m = 3;
                 fac1 = 1.0 / lhsp_(k,i,j1,2);
                 lhsp_(k,i,j1,3) = fac1 * lhsp_(k,i,j1,3);
                 lhsp_(k,i,j1,4) = fac1 * lhsp_(k,i,j1,4);

                 rhs(k,j1,i,m) = fac1 * rhs(k,j1,i,m);
                 lhsp_(k,i,j2,2) = lhsp_(k,i,j2,2) - lhsp_(k,i,j2,1) * lhsp_(k,i,j1,3);
                 lhsp_(k,i,j2,3) = lhsp_(k,i,j2,3) - lhsp_(k,i,j2,1) * lhsp_(k,i,j1,4);
                 rhs(k,j2,i,m) = rhs(k,j2,i,m) - lhsp_(k,i,j2,1) * rhs(k,j1,i,m);

                 m = 4;
                 fac1 = 1.0 / lhsm_(k,i,j1,2);
                 lhsm_(k,i,j1,3) = fac1 * lhsm_(k,i,j1,3);
                 lhsm_(k,i,j1,4) = fac1 * lhsm_(k,i,j1,4);
                 rhs(k,j1,i,m) = fac1 * rhs(k,j1,i,m);

                 lhsm_(k,i,j2,2) = lhsm_(k,i,j2,2) - lhsm_(k,i,j2,1) * lhsm_(k,i,j1,3);
                 lhsm_(k,i,j2,3) = lhsm_(k,i,j2,3) - lhsm_(k,i,j2,1) * lhsm_(k,i,j1,4);
                 rhs(k,j2,i,m) = rhs(k,j2,i,m) - lhsm_(k,i,j2,1) * rhs(k,j1,i,m);

                 rhs(k,j2,i,3) = rhs(k,j2,i,3) / lhsp_(k,i,j2,2);
                 rhs(k,j2,i,4) = rhs(k,j2,i,4) / lhsm_(k,i,j2,2);

                 #pragma unroll 3
                 for (m = 0; m < 3; m++)
                     rhs(k,j1,i,m) = rhs(k,j1,i,m) - lhs_(k,i,j1,3) * rhs(k,j2,i,m);
                 rhs(k,j1,i,3) = rhs(k,j1,i,3) - lhsp_(k,i,j1,3) * rhs(k,j2,i,3);
                 rhs(k,j1,i,4) = rhs(k,j1,i,4) - lhsm_(k,i,j,3) * rhs(k,j2,i,4);
             }
        }
    }
}


__global__ void y_solve_kernel_four(double* lhs_, double* lhsp_, double* lhsm_, double* rhs, double* rho_i, double* vs, double* speed, double c3c4, double dy3, double  con43, double  dy5, double c1c5, double dy1, double dtty2, double dtty1, double dymax, double c2dtty1, double comz1, double comz4, double comz5, double comz6, int nx2, int ny2, int nz2, int ny)
{
	int  j1, j2, m;

	int k = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j;
	int i = threadIdx.y + blockIdx.y * blockDim.y + 1;

	//part 4
	if ((k <= nz2) && (i <= nx2))
	{
	    for (j = ny2; j >= 1; j--)
        {
            j1 = j;
            j2 = j + 1;

            for (m = 0; m < 3; m++)
               rhs(k,j - 1,i,m) = rhs(k,j - 1,i,m) - lhs_(k,i,j - 1,3) * rhs(k,j1,i,m) - lhs_(k,i,j - 1,4) * rhs(k,j2,i,m);
            
            rhs(k,j - 1,i,3) = rhs(k,j - 1,i,3) - lhsp_(k,i,j - 1,3) * rhs(k,j1,i,3) - lhsp_(k,i,j - 1,4) * rhs(k,j2,i,3);
            rhs(k,j - 1,i,4) = rhs(k,j - 1,i,4) - lhsm_(k,i,j - 1,3) * rhs(k,j1,i,4) - lhsm_(k,i,j - 1,4) * rhs(k,j2,i,4);
        }
	}
}

__global__ void y_solve_inversion(double* rhs, double bt, int nx2, int ny2, int nz2)
{
    double r1, r2, r3, r4, r5, t1, t2;

	int k = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int i = threadIdx.z + blockIdx.z * blockDim.z + 1;

	if ((k <= nz2) && (j <= ny2) && (i <= nx2))
    {
        r1 = rhs(k,j,i,0);
        r2 = rhs(k,j,i,1);
        r3 = rhs(k,j,i,2);
        r4 = rhs(k,j,i,3);
        r5 = rhs(k,j,i,4);

        t1 = bt * r1;
        t2 = 0.5 * (r4 + r5);
        
        rhs(k,j,i,0) = bt * (r4 - r5);
        rhs(k,j,i,1) = -r3;
        rhs(k,j,i,2) = r2;
        rhs(k,j,i,3) = -t1 + t2;
        rhs(k,j,i,4) = t1 + t2;
    }
}




void y_solve()
{
    int i, j, k, j1, j2, m;
    double ru1, rhoq1, fac1, fac2;

	const int size5 = sizeof(double)*P_SIZE*P_SIZE*P_SIZE*5;
	const int size = sizeof(double)*P_SIZE*P_SIZE*P_SIZE;

	dim3 blocks = dim3(nx2 / 32+1, ny2, nz2);
	dim3 threads = dim3(32, 1, 1);

    dim3 blocks2 = dim3(nx2 / 32 + 1, nz2 / 8 + 1);
	dim3 threads2 = dim3(32, 8);

    if (timeron) timer_start(t_ysolve);

    cudaDeviceSynchronize();
	y_solve_kernel_one<<<blocks2, threads2>>>((double*)lhs_gpu, (double*)lhsp_gpu, (double*)lhsm_gpu, nx2, ny2, nz2);
    
    cudaDeviceSynchronize();
    y_solve_kernel_two<<<blocks, threads>>>((double*) lhs_gpu, (double*) lhsp_gpu, (double*) lhsm_gpu, (double*) gpuRhs, (double*) gpuRho_i, (double*) gpuVs, (double*) gpuSpeed, c3c4, dy3, con43, dy5, c1c5, dy1, dtty2, dtty1, dymax, c2dtty1, comz1, comz4, comz5, comz6, nx2, ny2, nz2, ny);
    
	cudaDeviceSynchronize();
    y_solve_kernel_three<<<blocks2, threads2>>>((double*) lhs_gpu, (double*) lhsp_gpu, (double*) lhsm_gpu, (double*) gpuRhs, (double*) gpuRho_i, (double*) gpuVs, (double*) gpuSpeed, c3c4, dy3, con43, dy5, c1c5, dy1, dtty2, dtty1, dymax, c2dtty1, comz1, comz4, comz5, comz6, nx2, ny2, nz2, ny);
   
	cudaDeviceSynchronize();
    y_solve_kernel_four<<<blocks2, threads2>>>((double*) lhs_gpu, (double*) lhsp_gpu, (double*) lhsm_gpu, (double*) gpuRhs, (double*) gpuRho_i, (double*) gpuVs, (double*) gpuSpeed, c3c4, dy3, con43, dy5, c1c5, dy1, dtty2, dtty1, dymax, c2dtty1, comz1, comz4, comz5, comz6, nx2, ny2, nz2, ny);
   

    //---------------------------------------------------------------------
    // block-diagonal matrix-vector multiplication                       
    //---------------------------------------------------------------------
    if (timeron) timer_start(t_pinvr);

	y_solve_inversion<<<blocks, threads>>>((double*)gpuRhs, bt, nx2, ny2, nz2);

    if (timeron) timer_stop(t_pinvr);
    if (timeron) timer_stop(t_ysolve);
}

#undef lhs

#include "header.h"
#include <stdio.h>
#include <stdlib.h>

//---------------------------------------------------------------------
// This subroutine initializes the field variable u using 
// tri-linear transfinite interpolation of the boundary values     
//---------------------------------------------------------------------
void initialize()
{
    int i, j, k, m, ix, iy, iz;
    double xi, eta, zeta, Pface[2][3][5], Pxi, Peta, Pzeta, temp[5];
    
    for (k = 0; k <= nz - 1; k++) 
    {
        for (j = 0; j <= ny - 1; j++) 
        {
            for (i = 0; i <= nx - 1; i++) 
            {
                u[k][j][i][0] = 1.0;
                u[k][j][i][1] = 0.0;
                u[k][j][i][2] = 0.0;
                u[k][j][i][3] = 0.0;
                u[k][j][i][4] = 1.0;

                xi = i * dnxm1;
                eta = j * dnym1;
                zeta = k * dnzm1;

                for (ix = 0; ix < 2; ix++) 
                {
                    Pxi = ix;
                    exact_solution(Pxi, eta, zeta, &Pface[ix][0][0]);
                }

                for (iy = 0; iy < 2; iy++) 
                {
                    Peta = iy;
                    exact_solution(xi, Peta, zeta, &Pface[iy][1][0]);
                }

                for (iz = 0; iz < 2; iz++) 
                {
                    Pzeta = iz;
                    exact_solution(xi, eta, Pzeta, &Pface[iz][2][0]);
                }

                for (m = 0; m < 5; m++)
                {
                    Pxi = xi * Pface[1][0][m] + (1.0 - xi) * Pface[0][0][m];
                    Peta = eta * Pface[1][1][m] + (1.0 - eta) * Pface[0][1][m];
                    Pzeta = zeta * Pface[1][2][m] + (1.0 - zeta) * Pface[0][2][m];

                    u[k][j][i][m] = Pxi + Peta + Pzeta - Pxi*Peta - Pxi*Pzeta - Peta*Pzeta + Pxi*Peta*Pzeta;
                }

                if (i == 0)
                {
                    xi = 0.0;
                    eta = j * dnym1;
                    zeta = k * dnzm1;
                    
                    exact_solution(xi, eta, zeta, temp);

                    for (m = 0; m < 5; m++)
                        u[k][j][i][m] = temp[m];
                }
                else if (i == nx - 1)
                {
                    xi = 1.0;
                    eta = j * dnym1;
                    zeta = k * dnzm1;

                    exact_solution(xi, eta, zeta, temp);
                 
                    for (m = 0; m < 5; m++)
                        u[k][j][i][m] = temp[m];
                }
                else if (j == 0)
                {
                    xi = i * dnxm1;
                    eta = 0.0;
                    zeta = k * dnzm1;

                    exact_solution(xi, eta, zeta, temp);
                   
                    for (m = 0; m < 5; m++)
                        u[k][j][i][m] = temp[m];
                }
                else if (j == ny - 1)
                {
                    xi = i * dnxm1;
                    eta = 1.0;
                    zeta = k * dnzm1;

                    exact_solution(xi, eta, zeta, temp);
                  
                    for (m = 0; m < 5; m++)
                        u[k][j][i][m] = temp[m];
                }
                else if (k == 0)
                {
                    xi = i * dnxm1;
                    eta = j * dnym1;
                    zeta = 0.0;

                    exact_solution(xi, eta, zeta, temp);

                    for (m = 0; m < 5; m++)
                        u[k][j][i][m] = temp[m];
                }
                else if (k == nz - 1)
                {
                    xi = i * dnxm1;
                    eta = j * dnym1;
                    zeta = 1.0;

                    exact_solution(xi, eta, zeta, temp);

                    for (m = 0; m < 5; m++)
                        u[k][j][i][m] = temp[m];
                }
            }
        }
    }
}

logical inittrace(const char** t_names)
{
    logical timeron = false;
    if (TRACE)
    {
        timeron = true;
        t_names[t_total] = "total";
        t_names[t_rhsx] = "x direction";
        t_names[t_rhsy] = "y direction";
        t_names[t_rhsz] = "z direction";
        t_names[t_rhs] = "rhs";
        t_names[t_xsolve] = "x_solve";
        t_names[t_ysolve] = "y_solve";
        t_names[t_zsolve] = "z_solve";
        t_names[t_tzetar] = "block-diag mv";
        t_names[t_ninvr] = "block-diag inv";
        t_names[t_pinvr] = "block-diag inv";
        t_names[t_txinvr] = "x invr";
        t_names[t_add] = "add";
    }
    return timeron;
}

int initparameters(int argc, char **argv, int *niter)
{
    int OK = true;

    FILE *fp;
    if ((fp = fopen("input.data", "r")) != NULL)
    {
        int result = 0;
        printf(" Reading from input file input.data\n");
        result = fscanf(fp, "%d", niter);
        while (fgetc(fp) != '\n');
        result = fscanf(fp, "%lf", &dt);
        while (fgetc(fp) != '\n');
        result = fscanf(fp, "%d%d%d", &nx, &ny, &nz);
        fclose(fp);
    }
    else
    {
        //printf(" No input file input.data. Using compiled defaults\n");
        *niter = NITER_DEFAULT;
        dt = DT_DEFAULT;
        nx = P_SIZE;
        ny = P_SIZE;
        nz = P_SIZE;
    }

    printf(" Size: %4dx%4dx%4d\n", nx, ny, nz);
    printf(" Iterations: %4d    dt: %10.6f\n", *niter, dt);
    printf("\n");

    if ((nx > P_SIZE) || (ny > P_SIZE) || (nz > P_SIZE))
    {
        printf(" %d, %d, %d\n", nx, ny, nz);
        printf(" Problem size too big for compiled array sizes\n");
        OK = false;
    }
    else
    {
        nx2 = nx - 2;
        ny2 = ny - 2;
        nz2 = nz - 2;
    }

    return OK;
}

int allocateArrays()
{
	CudaSafeCall(cudaMalloc((void**) &lhs_gpu, sizeof(double) * nx * ny * nz * 5));
	CudaSafeCall(cudaMalloc((void**) &lhsp_gpu, sizeof(double) * nx * ny * nz * 5));
	CudaSafeCall(cudaMalloc((void**) &lhsm_gpu, sizeof(double) * nx * ny * nz * 5));

	CudaSafeCall(cudaMalloc((void**) &gpuU, sizeof(double) * nx * ny * nz * 5));
	CudaSafeCall(cudaMalloc((void**) &gpuRhs, sizeof(double) * nx * ny * nz * 5));
	CudaSafeCall(cudaMalloc((void**) &gpuForcing, sizeof(double) * nx * ny * nz * 5));
    CudaSafeCall(cudaMalloc((void**) &gpuTmp, sizeof(double) * nx * ny * nz * 5));

    u = (double (*)[P_SIZE][P_SIZE][5] ) malloc(sizeof(double) * nx * ny * nz * 5);
    rhs = (double (*)[P_SIZE][P_SIZE][5] ) malloc(sizeof(double) * nx * ny * nz * 5);
    forcing = (double (*)[P_SIZE][P_SIZE][5] ) malloc(sizeof(double) * nx * ny * nz * 5);
    
	CudaSafeCall(cudaMalloc((void**) &gpuUs, sizeof(double) * nx * ny * nz));
	CudaSafeCall(cudaMalloc((void**) &gpuVs, sizeof(double) * nx * ny * nz));
	CudaSafeCall(cudaMalloc((void**) &gpuWs, sizeof(double) * nx * ny * nz));
	CudaSafeCall(cudaMalloc((void**) &gpuQs, sizeof(double) * nx * ny * nz));
	CudaSafeCall(cudaMalloc((void**) &gpuRho_i, sizeof(double) * nx * ny * nz));
	CudaSafeCall(cudaMalloc((void**) &gpuSpeed, sizeof(double) * nx * ny * nz));
	CudaSafeCall(cudaMalloc((void**) &gpuSquare, sizeof(double) * nx * ny * nz));
    
    us = (double (*)[P_SIZE][P_SIZE] ) malloc(sizeof(double) * nx * ny * nz);
    vs = (double (*)[P_SIZE][P_SIZE] ) malloc(sizeof(double) * nx * ny * nz);
    ws = (double (*)[P_SIZE][P_SIZE] ) malloc(sizeof(double) * nx * ny * nz);
    qs = (double (*)[P_SIZE][P_SIZE] ) malloc(sizeof(double) * nx * ny * nz);    
    rho_i = (double (*)[P_SIZE][P_SIZE] ) malloc(sizeof(double) * nx * ny * nz);    
    speed = (double (*)[P_SIZE][P_SIZE] ) malloc(sizeof(double) * nx * ny * nz);    
    square = (double (*)[P_SIZE][P_SIZE] ) malloc(sizeof(double) * nx * ny * nz);    
    
    return 1;
}

int deallocateArrays()
{
	cudaFree(gpuU);
	cudaFree(gpuRhs);
	cudaFree(gpuRho_i);
	cudaFree(gpuUs);
	cudaFree(gpuVs);
	cudaFree(gpuWs);
	cudaFree(gpuQs);
	cudaFree(gpuSquare);
	cudaFree(gpuSpeed);
	cudaFree(gpuForcing);
	cudaFree(lhs_gpu);
	cudaFree(lhsp_gpu);
	cudaFree(lhsm_gpu);

    free(u);
    free(rhs);
    free(forcing);
    
    free(us);
    free(vs);
    free(ws);
    free(qs);
    free(rho_i);
    free(speed);
    free(square);
    
    return 1;
}

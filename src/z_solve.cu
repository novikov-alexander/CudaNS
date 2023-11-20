#include "header.hpp"
#include <algorithm>

//---------------------------------------------------------------------
// this function performs the solution of the approximate factorization
// step in the z-direction for all five matrix components
// simultaneously. The Thomas algorithm is employed to solve the
// systems for the z-lines. Boundary conditions are non-periodic
//---------------------------------------------------------------------

void z_solve_two(
    CUDAParameters cudaParams,
    double *lhs_, double *lhsp_, double *lhsm_, double *rho_i, double *ws, double *speed, int nx2, int ny2, int nz2, double c3c4, double dz4, double con43, double dz5, double c1c5, double dzmax, double dz1, double dttz2, double dttz1, double c2dttz1, double comz1, double comz4, double comz5, double comz6)
{
    run_solve_kernels(cudaParams, (double *)lhs_, (double *)lhsp_, (double *)lhsm_, (double *)rhs, (double *)rho_i, (double *)us, (double *)speed, c3c4, dy3, con43, dy5, c1c5, dy1, dtty2, dtty1, dymax, c2dtty1, comz1, comz4, comz5, comz6, ny2, nx2, nz2, ny);
}

void z_solve_inversion(dim3 blocks, dim3 threads, double *rhs, double bt, int ny2, int nx2, int nz2)
{
    run_inversion_kernels(blocks, threads, rhs, bt, nz2, nx2, ny2);
}

#define src(x, y, z, m) src[x + (y) * P_SIZE + (z) * P_SIZE * P_SIZE + (m) * P_SIZE * P_SIZE * P_SIZE]
#define dst(x, y, z, m) dst[y + (z) * P_SIZE + (x) * P_SIZE * P_SIZE + (m) * P_SIZE * P_SIZE * P_SIZE]
__global__ void z_solve_transpose(double *dst, double *src, int nx2, int ny2, int nz2)
{
    int m;

    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.z + blockIdx.z * blockDim.z;

    if ((k <= nz2 + 1) && (j <= ny2 + 1) && (i <= nx2 + 1))
    {
#pragma unroll 5
        for (m = 0; m < 5; m++)
        {
            dst(i, j, k, m) = src(i, j, k, m);
        }
    }
}

#undef src
#define src(x, y, z, m) src[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE + (m) * P_SIZE * P_SIZE * P_SIZE]
__global__ void z_solve_inv_transpose(double *dst, double *src, int nx2, int ny2, int nz2)
{
    int m;

    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.z + blockIdx.z * blockDim.z;

    if ((k <= nz2 + 1) && (j <= ny2 + 1) && (i <= nx2 + 1))
    {
#pragma unroll 5
        for (m = 0; m < 5; m++)
        {
            src(i, j, k, m) = dst(i, j, k, m);
        }
    }
}

#undef src
#undef dst

#define src(x, y, z) src[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE]
#define dst(x, y, z) dst[y + (z) * P_SIZE + (x) * P_SIZE * P_SIZE]
__global__ void z_solve_transpose_3D(double *dst, double *src, int nx2, int ny2, int nz2)
{
    int m;

    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.z + blockIdx.z * blockDim.z;

    if ((k <= nz2 + 1) && (j <= ny2 + 1) && (i <= nx2 + 1))
    {
        dst(i, j, k) = src(i, j, k);
    }
}

__global__ void z_solve_inv_transpose_3D(double *dst, double *src, int nx2, int ny2, int nz2)
{
    int m;

    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.z + blockIdx.z * blockDim.z;

    if ((k <= nz2 + 1) && (j <= ny2 + 1) && (i <= nx2 + 1))
    {
        src(i, j, k) = dst(i, j, k);
    }
}

#undef src
#undef dst

void z_solve()
{

    int i, j, k, k1, k2, m;
    double ru1, rhos1, fac1, fac2;

    CUDAParameters cudaParams = setupDimensions(nx2, ny2, nz2, nx, ny, nz);

    if (timeron)
        timer_start(t_zsolve);

    z_solve_transpose_3D<<<cudaParams.blockst, cudaParams.threadst>>>((double *)gpuTmp3D, (double *)gpuWs, nx2, ny2, nz2);
    z_solve_transpose<<<cudaParams.blockst, cudaParams.threadst>>>((double *)gpuTmp, (double *)gpuRhs, nx2, ny2, nz2);
    cudaDeviceSynchronize();
    std::swap(gpuTmp, gpuRhs);
    std::swap(gpuTmp3D, gpuWs);
    cudaDeviceSynchronize();

    z_solve_transpose_3D<<<cudaParams.blockst, cudaParams.threadst>>>((double *)gpuTmp3D, (double *)gpuSpeed, nx2, ny2, nz2);
    std::swap(gpuTmp3D, gpuSpeed);

    z_solve_two(
        cudaParams,
        (double *)lhs_gpu, (double *)lhsp_gpu, (double *)lhsm_gpu, (double *)gpuRho_i, (double *)gpuWs, (double *)gpuSpeed, nx2, ny2, nz2, c3c4, dz4, con43, dz5, c1c5, dzmax, dz1, dttz2, dttz1, c2dttz1, comz1, comz4, comz5, comz6);

    //---------------------------------------------------------------------
    // block-diagonal matrix-vector multiplication
    //---------------------------------------------------------------------

    if (timeron)
        timer_start(t_tzetar);

    z_solve_inversion(cudaParams.blocks, cudaParams.threads, (double *)gpuRhs, bt, nx2, ny2, nz2);

    if (timeron)
        timer_stop(t_tzetar);

    std::swap(gpuTmp, gpuRhs);
    z_solve_inv_transpose<<<cudaParams.blockst, cudaParams.threadst>>>((double *)gpuTmp, (double *)gpuRhs, nx2, ny2, nz2);
    std::swap(gpuTmp3D, gpuWs);
    z_solve_inv_transpose_3D<<<cudaParams.blockst, cudaParams.threadst>>>((double *)gpuTmp3D, (double *)gpuWs, nx2, ny2, nz2);

    cudaDeviceSynchronize();

    std::swap(gpuTmp3D, gpuSpeed);
    z_solve_inv_transpose_3D<<<cudaParams.blockst, cudaParams.threadst>>>((double *)gpuTmp3D, (double *)gpuSpeed, nx2, ny2, nz2);

    if (timeron)
        timer_stop(t_zsolve);
}

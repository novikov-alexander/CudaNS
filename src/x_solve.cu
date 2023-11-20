#include "header.hpp"
#include <algorithm>

//---------------------------------------------------------------------
// this function performs the solution of the approximate factorization
// step in the x-direction for all five matrix components
// simultaneously. The Thomas algorithm is employed to solve the
// systems for the x-lines. Boundary conditions are non-periodic
//---------------------------------------------------------------------

void x_solve_two(
    dim3 blocks, dim3 threads,
    dim3 blocks2, dim3 threads2,
    double *lhs_, double *lhsp_, double *lhsm_, double *rhs, double *rho_i, double *us, double *speed, double c3c4, double dx2, double con43, double dx5, double c1c5, double dx1, double dttx2, double dttx1, double dxmax, double c2dttx1, double comz1, double comz4, double comz5, double comz6,
    int nx2, int ny2, int nz2, int nx)
{
    // reassign x- and z- dimensions
    run_solve_kernels(blocks, threads, blocks2, threads2, (double *)lhs_, (double *)lhsp_, (double *)lhsm_, rhs, (double *)rho_i, (double *)us, (double *)speed, c3c4, dy3, con43, dy5, c1c5, dy1, dtty2, dtty1, dymax, c2dtty1, comz1, comz4, comz5, comz6, nz2, ny2, nx2, ny);
}

#undef rhs
#define rhs(x, y, z, m) rhs[INDEX(x, y, z, m)]

#undef rho_i
#undef s
#undef speed
#define rho_i(x, y, z) rho_i[INDEX_3D(x, z, y)]
#define s(x, y, z) s[INDEX_3D(x, z, y)]
#define speed(x, y, z) speed[INDEX_3D(x, z, y)]

__global__ void x_solve_inversion(double *rhs, double bt, int nx2, int ny2, int nz2)
{
    double r1, r2, r3, r4, r5, t1, t2;

    int k = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int i = threadIdx.z + blockIdx.z * blockDim.z + 1;

    if ((k <= nz2) && (j <= ny2) && (i <= nx2))
    {
        r1 = rhs(k, j, i, 0);
        r2 = rhs(k, j, i, 1);
        r3 = rhs(k, j, i, 2);
        r4 = rhs(k, j, i, 3);
        r5 = rhs(k, j, i, 4);

        t1 = bt * r3;
        t2 = 0.5 * (r4 + r5);

        rhs(k, j, i, 0) = -r2;
        rhs(k, j, i, 1) = r1;
        rhs(k, j, i, 2) = bt * (r4 - r5);
        rhs(k, j, i, 3) = -t1 + t2;
        rhs(k, j, i, 4) = t1 + t2;
    }
}

#define src(x, y, z, m) src[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE + (m) * P_SIZE * P_SIZE * P_SIZE]
#define dst(x, y, z, m) dst[x + (y) * P_SIZE + (z) * P_SIZE * P_SIZE + (m) * P_SIZE * P_SIZE * P_SIZE]
__global__ void x_solve_transpose(double *dst, double *src, int nx2, int ny2, int nz2)
{
    int m;

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

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
#undef dst
#define src(x, y, z) src[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE]
#define dst(x, y, z) dst[x + (z) * P_SIZE + (y) * P_SIZE * P_SIZE]
__global__ void x_solve_transpose_3D(double *dst, double *src, int nx2, int ny2, int nz2)
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

__global__ void x_solve_inv_transpose_3D(double *dst, double *src, int nx2, int ny2, int nz2)
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
/*__global__ void x_solve_inv_transpose(double *dst, double *src, int nx2, int ny2, int nz2){
    int m;

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if ((k <= nz2 + 1) && (j <= ny2 + 1) && (i <= nx2 + 1))
    {
        #pragma unroll 5
        for (m = 0; m < 5; m++)
        {
            src(i,j,k,m) = dst(i,j,k,m);
        }
    }
}
*/

void x_solve()
{
    int i, j, k, i1, i2, m;
    double ru1, rhon1, fac1, fac2;

    CUDAParameters cudaParams = setupDimensions(nx2, ny2, nz2, nx, ny, nz);

    if (timeron)
        timer_start(t_xsolve);

    x_solve_transpose<<<cudaParams.blockst, cudaParams.threadst>>>((double *)gpuTmp, (double *)gpuRhs, nx2, ny2, nz2);
    std::swap(gpuTmp, gpuRhs);
    x_solve_transpose_3D<<<cudaParams.blockst, cudaParams.threadst>>>((double *)gpuTmp3D, (double *)gpuUs, nx2, ny2, nz2);
    std::swap(gpuTmp3D, gpuUs);
    x_solve_transpose_3D<<<cudaParams.blockst, cudaParams.threadst>>>((double *)gpuTmp3D, (double *)gpuSpeed, nx2, ny2, nz2);
    std::swap(gpuTmp3D, gpuSpeed);
    x_solve_transpose_3D<<<cudaParams.blockst, cudaParams.threadst>>>((double *)gpuTmp3D, (double *)gpuRho_i, nx2, ny2, nz2);
    std::swap(gpuTmp3D, gpuRho_i);

    x_solve_two(
        cudaParams.blocks, cudaParams.threads,
        cudaParams.blocks2, cudaParams.threads2,
        (double *)lhs_gpu, (double *)lhsp_gpu, (double *)lhsm_gpu, (double *)gpuRhs, (double *)gpuRho_i, (double *)gpuUs, (double *)gpuSpeed, c3c4, dx2, con43, dx5, c1c5, dx1, dttx2, dttx1, dxmax, c2dttx1, comz1, comz4, comz5, comz6, nx2, ny2, nz2, nx);

    //---------------------------------------------------------------------
    // Do the block-diagonal inversion
    //---------------------------------------------------------------------
    if (timeron)
        timer_start(t_ninvr);

    x_solve_inversion<<<cudaParams.blocks, cudaParams.threads>>>((double *)gpuRhs, bt, nx2, ny2, nz2);

    if (timeron)
        timer_stop(t_ninvr);

    std::swap(gpuTmp3D, gpuUs);
    x_solve_inv_transpose_3D<<<cudaParams.blockst, cudaParams.threadst>>>((double *)gpuTmp3D, (double *)gpuUs, nx2, ny2, nz2);

    // std::swap((double**)&gpuTmp, (double**)&gpuRhs);

    // x_solve_inv_transpose<<<blockst, threadst>>>((double*)gpuTmp, (double*)gpuRhs, nx2, ny2, nz2);
    // cudaDeviceSynchronize();

    if (timeron)
        timer_stop(t_xsolve);
}

#undef rhs

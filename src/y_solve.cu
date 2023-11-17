#include "header.hpp"
#include <algorithm>

//---------------------------------------------------------------------
// this function performs the solution of the approximate factorization
// step in the y-direction for all five matrix components
// simultaneously. The Thomas algorithm is employed to solve the
// systems for the y-lines. Boundary conditions are non-periodic
//---------------------------------------------------------------------

void y_solve_one(
    dim3 blocks, dim3 threads,
    double *lhs_, double *lhsp_, double *lhsm_,
    int nx2, int ny2, int nz2)
{
    // reassign dimensions
    solve_kernel_one<<<blocks, threads>>>(lhs_, lhsp_, lhsm_, nz2, nx2, ny2);
}

void y_solve_two(
    dim3 blocks, dim3 threads,
    dim3 blocks2, dim3 threads2,
    double *lhs_, double *lhsp_, double *lhsm_, double *rhs, double *rho_i, double *vs, double *speed, double c3c4, double dy3, double con43, double dy5, double c1c5, double dy1, double dtty2, double dtty1, double dymax, double c2dtty1, double comz1, double comz4, double comz5, double comz6,
    int nx2, int ny2, int nz2, int ny)
{
    solve_kernel_two<<<blocks, threads>>>((double *)lhs_, (double *)lhsp_, (double *)lhsm_, (double *)rho_i, (double *)vs, (double *)speed, c3c4, dy3, con43, dy5, c1c5, dy1, dtty2, dtty1, dymax, c2dtty1, comz1, comz4, comz5, comz6, nz2, nx2, ny2);
    solve_kernel_two1<<<blocks2, threads2>>>((double *)lhs_, (double *)lhsp_, (double *)lhsm_, (double *)rho_i, (double *)vs, (double *)speed, c3c4, dy3, con43, dy5, c1c5, dy1, dtty2, dtty1, dymax, c2dtty1, comz1, comz4, comz5, comz6, nz2, nx2, ny2);
    solve_kernel_two2<<<blocks2, threads2>>>((double *)lhs_, (double *)lhsp_, (double *)lhsm_, (double *)rho_i, (double *)vs, (double *)speed, c3c4, dy3, con43, dy5, c1c5, dy1, dtty2, dtty1, dymax, c2dtty1, comz1, comz4, comz5, comz6, nz2, nx2, ny2);
    cudaDeviceSynchronize();
    solve_kernel_two_nz3<<<blocks2, threads2>>>((double *)lhs_, (double *)lhsp_, (double *)lhsm_, (double *)rho_i, (double *)vs, (double *)speed, c3c4, dy3, con43, dy5, c1c5, dy1, dtty2, dtty1, dymax, c2dtty1, comz1, comz4, comz5, comz6, nz2, nx2, ny2);
    solve_kernel_two_nz2<<<blocks2, threads2>>>((double *)lhs_, (double *)lhsp_, (double *)lhsm_, (double *)rho_i, (double *)vs, (double *)speed, c3c4, dy3, con43, dy5, c1c5, dy1, dtty2, dtty1, dymax, c2dtty1, comz1, comz4, comz5, comz6, nz2, nx2, ny2);
    cudaDeviceSynchronize();
    solve_kernel_three<<<blocks2, threads2>>>((double *)lhs_gpu, (double *)lhsp_gpu, (double *)lhsm_gpu, (double *)gpuRhs, (double *)gpuRho_i, (double *)gpuVs, (double *)gpuSpeed, c3c4, dy3, con43, dy5, c1c5, dy1, dtty2, dtty1, dymax, c2dtty1, comz1, comz4, comz5, comz6, nz2, nx2, ny2, ny);
    cudaDeviceSynchronize();
    solve_kernel_four<<<blocks2, threads2>>>((double *)lhs_gpu, (double *)lhsp_gpu, (double *)lhsm_gpu, (double *)gpuRhs, nz2, nx2, ny2);
}

#undef rhs
#define rhs(x, y, z, m) rhs[INDEX(x, y, z, m)]

#undef rho_i
#undef s
#undef speed
#define rho_i(x, y, z) rho_i[INDEX_3D(x, z, y)]
#define s(x, y, z) s[INDEX_3D(x, z, y)]
#define speed(x, y, z) speed[INDEX_3D(x, z, y)]

__global__ void y_solve_inversion(double *rhs, double bt, int nx2, int ny2, int nz2)
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

        t1 = bt * r1;
        t2 = 0.5 * (r4 + r5);

        rhs(k, j, i, 0) = bt * (r4 - r5);
        rhs(k, j, i, 1) = -r3;
        rhs(k, j, i, 2) = r2;
        rhs(k, j, i, 3) = -t1 + t2;
        rhs(k, j, i, 4) = t1 + t2;
    }
}

#define src(x, y, z) src[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE]
#define dst(x, y, z) dst[x + (z) * P_SIZE + (y) * P_SIZE * P_SIZE]
__global__ void y_solve_transpose_3D(double *dst, double *src, int nx2, int ny2, int nz2)
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

__global__ void y_solve_inv_transpose_3D(double *dst, double *src, int nx2, int ny2, int nz2)
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

void y_solve()
{
    int i, j, k, j1, j2, m;
    double ru1, rhoq1, fac1, fac2;

    const int size5 = sizeof(double) * P_SIZE * P_SIZE * P_SIZE * 5;
    const int size = sizeof(double) * P_SIZE * P_SIZE * P_SIZE;

    dim3 blocks = dim3(nx2 / 32 + 1, ny2, nz2);
    dim3 threads = dim3(32, 1, 1);

    dim3 blocks2 = dim3(nx2 / 32 + 1, nz2 / 8 + 1);
    dim3 threads2 = dim3(32, 8);

    dim3 blockst = dim3(nx / 8 + 1, ny / 8 + 1, nz / 8 + 1);
    dim3 threadst = dim3(8, 8, 8);

    if (timeron)
        timer_start(t_ysolve);

    y_solve_transpose_3D<<<blockst, threadst>>>((double *)gpuTmp3D, (double *)gpuVs, nx2, ny2, nz2);
    std::swap(gpuTmp3D, gpuVs);
    cudaDeviceSynchronize();
    y_solve_one(blocks2, threads2, (double *)lhs_gpu, (double *)lhsp_gpu, (double *)lhsm_gpu, nx2, ny2, nz2);

    cudaDeviceSynchronize();
    y_solve_two(
        blocks, threads,
        blocks2, threads2,
        (double *)lhs_gpu, (double *)lhsp_gpu, (double *)lhsm_gpu, (double *)gpuRhs, (double *)gpuRho_i, (double *)gpuVs, (double *)gpuSpeed, c3c4, dy3, con43, dy5, c1c5, dy1, dtty2, dtty1, dymax, c2dtty1, comz1, comz4, comz5, comz6, nx2, ny2, nz2, ny);

    //---------------------------------------------------------------------
    // block-diagonal matrix-vector multiplication
    //---------------------------------------------------------------------
    if (timeron)
        timer_start(t_pinvr);

    y_solve_inversion<<<blocks, threads>>>((double *)gpuRhs, bt, nx2, ny2, nz2);

    if (timeron)
        timer_stop(t_pinvr);

    std::swap(gpuTmp3D, gpuRho_i);
    y_solve_inv_transpose_3D<<<blockst, threadst>>>((double *)gpuTmp3D, (double *)gpuRho_i, nx2, ny2, nz2);
    std::swap(gpuTmp3D, gpuVs);
    y_solve_inv_transpose_3D<<<blockst, threadst>>>((double *)gpuTmp3D, (double *)gpuVs, nx2, ny2, nz2);
    std::swap(gpuTmp3D, gpuSpeed);
    y_solve_inv_transpose_3D<<<blockst, threadst>>>((double *)gpuTmp3D, (double *)gpuSpeed, nx2, ny2, nz2);

    if (timeron)
        timer_stop(t_ysolve);
}

#undef lhs

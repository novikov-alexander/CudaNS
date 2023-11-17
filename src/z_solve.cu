#include "header.hpp"
#include <algorithm>

//---------------------------------------------------------------------
// this function performs the solution of the approximate factorization
// step in the z-direction for all five matrix components
// simultaneously. The Thomas algorithm is employed to solve the
// systems for the z-lines. Boundary conditions are non-periodic
//---------------------------------------------------------------------

void z_solve_one(
    dim3 blocks, dim3 threads,
    double *lhs_, double *lhsp_, double *lhsm_,
    int nx2, int ny2, int nz2)
{
    // reassign dimensions
    solve_kernel_one<<<blocks, threads>>>(lhs_, lhsp_, lhsm_, ny2, nx2, nz2);
}

void z_solve_two(
    dim3 blocks, dim3 threads,
    dim3 blocks2, dim3 threads2,
    double *lhs_, double *lhsp_, double *lhsm_, double *rho_i, double *ws, double *speed, int nx2, int ny2, int nz2, double c3c4, double dz4, double con43, double dz5, double c1c5, double dzmax, double dz1, double dttz2, double dttz1, double c2dttz1, double comz1, double comz4, double comz5, double comz6)
{
    solve_kernel_two<<<blocks, threads>>>((double *)lhs_, (double *)lhsp_, (double *)lhsm_, (double *)rho_i, (double *)ws, (double *)speed, ny2, nx2, nz2, c3c4, dz4, con43, dz5, c1c5, dzmax, dz1, dttz2, dttz1, c2dttz1, comz1, comz4, comz5, comz6);
    solve_kernel_two1<<<blocks2, threads2>>>((double *)lhs_, (double *)lhsp_, (double *)lhsm_, (double *)rho_i, (double *)ws, (double *)speed, ny2, nx2, nz2, c3c4, dz4, con43, dz5, c1c5, dzmax, dz1, dttz2, dttz1, c2dttz1, comz1, comz4, comz5, comz6);
    solve_kernel_two2<<<blocks2, threads2>>>((double *)lhs_, (double *)lhsp_, (double *)lhsm_, (double *)rho_i, (double *)ws, (double *)speed, ny2, nx2, nz2, c3c4, dz4, con43, dz5, c1c5, dzmax, dz1, dttz2, dttz1, c2dttz1, comz1, comz4, comz5, comz6);
    cudaDeviceSynchronize();
    solve_kernel_two_nz3<<<blocks2, threads2>>>((double *)lhs_gpu, (double *)lhsp_gpu, (double *)lhsm_gpu, (double *)gpuRho_i, (double *)gpuWs, (double *)gpuSpeed, nx2, ny2, nz2, c3c4, dz4, con43, dz5, c1c5, dzmax, dz1, dttz2, dttz1, c2dttz1, comz1, comz4, comz5, comz6);
    solve_kernel_two_nz2<<<blocks2, threads2>>>((double *)lhs_, (double *)lhsp_, (double *)lhsm_, (double *)rho_i, (double *)ws, (double *)speed, ny2, nx2, nz2, c3c4, dz4, con43, dz5, c1c5, dzmax, dz1, dttz2, dttz1, c2dttz1, comz1, comz4, comz5, comz6);
    cudaDeviceSynchronize();
    solve_kernel_three<<<blocks2, threads2>>>((double *)lhs_, (double *)lhsp_, (double *)lhsm_, (double *)rhs, (double *)rho_i, (double *)ws, (double *)speed, ny2, nx2, nz2, c3c4, dz4, con43, dz5, c1c5, dzmax, dz1, dttz2, dttz1, c2dttz1, comz1, comz4, comz5, comz6, nz);
    cudaDeviceSynchronize();
    solve_kernel_four<<<blocks2, threads2>>>((double *)lhs_gpu, (double *)lhsp_gpu, (double *)lhsm_gpu, (double *)gpuRhs, ny2, nx2, nz2);
}

#undef rhs
#define rhs(x, y, z, m) rhs[INDEX(x, y, z, m)]

#undef s
#undef speed
#define s(x, y, z) s[INDEX_3D(y, z, x)]
#define speed(x, y, z) speed[INDEX_3D(y, z, x)]

__global__ void z_solve_inversion(double *rhs, double *us, double *vs, double *ws, double *qs, double *speed, double *u, int nx2, int ny2, int nz2, double bt, double c2iv)
{
    double t1, t2, t3, ac, xvel, yvel, zvel;
    double btuz, ac2u, uzik1, r1, r2, r3, r4, r5;

    int j = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int i = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1;

    if (i <= nx2 && j <= ny2 && k <= nz2)
    {
        xvel = us(k, j, i);
        yvel = vs(k, j, i);
        zvel = ws(k, j, i);
        ac = speed(k, j, i);

        ac2u = ac * ac;

        r1 = rhs(k, j, i, 0);
        r2 = rhs(k, j, i, 1);
        r3 = rhs(k, j, i, 2);
        r4 = rhs(k, j, i, 3);
        r5 = rhs(k, j, i, 4);

        uzik1 = u(k, j, i, 0);
        btuz = bt * uzik1;

        t1 = btuz / ac * (r4 + r5);
        t2 = r3 + t1;
        t3 = btuz * (r4 - r5);

        rhs(k, j, i, 0) = t2;
        rhs(k, j, i, 1) = -uzik1 * r2 + xvel * t2;
        rhs(k, j, i, 2) = uzik1 * r1 + yvel * t2;
        rhs(k, j, i, 3) = zvel * t2 + t3;
        rhs(k, j, i, 4) = uzik1 * (-xvel * r2 + yvel * r1) + qs(k, j, i) * t2 + c2iv * ac2u * t1 + zvel * t3;
    }
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

    const int size5 = sizeof(double) * P_SIZE * P_SIZE * P_SIZE * 5;
    const int size = sizeof(double) * P_SIZE * P_SIZE * P_SIZE;

    dim3 blocks = dim3(nx2 / 32 + 1, ny2, nz2);
    dim3 threads = dim3(32, 1, 1);

    dim3 blocks2 = dim3(nx2 / 32 + 1, ny2 / 8 + 1);
    dim3 threads2 = dim3(32, 8);

    dim3 blockst = dim3(nx / 8 + 1, ny / 8 + 1, nz / 8 + 1);
    dim3 threadst = dim3(8, 8, 8);

    dim3 blocks3 = dim3(nx2 / 32 + 1, ny2 / 8 + 1, nz2);
    dim3 threads3 = dim3(32, 8, 1);

    if (timeron)
        timer_start(t_zsolve);

    z_solve_transpose_3D<<<blockst, threadst>>>((double *)gpuTmp3D, (double *)gpuWs, nx2, ny2, nz2);
    z_solve_transpose<<<blockst, threadst>>>((double *)gpuTmp, (double *)gpuRhs, nx2, ny2, nz2);
    cudaDeviceSynchronize();
    std::swap(gpuTmp, gpuRhs);
    std::swap(gpuTmp3D, gpuWs);
    cudaDeviceSynchronize();

    z_solve_transpose_3D<<<blockst, threadst>>>((double *)gpuTmp3D, (double *)gpuSpeed, nx2, ny2, nz2);
    std::swap(gpuTmp3D, gpuSpeed);

    z_solve_one(blocks2, threads2, (double *)lhs_gpu, (double *)lhsp_gpu, (double *)lhsm_gpu, nx2, ny2, nz2);

    cudaDeviceSynchronize();
    z_solve_two(
        blocks, threads,
        blocks2, threads2,
        (double *)lhs_gpu, (double *)lhsp_gpu, (double *)lhsm_gpu, (double *)gpuRho_i, (double *)gpuWs, (double *)gpuSpeed, nx2, ny2, nz2, c3c4, dz4, con43, dz5, c1c5, dzmax, dz1, dttz2, dttz1, c2dttz1, comz1, comz4, comz5, comz6);

    //---------------------------------------------------------------------
    // block-diagonal matrix-vector multiplication
    //---------------------------------------------------------------------

    if (timeron)
        timer_start(t_tzetar);

    z_solve_inversion<<<blocks, threads>>>((double *)gpuRhs, (double *)gpuUs, (double *)gpuVs, (double *)gpuWs, (double *)gpuQs, (double *)gpuSpeed, (double *)gpuU, nx2, ny2, nz2, bt, c2iv);

    if (timeron)
        timer_stop(t_tzetar);

    std::swap(gpuTmp, gpuRhs);
    z_solve_inv_transpose<<<blockst, threadst>>>((double *)gpuTmp, (double *)gpuRhs, nx2, ny2, nz2);
    std::swap(gpuTmp3D, gpuWs);
    z_solve_inv_transpose_3D<<<blockst, threadst>>>((double *)gpuTmp3D, (double *)gpuWs, nx2, ny2, nz2);

    cudaDeviceSynchronize();

    std::swap(gpuTmp3D, gpuSpeed);
    z_solve_inv_transpose_3D<<<blockst, threadst>>>((double *)gpuTmp3D, (double *)gpuSpeed, nx2, ny2, nz2);

    if (timeron)
        timer_stop(t_zsolve);
}

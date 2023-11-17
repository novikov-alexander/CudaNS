#include "header.hpp"
#include <algorithm>

//---------------------------------------------------------------------
// this function performs the solution of the approximate factorization
// step in the x-direction for all five matrix components
// simultaneously. The Thomas algorithm is employed to solve the
// systems for the x-lines. Boundary conditions are non-periodic
//---------------------------------------------------------------------

void x_solve_one(
    dim3 blocks, dim3 threads,
    double *lhs_, double *lhsp_, double *lhsm_,
    int nx2, int ny2, int nz2)
{
    // reassign x- and z- dimensions
    solve_kernel_one<<<blocks, threads>>>(lhs_, lhsp_, lhsm_, nz2, ny2, nx2);
}
void x_solve_two(
    dim3 blocks, dim3 threads,
    dim3 blocks2, dim3 threads2,
    double *lhs_, double *lhsp_, double *lhsm_, double *rhs, double *rho_i, double *us, double *speed, double c3c4, double dx2, double con43, double dx5, double c1c5, double dx1, double dttx2, double dttx1, double dxmax, double c2dttx1, double comz1, double comz4, double comz5, double comz6,
    int nx2, int ny2, int nz2, int nx)
{
    solve_kernel_two<<<blocks, threads>>>((double *)lhs_, (double *)lhsp_, (double *)lhsm_, (double *)rho_i, (double *)us, (double *)speed, c3c4, dy3, con43, dy5, c1c5, dy1, dtty2, dtty1, dymax, c2dtty1, comz1, comz4, comz5, comz6, nz2, ny2, nx2);
    solve_kernel_two1<<<blocks2, threads2>>>((double *)lhs_, (double *)lhsp_, (double *)lhsm_, (double *)rho_i, (double *)us, (double *)speed, c3c4, dy3, con43, dy5, c1c5, dy1, dtty2, dtty1, dymax, c2dtty1, comz1, comz4, comz5, comz6, nz2, ny2, nx2);
    solve_kernel_two2<<<blocks2, threads2>>>((double *)lhs_, (double *)lhsp_, (double *)lhsm_, (double *)rho_i, (double *)us, (double *)speed, c3c4, dy3, con43, dy5, c1c5, dy1, dtty2, dtty1, dymax, c2dtty1, comz1, comz4, comz5, comz6, nz2, ny2, nx2);
    cudaDeviceSynchronize();
    solve_kernel_two_nz2<<<blocks2, threads2>>>((double *)lhs_, (double *)lhsp_, (double *)lhsm_, (double *)rho_i, (double *)us, (double *)speed, c3c4, dy3, con43, dy5, c1c5, dy1, dtty2, dtty1, dymax, c2dtty1, comz1, comz4, comz5, comz6, nz2, ny2, nx2);
}

#undef rhs
#define rhs(x, y, z, m) rhs[INDEX(x, y, z, m)]

#undef rho_i
#undef s
#undef speed
#define rho_i(x, y, z) rho_i[INDEX_3D(x, z, y)]
#define s(x, y, z) s[INDEX_3D(x, z, y)]
#define speed(x, y, z) speed[INDEX_3D(x, z, y)]
__global__ void x_solve_kernel_two_nx3(double *lhs_, double *lhsp_, double *lhsm_, double *rhs, double *rho_i, double *us, double *speed, double c3c4, double dx2, double con43, double dx5, double c1c5, double dx1, double dttx2, double dttx1, double dxmax, double c2dttx1, double comz1, double comz4, double comz5, double comz6, int nx2, int ny2, int nz2, int nx)
{
    int i1, i2, m;
    double ru1, rhon1;

    int k = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int i = nx - 3;

    // part 2
    if (k <= nz2 && j <= ny2)
    {
        lhs_(k, j, i, 0) = 0.0;
        ru1 = c3c4 * rho_i(k, j, i - 1);
        rhon1 = fmax(fmax(dx2 + con43 * ru1, dx5 + c1c5 * ru1), fmax(dxmax + ru1, dx1));
        lhs_(k, j, i, 1) = -dttx2 * us(k, j, i - 1) - dttx1 * rhon1;

        ru1 = c3c4 * rho_i(k, j, i);
        rhon1 = fmax(fmax(dx2 + con43 * ru1, dx5 + c1c5 * ru1), fmax(dxmax + ru1, dx1));
        lhs_(k, j, i, 2) = 1.0 + c2dttx1 * rhon1;

        ru1 = c3c4 * rho_i(k, j, i + 1);
        rhon1 = fmax(fmax(dx2 + con43 * ru1, dx5 + c1c5 * ru1), fmax(dxmax + ru1, dx1));
        lhs_(k, j, i, 3) = dttx2 * us(k, j, i + 1) - dttx1 * rhon1;
        lhs_(k, j, i, 4) = 0.0;

        lhs_(k, j, i, 0) = lhs_(k, j, i, 0) + comz1;
        lhs_(k, j, i, 1) = lhs_(k, j, i, 1) - comz4;
        lhs_(k, j, i, 2) = lhs_(k, j, i, 2) + comz6;
        lhs_(k, j, i, 3) = lhs_(k, j, i, 3) - comz4;

        lhsp_(k, j, i, 0) = lhs_(k, j, i, 0);
        lhsp_(k, j, i, 1) = lhs_(k, j, i, 1) - dttx2 * speed(k, j, i - 1);
        lhsp_(k, j, i, 2) = lhs_(k, j, i, 2);
        lhsp_(k, j, i, 3) = lhs_(k, j, i, 3) + dttx2 * speed(k, j, i + 1);
        lhsp_(k, j, i, 4) = lhs_(k, j, i, 4);

        lhsm_(k, j, i, 0) = lhs_(k, j, i, 0);
        lhsm_(k, j, i, 1) = lhs_(k, j, i, 1) + dttx2 * speed(k, j, i - 1);
        lhsm_(k, j, i, 2) = lhs_(k, j, i, 2);
        lhsm_(k, j, i, 3) = lhs_(k, j, i, 3) - dttx2 * speed(k, j, i + 1);
        lhsm_(k, j, i, 4) = lhs_(k, j, i, 4);
    }
}

__global__ void x_solve_kernel_three(double *lhs_, double *lhsp_, double *lhsm_, double *rhs, double *rho_i, double *us, double *speed, double c3c4, double dx2, double con43, double dx5, double c1c5, double dx1, double dttx2, double dttx1, double dxmax, double c2dttx1, double comz1, double comz4, double comz5, double comz6, int nx2, int ny2, int nz2, int nx)
{
    int i1, i2, m;
    double ru1, rhon1, fac1, fac2;

    int i;
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    // part 3
    if (k > 0 && k <= nz2 && j > 0 && j <= ny2)
    {
        for (i = 1; i <= nx2; i++)
        {
            i1 = i;
            i2 = i + 1;
            fac1 = 1.0 / lhs_(k, j, i - 1, 2);
            lhs_(k, j, i - 1, 3) = fac1 * lhs_(k, j, i - 1, 3);
            lhs_(k, j, i - 1, 4) = fac1 * lhs_(k, j, i - 1, 4);

#pragma unroll 3
            for (m = 0; m < 3; m++)
                rhs(k, j, i - 1, m) = fac1 * rhs(k, j, i - 1, m);

            lhs_(k, j, i1, 2) = lhs_(k, j, i1, 2) - lhs_(k, j, i1, 1) * lhs_(k, j, i - 1, 3);
            lhs_(k, j, i1, 3) = lhs_(k, j, i1, 3) - lhs_(k, j, i1, 1) * lhs_(k, j, i - 1, 4);
#pragma unroll 3
            for (m = 0; m < 3; m++)
                rhs(k, j, i1, m) = rhs(k, j, i1, m) - lhs_(k, j, i1, 1) * rhs(k, j, i - 1, m);

            lhs_(k, j, i2, 1) = lhs_(k, j, i2, 1) - lhs_(k, j, i2, 0) * lhs_(k, j, i - 1, 3);
            lhs_(k, j, i2, 2) = lhs_(k, j, i2, 2) - lhs_(k, j, i2, 0) * lhs_(k, j, i - 1, 4);
#pragma unroll 3
            for (m = 0; m < 3; m++)
                rhs(k, j, i2, m) = rhs(k, j, i2, m) - lhs_(k, j, i2, 0) * rhs(k, j, i - 1, m);

            if (i == nx2)
            {
                fac1 = 1.0 / lhs_(k, j, i1, 2);
                lhs_(k, j, i1, 3) = fac1 * lhs_(k, j, i1, 3);
                lhs_(k, j, i1, 4) = fac1 * lhs_(k, j, i1, 4);
#pragma unroll 3
                for (m = 0; m < 3; m++)
                    rhs(k, j, i1, m) = fac1 * rhs(k, j, i1, m);

                lhs_(k, j, i2, 2) = lhs_(k, j, i2, 2) - lhs_(k, j, i2, 1) * lhs_(k, j, i1, 3);
                lhs_(k, j, i2, 3) = lhs_(k, j, i2, 3) - lhs_(k, j, i2, 1) * lhs_(k, j, i1, 4);
#pragma unroll 3
                for (m = 0; m < 3; m++)
                    rhs(k, j, i2, m) = rhs(k, j, i2, m) - lhs_(k, j, i2, 1) * rhs(k, j, i1, m);

                fac2 = 1.0 / lhs_(k, j, i2, 2);
#pragma unroll 3
                for (m = 0; m < 3; m++)
                    rhs(k, j, i2, m) = fac2 * rhs(k, j, i2, m);
            }

            m = 3;
            fac1 = 1.0 / lhsp_(k, j, i - 1, 2);
            lhsp_(k, j, i - 1, 3) = fac1 * lhsp_(k, j, i - 1, 3);
            lhsp_(k, j, i - 1, 4) = fac1 * lhsp_(k, j, i - 1, 4);
            rhs(k, j, i - 1, m) = fac1 * rhs(k, j, i - 1, m);

            lhsp_(k, j, i1, 2) = lhsp_(k, j, i1, 2) - lhsp_(k, j, i1, 1) * lhsp_(k, j, i - 1, 3);
            lhsp_(k, j, i1, 3) = lhsp_(k, j, i1, 3) - lhsp_(k, j, i1, 1) * lhsp_(k, j, i - 1, 4);
            rhs(k, j, i1, m) = rhs(k, j, i1, m) - lhsp_(k, j, i1, 1) * rhs(k, j, i - 1, m);

            lhsp_(k, j, i2, 1) = lhsp_(k, j, i2, 1) - lhsp_(k, j, i2, 0) * lhsp_(k, j, i - 1, 3);
            lhsp_(k, j, i2, 2) = lhsp_(k, j, i2, 2) - lhsp_(k, j, i2, 0) * lhsp_(k, j, i - 1, 4);
            rhs(k, j, i2, m) = rhs(k, j, i2, m) - lhsp_(k, j, i2, 0) * rhs(k, j, i - 1, m);

            m = 4;
            fac1 = 1.0 / lhsm_(k, j, i - 1, 2);
            lhsm_(k, j, i - 1, 3) = fac1 * lhsm_(k, j, i - 1, 3);
            lhsm_(k, j, i - 1, 4) = fac1 * lhsm_(k, j, i - 1, 4);
            rhs(k, j, i - 1, m) = fac1 * rhs(k, j, i - 1, m);
            lhsm_(k, j, i1, 2) = lhsm_(k, j, i1, 2) - lhsm_(k, j, i1, 1) * lhsm_(k, j, i - 1, 3);
            lhsm_(k, j, i1, 3) = lhsm_(k, j, i1, 3) - lhsm_(k, j, i1, 1) * lhsm_(k, j, i - 1, 4);
            rhs(k, j, i1, m) = rhs(k, j, i1, m) - lhsm_(k, j, i1, 1) * rhs(k, j, i - 1, m);
            lhsm_(k, j, i2, 1) = lhsm_(k, j, i2, 1) - lhsm_(k, j, i2, 0) * lhsm_(k, j, i - 1, 3);
            lhsm_(k, j, i2, 2) = lhsm_(k, j, i2, 2) - lhsm_(k, j, i2, 0) * lhsm_(k, j, i - 1, 4);
            rhs(k, j, i2, m) = rhs(k, j, i2, m) - lhsm_(k, j, i2, 0) * rhs(k, j, i - 1, m);

            if (i == nx2)
            {
                m = 3;
                fac1 = 1.0 / lhsp_(k, j, i1, 2);
                lhsp_(k, j, i1, 3) = fac1 * lhsp_(k, j, i1, 3);
                lhsp_(k, j, i1, 4) = fac1 * lhsp_(k, j, i1, 4);
                rhs(k, j, i1, m) = fac1 * rhs(k, j, i1, m);

                lhsp_(k, j, i2, 2) = lhsp_(k, j, i2, 2) - lhsp_(k, j, i2, 1) * lhsp_(k, j, i1, 3);
                lhsp_(k, j, i2, 3) = lhsp_(k, j, i2, 3) - lhsp_(k, j, i2, 1) * lhsp_(k, j, i1, 4);
                rhs(k, j, i2, m) = rhs(k, j, i2, m) - lhsp_(k, j, i2, 1) * rhs(k, j, i1, m);

                m = 4;
                fac1 = 1.0 / lhsm_(k, j, i1, 2);
                lhsm_(k, j, i1, 3) = fac1 * lhsm_(k, j, i1, 3);
                lhsm_(k, j, i1, 4) = fac1 * lhsm_(k, j, i1, 4);
                rhs(k, j, i1, m) = fac1 * rhs(k, j, i1, m);

                lhsm_(k, j, i2, 2) = lhsm_(k, j, i2, 2) - lhsm_(k, j, i2, 1) * lhsm_(k, j, i1, 3);
                lhsm_(k, j, i2, 3) = lhsm_(k, j, i2, 3) - lhsm_(k, j, i2, 1) * lhsm_(k, j, i1, 4);
                rhs(k, j, i2, m) = rhs(k, j, i2, m) - lhsm_(k, j, i2, 1) * rhs(k, j, i1, m);

                rhs(k, j, i2, 3) = rhs(k, j, i2, 3) / lhsp_(k, j, i2, 2);
                rhs(k, j, i2, 4) = rhs(k, j, i2, 4) / lhsm_(k, j, i2, 2);

#pragma unroll 3
                for (m = 0; m < 3; m++)
                    rhs(k, j, i1, m) = rhs(k, j, i1, m) - lhs_(k, j, i1, 3) * rhs(k, j, i2, m);

                rhs(k, j, i1, 3) = rhs(k, j, i1, 3) - lhsp_(k, j, i1, 3) * rhs(k, j, i2, 3);
                rhs(k, j, i1, 4) = rhs(k, j, i1, 4) - lhsm_(k, j, i1, 3) * rhs(k, j, i2, 4);
            }
        }
    }
}

__global__ void x_solve_kernel_four(double *lhs_, double *lhsp_, double *lhsm_, double *rhs, double *rho_i, double *us, double *speed, double c3c4, double dx2, double con43, double dx5, double c1c5, double dx1, double dttx2, double dttx1, double dxmax, double c2dttx1, double comz1, double comz4, double comz5, double comz6, int nx2, int ny2, int nz2, int nx)
{
    int i1, i2, m, i;

    int k = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;

    // part 4
    if ((k <= nz2) && (j <= ny2))
    {
        for (i = nx2; i >= 1; i--)
        {
            i1 = i;
            i2 = i + 1;
            for (m = 0; m < 3; m++)
                rhs(k, j, i - 1, m) = rhs(k, j, i - 1, m) - lhs_(k, j, i - 1, 3) * rhs(k, j, i1, m) - lhs_(k, j, i - 1, 4) * rhs(k, j, i2, m);

            rhs(k, j, i - 1, 3) = rhs(k, j, i - 1, 3) - lhsp_(k, j, i - 1, 3) * rhs(k, j, i1, 3) - lhsp_(k, j, i - 1, 4) * rhs(k, j, i2, 3);
            rhs(k, j, i - 1, 4) = rhs(k, j, i - 1, 4) - lhsm_(k, j, i - 1, 3) * rhs(k, j, i1, 4) - lhsm_(k, j, i - 1, 4) * rhs(k, j, i2, 4);
        }
    }
}

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

    dim3 blocks = dim3(nx2 / 32 + 1, ny2, nz2);
    dim3 threads = dim3(32, 1, 1);

    dim3 blocks2 = dim3(ny2 / 32 + 1, nz2 / 8 + 1);
    dim3 threads2 = dim3(32, 8);

    dim3 blockst = dim3(nx / 8 + 1, ny / 8 + 1, nz / 8 + 1);
    dim3 threadst = dim3(8, 8, 8);

    if (timeron)
        timer_start(t_xsolve);

    x_solve_transpose<<<blockst, threadst>>>((double *)gpuTmp, (double *)gpuRhs, nx2, ny2, nz2);
    std::swap(gpuTmp, gpuRhs);
    x_solve_transpose_3D<<<blockst, threadst>>>((double *)gpuTmp3D, (double *)gpuUs, nx2, ny2, nz2);
    std::swap(gpuTmp3D, gpuUs);
    x_solve_transpose_3D<<<blockst, threadst>>>((double *)gpuTmp3D, (double *)gpuSpeed, nx2, ny2, nz2);
    std::swap(gpuTmp3D, gpuSpeed);
    x_solve_transpose_3D<<<blockst, threadst>>>((double *)gpuTmp3D, (double *)gpuRho_i, nx2, ny2, nz2);
    std::swap(gpuTmp3D, gpuRho_i);

    x_solve_one(blocks2, threads2, (double *)lhs_gpu, (double *)lhsp_gpu, (double *)lhsm_gpu, nz2, ny2, nx2);

    cudaDeviceSynchronize();
    x_solve_two(
        blocks, threads,
        blocks2, threads2,
        (double *)lhs_gpu, (double *)lhsp_gpu, (double *)lhsm_gpu, (double *)gpuRhs, (double *)gpuRho_i, (double *)gpuUs, (double *)gpuSpeed, c3c4, dx2, con43, dx5, c1c5, dx1, dttx2, dttx1, dxmax, c2dttx1, comz1, comz4, comz5, comz6, nx2, ny2, nz2, nx);
    x_solve_kernel_two_nx3<<<blocks2, threads2>>>((double *)lhs_gpu, (double *)lhsp_gpu, (double *)lhsm_gpu, (double *)gpuRhs, (double *)gpuRho_i, (double *)gpuUs, (double *)gpuSpeed, c3c4, dx2, con43, dx5, c1c5, dx1, dttx2, dttx1, dxmax, c2dttx1, comz1, comz4, comz5, comz6, nx2, ny2, nz2, nx);

    cudaDeviceSynchronize();
    x_solve_kernel_three<<<blocks2, threads2>>>((double *)lhs_gpu, (double *)lhsp_gpu, (double *)lhsm_gpu, (double *)gpuRhs, (double *)gpuRho_i, (double *)gpuUs, (double *)gpuSpeed, c3c4, dx2, con43, dx5, c1c5, dx1, dttx2, dttx1, dxmax, c2dttx1, comz1, comz4, comz5, comz6, nx2, ny2, nz2, nx);
    cudaDeviceSynchronize();

    x_solve_kernel_four<<<blocks2, threads2>>>((double *)lhs_gpu, (double *)lhsp_gpu, (double *)lhsm_gpu, (double *)gpuRhs, (double *)gpuRho_i, (double *)gpuUs, (double *)gpuSpeed, c3c4, dx2, con43, dx5, c1c5, dx1, dttx2, dttx1, dxmax, c2dttx1, comz1, comz4, comz5, comz6, nx2, ny2, nz2, nx);
    cudaDeviceSynchronize();

    //---------------------------------------------------------------------
    // Do the block-diagonal inversion
    //---------------------------------------------------------------------
    if (timeron)
        timer_start(t_ninvr);

    x_solve_inversion<<<blocks, threads>>>((double *)gpuRhs, bt, nx2, ny2, nz2);

    if (timeron)
        timer_stop(t_ninvr);

    std::swap(gpuTmp3D, gpuUs);
    x_solve_inv_transpose_3D<<<blockst, threadst>>>((double *)gpuTmp3D, (double *)gpuUs, nx2, ny2, nz2);

    // std::swap((double**)&gpuTmp, (double**)&gpuRhs);

    // x_solve_inv_transpose<<<blockst, threadst>>>((double*)gpuTmp, (double*)gpuRhs, nx2, ny2, nz2);
    // cudaDeviceSynchronize();

    if (timeron)
        timer_stop(t_xsolve);
}

#undef rhs

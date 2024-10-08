#include "header.hpp"

__global__ void inversion_kernel(double *rhs, double bt, int nx2, int ny2, int nz2)
{
    int i = threadIdx.z + blockIdx.z * blockDim.z + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int k = threadIdx.x + blockIdx.x * blockDim.x + 1;

    double r1, r2, r3, r4, r5, t1, t2;

    if ((i <= nx2) && (j <= ny2) && (k <= nz2))
    {
        r1 = rhs(i, j, k, 0);
        r2 = rhs(i, j, k, 1);
        r3 = rhs(i, j, k, 2);
        r4 = rhs(i, j, k, 3);
        r5 = rhs(i, j, k, 4);

        t1 = bt * r3;
        t2 = 0.5 * (r4 + r5);

        rhs(i, j, k, 0) = -r2;
        rhs(i, j, k, 1) = r1;
        rhs(i, j, k, 2) = bt * (r4 - r5);
        rhs(i, j, k, 3) = -t1 + t2;
        rhs(i, j, k, 4) = t1 + t2;
    }
}

void run_inversion_kernels(dim3 blocks, dim3 threads, double *rhs, double bt, int nx2, int ny2, int nz2)
{
    inversion_kernel<<<blocks, threads>>>(rhs, bt, nx2, ny2, nz2);
}

__global__ void solve_kernel_one(double *lhs_, double *lhsp_, double *lhsm_, int nx2, int ny2, int nz2)
{
    int m;

    int i = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int j = threadIdx.x + blockIdx.x * blockDim.x + 1;

    // part 1
    if (i <= nx2 && j <= ny2)
    {
#pragma unroll 5
        for (m = 0; m < 5; m++)
        {
            lhs_(i, j, 0, m) = lhs_(i, j, nz2 + 1, m) = 0.0;
            lhsp_(i, j, 0, m) = lhsp_(i, j, nz2 + 1, m) = 0.0;
            lhsm_(i, j, 0, m) = lhsm_(i, j, nz2 + 1, m) = 0.0;
        }

        lhs_(i, j, 0, 2) = lhs_(i, j, nz2 + 1, 2) = 1.0;
        lhsp_(i, j, 0, 2) = lhsp_(i, j, nz2 + 1, 2) = 1.0;
        lhsm_(i, j, 0, 2) = lhsm_(i, j, nz2 + 1, 2) = 1.0;
    }
}

__device__ inline void update_lhs(int i, int j, int k, double *us, double &lhs, const double ru1, double factor1, double factor2, double factor3,
    double dz1, double dz4, double dz5, double dzmax, double c1c5, double con43)
{
    lhs = factor1 * ru1;
    auto rhos1 = fmax(fmax(dz4 + con43 * ru1, dz5 + c1c5 * ru1), fmax(dzmax + ru1, dz1));
    lhs = factor2 * us(k - 1, i, j) - factor3 * rhos1;
}

__device__ inline void update_lhs_values(int i, int j, int k, double *rho_i, double *us, double &lhs_0, double &lhs_1, double &lhs_2, double &lhs_3, double &lhs_4,
    double dz1, double dz4, double dz5, double dzmax, double c1c5, double con43,
    double c3c4, double dttz1, double dttz2, double c2dttz1)
{
    lhs_0 = 0.0;

    update_lhs(i, j, k, us, lhs_1, c3c4 * rho_i(k - 1, i, j), -dttz2, -dttz1, 0.0,dz1,dz4,dz5,dzmax,c1c5,con43);
    update_lhs(i, j, k, us, lhs_2, c3c4 * rho_i(k, i, j), 1.0, c2dttz1, 0.0,dz1,dz4,dz5,dzmax,c1c5,con43);
    update_lhs(i, j, k, us, lhs_3, c3c4 * rho_i(k + 1, i, j), dttz2, -dttz1, 0.0,dz1,dz4,dz5,dzmax,c1c5,con43);

    lhs_4 = 0.0;
}

#undef us
#undef speed
#define us(x, y, z) us[INDEX_3D(y, z, x)]
#define speed(x, y, z) speed[INDEX_3D(y, z, x)]
__global__ void solve_kernel_two1(
    double *lhs_, double *lhsp_, double *lhsm_,
    double *rho_i, double *us, double *speed,
    int nx2, int ny2, int nz2,
    double c3c4, double dz4, double con43, double dz5,
    double c1c5, double dzmax, double dz1, double dttz2, double dttz1,
    double c2dttz1, double comz1, double comz4, double comz5, double comz6)
{
    int m;
    double ru1, rhos1;

    int i = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int j = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int k = 1;

    if (j <= ny2 && i <= nx2)
    {
        update_lhs_values(i, j, k, rho_i, us, lhs_(i, j, k, 0), lhs_(i, j, k, 1), lhs_(i, j, k, 2), lhs_(i, j, k, 3), lhs_(i, j, k, 4),dz1,dz4,dz5,dzmax,c1c5,con43,c3c4,dttz1,dttz2,c2dttz1);

        lhs_(i, j, k, 1) = lhs_(i, j, k, 1) - comz4;
        lhs_(i, j, k, 2) = lhs_(i, j, k, 2) + comz6;
        lhs_(i, j, k, 3) = lhs_(i, j, k, 3) - comz4;
        lhs_(i, j, k, 4) = lhs_(i, j, k, 4) + comz1;

        lhsp_(i, j, k, 0) = lhs_(i, j, k, 0);
        lhsp_(i, j, k, 1) = lhs_(i, j, k, 1) - dttz2 * speed(k - 1, i, j);
        lhsp_(i, j, k, 2) = lhs_(i, j, k, 2);
        lhsp_(i, j, k, 3) = lhs_(i, j, k, 3) + dttz2 * speed(k + 1, i, j);
        lhsp_(i, j, k, 4) = lhs_(i, j, k, 4);
        lhsm_(i, j, k, 0) = lhs_(i, j, k, 0);
        lhsm_(i, j, k, 1) = lhs_(i, j, k, 1) + dttz2 * speed(k - 1, i, j);
        lhsm_(i, j, k, 2) = lhs_(i, j, k, 2);
        lhsm_(i, j, k, 3) = lhs_(i, j, k, 3) - dttz2 * speed(k + 1, i, j);
        lhsm_(i, j, k, 4) = lhs_(i, j, k, 4);
    }
};

__global__ void solve_kernel_two2(
    double *lhs_, double *lhsp_, double *lhsm_,
    double *rho_i, double *us, double *speed,
    int nx2, int ny2, int nz2,
    double c3c4, double dz4, double con43, double dz5,
    double c1c5, double dzmax, double dz1, double dttz2, double dttz1,
    double c2dttz1, double comz1, double comz4, double comz5, double comz6)
{
    int m;
    double ru1, rhos1;

    int i = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int j = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int k = 2;

    if (j <= ny2 && i <= nx2)
    {
        update_lhs_values(i, j, k, rho_i, us, lhs_(i, j, k, 0), lhs_(i, j, k, 1), lhs_(i, j, k, 2), lhs_(i, j, k, 3), lhs_(i, j, k, 4),dz1,dz4,dz5,dzmax,c1c5,con43,c3c4,dttz1,dttz2,c2dttz1);

        lhs_(i, j, k, 1) = lhs_(i, j, k, 1) - comz4;
        lhs_(i, j, k, 2) = lhs_(i, j, k, 2) + comz5;
        lhs_(i, j, k, 3) = lhs_(i, j, k, 3) - comz4;
        lhs_(i, j, k, 4) = lhs_(i, j, k, 4) + comz1;

        lhsp_(i, j, k, 0) = lhs_(i, j, k, 0);
        lhsp_(i, j, k, 1) = lhs_(i, j, k, 1) - dttz2 * speed(k - 1, i, j);
        lhsp_(i, j, k, 2) = lhs_(i, j, k, 2);
        lhsp_(i, j, k, 3) = lhs_(i, j, k, 3) + dttz2 * speed(k + 1, i, j);
        lhsp_(i, j, k, 4) = lhs_(i, j, k, 4);
        lhsm_(i, j, k, 0) = lhs_(i, j, k, 0);
        lhsm_(i, j, k, 1) = lhs_(i, j, k, 1) + dttz2 * speed(k - 1, i, j);
        lhsm_(i, j, k, 2) = lhs_(i, j, k, 2);
        lhsm_(i, j, k, 3) = lhs_(i, j, k, 3) - dttz2 * speed(k + 1, i, j);
        lhsm_(i, j, k, 4) = lhs_(i, j, k, 4);
    }
};

__global__ void solve_kernel_two_nz2(
    double *lhs_, double *lhsp_, double *lhsm_,
    double *rho_i, double *us, double *speed,
    int nx2, int ny2, int nz2,
    double c3c4, double dz4, double con43, double dz5,
    double c1c5, double dzmax, double dz1, double dttz2, double dttz1,
    double c2dttz1, double comz1, double comz4, double comz5, double comz6, double nz)
{
    int m;
    double ru1, rhos1;

    int i = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int j = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int k = nz - 2;

    if (j <= ny2 && i <= nx2)
    {
        update_lhs_values(i, j, k, rho_i, us, lhs_(i, j, k, 0), lhs_(i, j, k, 1), lhs_(i, j, k, 2), lhs_(i, j, k, 3), lhs_(i, j, k, 4),dz1,dz4,dz5,dzmax,c1c5,con43,c3c4,dttz1,dttz2,c2dttz1);

        lhs_(i, j, k, 0) = lhs_(i, j, k, 0) + comz1;
        lhs_(i, j, k, 1) = lhs_(i, j, k, 1) - comz4;
        lhs_(i, j, k, 2) = lhs_(i, j, k, 2) + comz5;

        lhsp_(i, j, k, 0) = lhs_(i, j, k, 0);
        lhsp_(i, j, k, 1) = lhs_(i, j, k, 1) - dttz2 * speed(k - 1, i, j);
        lhsp_(i, j, k, 2) = lhs_(i, j, k, 2);
        lhsp_(i, j, k, 3) = lhs_(i, j, k, 3) + dttz2 * speed(k + 1, i, j);
        lhsp_(i, j, k, 4) = lhs_(i, j, k, 4);
        lhsm_(i, j, k, 0) = lhs_(i, j, k, 0);
        lhsm_(i, j, k, 1) = lhs_(i, j, k, 1) + dttz2 * speed(k - 1, i, j);
        lhsm_(i, j, k, 2) = lhs_(i, j, k, 2);
        lhsm_(i, j, k, 3) = lhs_(i, j, k, 3) - dttz2 * speed(k + 1, i, j);
        lhsm_(i, j, k, 4) = lhs_(i, j, k, 4);
    }
};

__global__ void solve_kernel_two_nz3(
    double *lhs_, double *lhsp_, double *lhsm_,
    double *rho_i, double *us, double *speed,
    int nx2, int ny2, int nz2,
    double c3c4, double dz4, double con43, double dz5,
    double c1c5, double dzmax, double dz1, double dttz2, double dttz1,
    double c2dttz1, double comz1, double comz4, double comz5, double comz6, double nz)
{
    int m;
    double ru1, rhos1;

    int i = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int j = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int k = nz - 3;

    if (j <= ny2 && i <= nx2)
    {
        update_lhs_values(i, j, k, rho_i, us, lhs_(i, j, k, 0), lhs_(i, j, k, 1), lhs_(i, j, k, 2), lhs_(i, j, k, 3), lhs_(i, j, k, 4),dz1,dz4,dz5,dzmax,c1c5,con43,c3c4,dttz1,dttz2,c2dttz1);

        lhs_(i, j, k, 0) = lhs_(i, j, k, 0) + comz1;
        lhs_(i, j, k, 1) = lhs_(i, j, k, 1) - comz4;
        lhs_(i, j, k, 2) = lhs_(i, j, k, 2) + comz5;
        lhs_(i, j, k, 3) = lhs_(i, j, k, 3) - comz4;

        lhsp_(i, j, k, 0) = lhs_(i, j, k, 0);
        lhsp_(i, j, k, 1) = lhs_(i, j, k, 1) - dttz2 * speed(k - 1, i, j);
        lhsp_(i, j, k, 2) = lhs_(i, j, k, 2);
        lhsp_(i, j, k, 3) = lhs_(i, j, k, 3) + dttz2 * speed(k + 1, i, j);
        lhsp_(i, j, k, 4) = lhs_(i, j, k, 4);
        lhsm_(i, j, k, 0) = lhs_(i, j, k, 0);
        lhsm_(i, j, k, 1) = lhs_(i, j, k, 1) + dttz2 * speed(k - 1, i, j);
        lhsm_(i, j, k, 2) = lhs_(i, j, k, 2);
        lhsm_(i, j, k, 3) = lhs_(i, j, k, 3) - dttz2 * speed(k + 1, i, j);
        lhsm_(i, j, k, 4) = lhs_(i, j, k, 4);
    }
};

__global__ void solve_kernel_two(
    double *lhs_, double *lhsp_, double *lhsm_,
    double *rho_i, double *us, double *speed,
    int nx2, int ny2, int nz2,
    double c3c4, double dz4, double con43, double dz5,
    double c1c5, double dzmax, double dz1, double dttz2, double dttz1,
    double c2dttz1, double comz1, double comz4, double comz5, double comz6)
{
    int m;
    double ru1, rhos1;

    int i = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int j = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 3;

    if (j <= ny2 && i <= nx2 && (k <= nz2 - 2))
    {
        update_lhs_values(i, j, k, rho_i, us, lhs_(i, j, k, 0), lhs_(i, j, k, 1), lhs_(i, j, k, 2), lhs_(i, j, k, 3), lhs_(i, j, k, 4),dz1,dz4,dz5,dzmax,c1c5,con43,c3c4,dttz1,dttz2,c2dttz1);

        lhs_(i, j, k, 0) = lhs_(i, j, k, 0) + comz1;

        lhs_(i, j, k, 1) = lhs_(i, j, k, 1) - comz4;
        lhs_(i, j, k, 2) = lhs_(i, j, k, 2) + comz6;
        lhs_(i, j, k, 3) = lhs_(i, j, k, 3) - comz4;
        lhs_(i, j, k, 4) = lhs_(i, j, k, 4) + comz1;

        lhsp_(i, j, k, 0) = lhs_(i, j, k, 0);
        lhsp_(i, j, k, 1) = lhs_(i, j, k, 1) - dttz2 * speed(k - 1, i, j);
        lhsp_(i, j, k, 2) = lhs_(i, j, k, 2);
        lhsp_(i, j, k, 3) = lhs_(i, j, k, 3) + dttz2 * speed(k + 1, i, j);
        lhsp_(i, j, k, 4) = lhs_(i, j, k, 4);
        lhsm_(i, j, k, 0) = lhs_(i, j, k, 0);
        lhsm_(i, j, k, 1) = lhs_(i, j, k, 1) + dttz2 * speed(k - 1, i, j);
        lhsm_(i, j, k, 2) = lhs_(i, j, k, 2);
        lhsm_(i, j, k, 3) = lhs_(i, j, k, 3) - dttz2 * speed(k + 1, i, j);
        lhsm_(i, j, k, 4) = lhs_(i, j, k, 4);
    }
};

__global__ void solve_kernel_three(double *lhs_, double *lhsp_, double *lhsm_, double *rhs, double *rho_i, double *us, double *speed, double c3c4, double dx2, double con43, double dx5, double c1c5, double dx1, double dttx2, double dttx1, double dxmax, double c2dttx1, double comz1, double comz4, double comz5, double comz6, int nx2, int ny2, int nz2, int nz)
{
    int k1, k2, m;
    double ru1, rhon1, fac1, fac2;

    int k;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    // part 3
    if (i > 0 && i <= nx2 && j > 0 && j <= ny2)
    {
        for (k = 1; k <= nz2; k++)
        {
            k1 = k;
            k2 = k + 1;
            fac1 = 1.0 / lhs_(i, j, k - 1, 2);
            lhs_(i, j, k - 1, 3) = fac1 * lhs_(i, j, k - 1, 3);
            lhs_(i, j, k - 1, 4) = fac1 * lhs_(i, j, k - 1, 4);

#pragma unroll 3
            for (m = 0; m < 3; m++)
                rhs(i, j, k - 1, m) = fac1 * rhs(i, j, k - 1, m);

            lhs_(i, j, k1, 2) = lhs_(i, j, k1, 2) - lhs_(i, j, k1, 1) * lhs_(i, j, k - 1, 3);
            lhs_(i, j, k1, 3) = lhs_(i, j, k1, 3) - lhs_(i, j, k1, 1) * lhs_(i, j, k - 1, 4);
#pragma unroll 3
            for (m = 0; m < 3; m++)
                rhs(i, j, k1, m) = rhs(i, j, k1, m) - lhs_(i, j, k1, 1) * rhs(i, j, k - 1, m);

            lhs_(i, j, k2, 1) = lhs_(i, j, k2, 1) - lhs_(i, j, k2, 0) * lhs_(i, j, k - 1, 3);
            lhs_(i, j, k2, 2) = lhs_(i, j, k2, 2) - lhs_(i, j, k2, 0) * lhs_(i, j, k - 1, 4);
#pragma unroll 3
            for (m = 0; m < 3; m++)
                rhs(i, j, k2, m) = rhs(i, j, k2, m) - lhs_(i, j, k2, 0) * rhs(i, j, k - 1, m);

            if (k == nz2)
            {
                fac1 = 1.0 / lhs_(i, j, k1, 2);
                lhs_(i, j, k1, 3) = fac1 * lhs_(i, j, k1, 3);
                lhs_(i, j, k1, 4) = fac1 * lhs_(i, j, k1, 4);
#pragma unroll 3
                for (m = 0; m < 3; m++)
                    rhs(i, j, k1, m) = fac1 * rhs(i, j, k1, m);

                lhs_(i, j, k2, 2) = lhs_(i, j, k2, 2) - lhs_(i, j, k2, 1) * lhs_(i, j, k1, 3);
                lhs_(i, j, k2, 3) = lhs_(i, j, k2, 3) - lhs_(i, j, k2, 1) * lhs_(i, j, k1, 4);
#pragma unroll 3
                for (m = 0; m < 3; m++)
                    rhs(i, j, k2, m) = rhs(i, j, k2, m) - lhs_(i, j, k2, 1) * rhs(i, j, k1, m);

                fac2 = 1.0 / lhs_(i, j, k2, 2);
#pragma unroll 3
                for (m = 0; m < 3; m++)
                    rhs(i, j, k2, m) = fac2 * rhs(i, j, k2, m);
            }

            m = 3;
            fac1 = 1.0 / lhsp_(i, j, k - 1, 2);
            lhsp_(i, j, k - 1, 3) = fac1 * lhsp_(i, j, k - 1, 3);
            lhsp_(i, j, k - 1, 4) = fac1 * lhsp_(i, j, k - 1, 4);
            rhs(i, j, k - 1, m) = fac1 * rhs(i, j, k - 1, m);

            lhsp_(i, j, k1, 2) = lhsp_(i, j, k1, 2) - lhsp_(i, j, k1, 1) * lhsp_(i, j, k - 1, 3);
            lhsp_(i, j, k1, 3) = lhsp_(i, j, k1, 3) - lhsp_(i, j, k1, 1) * lhsp_(i, j, k - 1, 4);
            rhs(i, j, k1, m) = rhs(i, j, k1, m) - lhsp_(i, j, k1, 1) * rhs(i, j, k - 1, m);

            lhsp_(i, j, k2, 1) = lhsp_(i, j, k2, 1) - lhsp_(i, j, k2, 0) * lhsp_(i, j, k - 1, 3);
            lhsp_(i, j, k2, 2) = lhsp_(i, j, k2, 2) - lhsp_(i, j, k2, 0) * lhsp_(i, j, k - 1, 4);
            rhs(i, j, k2, m) = rhs(i, j, k2, m) - lhsp_(i, j, k2, 0) * rhs(i, j, k - 1, m);

            m = 4;
            fac1 = 1.0 / lhsm_(i, j, k - 1, 2);
            lhsm_(i, j, k - 1, 3) = fac1 * lhsm_(i, j, k - 1, 3);
            lhsm_(i, j, k - 1, 4) = fac1 * lhsm_(i, j, k - 1, 4);
            rhs(i, j, k - 1, m) = fac1 * rhs(i, j, k - 1, m);
            lhsm_(i, j, k1, 2) = lhsm_(i, j, k1, 2) - lhsm_(i, j, k1, 1) * lhsm_(i, j, k - 1, 3);
            lhsm_(i, j, k1, 3) = lhsm_(i, j, k1, 3) - lhsm_(i, j, k1, 1) * lhsm_(i, j, k - 1, 4);
            rhs(i, j, k1, m) = rhs(i, j, k1, m) - lhsm_(i, j, k1, 1) * rhs(i, j, k - 1, m);
            lhsm_(i, j, k2, 1) = lhsm_(i, j, k2, 1) - lhsm_(i, j, k2, 0) * lhsm_(i, j, k - 1, 3);
            lhsm_(i, j, k2, 2) = lhsm_(i, j, k2, 2) - lhsm_(i, j, k2, 0) * lhsm_(i, j, k - 1, 4);
            rhs(i, j, k2, m) = rhs(i, j, k2, m) - lhsm_(i, j, k2, 0) * rhs(i, j, k - 1, m);

            if (k == nz2)
            {
                m = 3;
                fac1 = 1.0 / lhsp_(i, j, k1, 2);
                lhsp_(i, j, k1, 3) = fac1 * lhsp_(i, j, k1, 3);
                lhsp_(i, j, k1, 4) = fac1 * lhsp_(i, j, k1, 4);
                rhs(i, j, k1, m) = fac1 * rhs(i, j, k1, m);

                lhsp_(i, j, k2, 2) = lhsp_(i, j, k2, 2) - lhsp_(i, j, k2, 1) * lhsp_(i, j, k1, 3);
                lhsp_(i, j, k2, 3) = lhsp_(i, j, k2, 3) - lhsp_(i, j, k2, 1) * lhsp_(i, j, k1, 4);
                rhs(i, j, k2, m) = rhs(i, j, k2, m) - lhsp_(i, j, k2, 1) * rhs(i, j, k1, m);

                m = 4;
                fac1 = 1.0 / lhsm_(i, j, k1, 2);
                lhsm_(i, j, k1, 3) = fac1 * lhsm_(i, j, k1, 3);
                lhsm_(i, j, k1, 4) = fac1 * lhsm_(i, j, k1, 4);
                rhs(i, j, k1, m) = fac1 * rhs(i, j, k1, m);

                lhsm_(i, j, k2, 2) = lhsm_(i, j, k2, 2) - lhsm_(i, j, k2, 1) * lhsm_(i, j, k1, 3);
                lhsm_(i, j, k2, 3) = lhsm_(i, j, k2, 3) - lhsm_(i, j, k2, 1) * lhsm_(i, j, k1, 4);
                rhs(i, j, k2, m) = rhs(i, j, k2, m) - lhsm_(i, j, k2, 1) * rhs(i, j, k1, m);

                rhs(i, j, k2, 3) = rhs(i, j, k2, 3) / lhsp_(i, j, k2, 2);
                rhs(i, j, k2, 4) = rhs(i, j, k2, 4) / lhsm_(i, j, k2, 2);

#pragma unroll 3
                for (m = 0; m < 3; m++)
                    rhs(i, j, k1, m) = rhs(i, j, k1, m) - lhs_(i, j, k1, 3) * rhs(i, j, k2, m);

                rhs(i, j, k1, 3) = rhs(i, j, k1, 3) - lhsp_(i, j, k1, 3) * rhs(i, j, k2, 3);
                rhs(i, j, k1, 4) = rhs(i, j, k1, 4) - lhsm_(i, j, k1, 3) * rhs(i, j, k2, 4);
            }
        }
    }
}

__global__ void solve_kernel_four(
    double *lhs_, double *lhsp_, double *lhsm_,
    double *rhs,
    int nx2, int ny2, int nz2)
{
    int i1, i2, m, k;

    int j = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int i = threadIdx.y + blockIdx.y * blockDim.y + 1;

    // part 4
    if ((j <= ny2) && (i <= nx2))
    {
        for (k = nz2; k >= 1; k--)
        {
            i1 = k;
            i2 = k + 1;
            for (m = 0; m < 3; m++)
                rhs(i, j, k - 1, m) = rhs(i, j, k - 1, m) - lhs_(i, j, k - 1, 3) * rhs(i1, j, k - 1, m) - lhs_(i, j, k - 1, 4) * rhs(i2, j, k - 1, m);

            rhs(i, j, k - 1, 3) = rhs(i, j, k - 1, 3) - lhsp_(i, j, k - 1, 3) * rhs(i1, j, k - 1, 3) - lhsp_(i, j, k - 1, 4) * rhs(i2, j, k - 1, 3);
            rhs(i, j, k - 1, 4) = rhs(i, j, k - 1, 4) - lhsm_(i, j, k - 1, 3) * rhs(i1, j, k - 1, 4) - lhsm_(i, j, k - 1, 4) * rhs(i2, j, k - 1, 4);
        }
    }
}

void run_solve_kernels(
    CUDAParameters cudaParams,
    double *lhs_, double *lhsp_, double *lhsm_, double *rhs, double *rho_i, double *us, double *speed, double c3c4, double dx2, double con43, double dx5, double c1c5, double dx1, double dttx2, double dttx1, double dxmax, double c2dttx1, double comz1, double comz4, double comz5, double comz6,
    int nx2, int ny2, int nz2, int nx)
{
    auto [blocks, threads, blocks2, threads2, blockst, threadst] = cudaParams;

    cudaDeviceSynchronize();
    solve_kernel_one<<<blocks, threads>>>(lhs_, lhsp_, lhsm_, nz2, ny2, nx2);
    solve_kernel_two<<<blocks, threads>>>((double *)lhs_, (double *)lhsp_, (double *)lhsm_, (double *)rho_i, (double *)us, (double *)speed, c3c4, dy3, con43, dy5, c1c5, dy1, dtty2, dtty1, dymax, c2dtty1, comz1, comz4, comz5, comz6, nx2, ny2, nz2);
    solve_kernel_two1<<<blocks2, threads2>>>((double *)lhs_, (double *)lhsp_, (double *)lhsm_, (double *)rho_i, (double *)us, (double *)speed, c3c4, dy3, con43, dy5, c1c5, dy1, dtty2, dtty1, dymax, c2dtty1, comz1, comz4, comz5, comz6, nx2, ny2, nz2);
    solve_kernel_two2<<<blocks2, threads2>>>((double *)lhs_, (double *)lhsp_, (double *)lhsm_, (double *)rho_i, (double *)us, (double *)speed, c3c4, dy3, con43, dy5, c1c5, dy1, dtty2, dtty1, dymax, c2dtty1, comz1, comz4, comz5, comz6, nx2, ny2, nz2);
    cudaDeviceSynchronize();
    solve_kernel_two_nz2<<<blocks2, threads2>>>((double *)lhs_, (double *)lhsp_, (double *)lhsm_, (double *)rho_i, (double *)us, (double *)speed, c3c4, dy3, con43, dy5, c1c5, dy1, dtty2, dtty1, dymax, c2dtty1, comz1, comz4, comz5, comz6, nx2, ny2, nz2, nx);
    solve_kernel_two_nz3<<<blocks2, threads2>>>((double *)lhs_, (double *)lhsp_, (double *)lhsm_, (double *)rho_i, (double *)us, (double *)speed, c3c4, dy3, con43, dy5, c1c5, dy1, dtty2, dtty1, dymax, c2dtty1, comz1, comz4, comz5, comz6, nx2, ny2, nz2, nx);
    cudaDeviceSynchronize();
    solve_kernel_three<<<blocks2, threads2>>>((double *)lhs_gpu, (double *)lhsp_gpu, (double *)lhsm_gpu, (double *)gpuRhs, (double *)gpuRho_i, (double *)gpuUs, (double *)gpuSpeed, c3c4, dx2, con43, dx5, c1c5, dx1, dttx2, dttx1, dxmax, c2dttx1, comz1, comz4, comz5, comz6, nx2, ny2, nz2, nx);
    cudaDeviceSynchronize();
    solve_kernel_four<<<blocks2, threads2>>>((double *)lhs_gpu, (double *)lhsp_gpu, (double *)lhsm_gpu, (double *)gpuRhs, nx2, ny2, nz2);
    cudaDeviceSynchronize();
}
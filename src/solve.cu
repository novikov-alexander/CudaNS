#include "header.hpp"

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

#undef s
#undef speed
#define s(x, y, z) s[INDEX_3D(y, z, x)]
#define speed(x, y, z) speed[INDEX_3D(y, z, x)]
__global__ void solve_kernel_two1(
    double *lhs_, double *lhsp_, double *lhsm_,
    double *rho_i, double *s, double *speed,
    int ny2, int nx2, int nz2,
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
        lhs_(i, j, k, 0) = 0.0;

        ru1 = c3c4 * rho_i(k - 1, i, j);
        rhos1 = fmax(fmax(dz4 + con43 * ru1, dz5 + c1c5 * ru1), fmax(dzmax + ru1, dz1));
        lhs_(i, j, k, 1) = -dttz2 * s(k - 1, i, j) - dttz1 * rhos1;

        ru1 = c3c4 * rho_i(k, i, j);
        rhos1 = fmax(fmax(dz4 + con43 * ru1, dz5 + c1c5 * ru1), fmax(dzmax + ru1, dz1));
        lhs_(i, j, k, 2) = 1.0 + c2dttz1 * rhos1;

        ru1 = c3c4 * rho_i(k + 1, i, j);
        rhos1 = fmax(fmax(dz4 + con43 * ru1, dz5 + c1c5 * ru1), fmax(dzmax + ru1, dz1));
        lhs_(i, j, k, 3) = dttz2 * s(k + 1, i, j) - dttz1 * rhos1;
        lhs_(i, j, k, 4) = 0.0;

        lhs_(i, j, k, 2) = lhs_(i, j, k, 2) + comz5;
        lhs_(i, j, k, 3) = lhs_(i, j, k, 3) - comz4;
        lhs_(i, j, k, 4) = lhs_(i, j, k, 4) + comz1;

        lhsp_(i, j, k, 0) = lhs_(i, j, k, 0);
        lhsp_(i, j, k, 1) = lhs_(i, j, k, 1) - dttz2 * speed(k - 1, i, j);
        lhsp_(i, j, k, 2) = lhs_(i, j, k, 2);
        lhsp_(i, i, k, 3) = lhs_(i, i, k, 3) + dttz2 * speed(k + 1, i, i);
        lhsp_(i, i, k, 4) = lhs_(i, i, k, 4);
        lhsm_(i, i, k, 0) = lhs_(i, i, k, 0);
        lhsm_(i, i, k, 1) = lhs_(i, i, k, 1) + dttz2 * speed(k - 1, i, i);
        lhsm_(i, i, k, 2) = lhs_(i, i, k, 2);
        lhsm_(i, i, k, 3) = lhs_(j, i, k, 3) - dttz2 * speed(k + 1, j, i);
        lhsm_(j, i, k, 4) = lhs_(j, i, k, 4);
    }
};
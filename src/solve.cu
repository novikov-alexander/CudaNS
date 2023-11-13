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
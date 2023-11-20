#include "adi.hpp"
#include "solver.hpp"
#include <array>

constexpr int printStep = 20;

void Solver::allocateArrays()
{
    const int size_5x3d = sizeof(double) * nx * ny * nz * 5;
    const int size_3d = sizeof(double) * nx * ny * nz;

    std::array<void *, 7> gpu_pointers_5x3d = {
        &gpuU, &gpuRhs, &gpuForcing, &gpuTmp,
        &lhs_gpu, &lhsp_gpu, &lhsm_gpu};

    for (void *gpu_ptr : gpu_pointers_5x3d)
    {
        CudaSafeCall(cudaMalloc((void **)&gpu_ptr, size_5x3d));
    }
    std::array<void *, 7> gpu_pointers_3d = {
        &gpuRho_i, &gpuUs, &gpuVs, &gpuWs, &gpuQs, &gpuSquare, &gpuSpeed};

    for (void *gpu_ptr : gpu_pointers_3d)
    {
        CudaSafeCall(cudaMalloc((void **)&gpu_ptr, size_3d));
    }
}

void Solver::deallocateArrays()
{
    std::array<void *, 14> gpu_pointers = {
        &gpuU, &gpuRhs, &gpuRho_i, &gpuUs, &gpuVs, &gpuWs, &gpuQs, &gpuSquare, &gpuSpeed, &gpuForcing, &gpuTmp,
        &lhs_gpu, &lhsp_gpu, &lhsm_gpu};

    for (void *gpu_ptr : gpu_pointers)
    {
        cudaFree(gpu_ptr);
    }
}

void Solver::step()
{
    compute_rhs();
    xinvr();
    x_solve();
    y_solve();
    z_solve();
    add();
}

Solver::Solver()
{
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    set_constants();

    allocateArrays();
}

Solver::~Solver()
{
    deallocateArrays();
}

void Solver::solve(int niter)
{
    for (int i = 1; i <= t_last; i++)
        timer_clear(i);

    timer_start(t_total);

    for (int step = 1; step <= niter; step++)
    {
        if ((step % printStep) == 0 || step == 1)
            printf(" Time step %4d\n", step);
        this->step();
    }

    timer_stop(t_total);
}

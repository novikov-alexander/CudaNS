#include "adi.hpp"
#include "solver.hpp"

constexpr int printStep = 20;

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

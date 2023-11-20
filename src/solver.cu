#include "adi.hpp"
#include "solver.hpp"

constexpr int printStep = 20;

class Solver
{
    void step()
    {
        compute_rhs();
        xinvr();
        x_solve();
        y_solve();
        z_solve();
        add();
    }

public:
    Solver()
    {
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

        set_constants();
    }

    void solve(int niter)
    {
        for (int step = 1; step <= niter; step++)
        {
            if ((step % printStep) == 0 || step == 1)
                printf(" Time step %4d\n", step);
            this->step();
        }
    }
};
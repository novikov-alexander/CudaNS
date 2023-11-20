#include "adi.hpp"
#include "solver.hpp"

class Solver
{
public:
    void step()
    {
        compute_rhs();
        xinvr();
        x_solve();
        y_solve();
        z_solve();
        add();
    }
};
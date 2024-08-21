#include "header.hpp"

class Solver
{
    void step();
    void allocateArrays();
    void deallocateArrays();

public:
    Solver();
    ~Solver();
    void solve(int niter);
};
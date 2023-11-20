#include "header.hpp"

class Solver
{
    void step();
    void allocateArrays();
    void deallocateArrays();

public:
    void solve(int niter);
};
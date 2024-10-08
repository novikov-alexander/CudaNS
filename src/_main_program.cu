//---------------------------------------------------------------------
// MAIN program
//---------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include "header.hpp"
#include "solver.hpp"

/* global parameters */
int nx2, ny2, nz2, nx, ny, nz;
logical timeron;

/* constants */
double tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3,
    dx1, dx2, dx3, dx4, dx5, dy1, dy2, dy3, dy4,
    dy5, dz1, dz2, dz3, dz4, dz5, dssp, dt,
    ce[5][13], dxmax, dymax, dzmax, xxcon1, xxcon2,
    xxcon3, xxcon4, xxcon5, dx1tx1, dx2tx1, dx3tx1,
    dx4tx1, dx5tx1, yycon1, yycon2, yycon3, yycon4,
    yycon5, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1,
    zzcon1, zzcon2, zzcon3, zzcon4, zzcon5, dz1tz1,
    dz2tz1, dz3tz1, dz4tz1, dz5tz1, dnxm1, dnym1,
    dnzm1, c1c2, c1c5, c3c4, c1345, conz1, c1, c2,
    c3, c4, c5, c4dssp, c5dssp, dtdssp, dttx1, bt,
    dttx2, dtty1, dtty2, dttz1, dttz2, c2dttx1,
    c2dtty1, c2dttz1, comz1, comz4, comz5, comz6,
    c3c4tx3, c3c4ty3, c3c4tz3, c2iv, con43, con16;

/* main arrays */
using Grid3DPtr = double (*)[P_SIZE][P_SIZE][P_SIZE];
using Grid2DPtr = double (*)[P_SIZE][P_SIZE];

Grid3DPtr u, rhs, forcing;
Grid2DPtr us, vs, ws, qs, rho_i, speed, square;

Grid3DPtr gpuU, gpuRhs, gpuForcing, gpuTmp;
Grid2DPtr gpuUs, gpuVs, gpuWs, gpuQs, gpuRho_i, gpuSpeed, gpuSquare;

Grid3DPtr lhs_gpu, lhsp_gpu, lhsm_gpu;

Grid2DPtr gpuTmp3D;

void copyGridsToDevice()
{
    const int size5 = sizeof(double) * P_SIZE * P_SIZE * P_SIZE * 5;
    const int size = sizeof(double) * P_SIZE * P_SIZE * P_SIZE;

    CudaSafeCall(cudaMemcpy(gpuU, u, size5, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(gpuRho_i, rho_i, size, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(gpuUs, us, size, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(gpuVs, vs, size, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(gpuWs, ws, size, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(gpuQs, qs, size, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(gpuSquare, square, size, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(gpuSpeed, speed, size, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(gpuForcing, forcing, size5, cudaMemcpyHostToDevice));
}

void copyGridsFromDevice()
{
    const int size5 = sizeof(double) * P_SIZE * P_SIZE * P_SIZE * 5;
    const int size = sizeof(double) * P_SIZE * P_SIZE * P_SIZE;

    CudaSafeCall(cudaMemcpy(u, gpuU, size5, cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(rho_i, gpuRho_i, size, cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(us, gpuUs, size, cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(vs, gpuVs, size, cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(ws, gpuWs, size, cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(qs, gpuQs, size, cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(square, gpuSquare, size, cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(speed, gpuSpeed, size, cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(forcing, gpuForcing, size5, cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(rhs, gpuRhs, size5, cudaMemcpyDeviceToHost));
}

int main(int argc, char *argv[])
{
    printf("\n Program started \n");

    int niter;
    const char *t_names[t_last + 1];
    timeron = inittrace(t_names);

    if (!initparameters(argc, argv, &niter))
        return -1;
    if (!allocateArrays())
        return -2;

    // init
    exact_rhs();
    init_u();
    auto solver = new Solver();

    copyGridsToDevice();

    // main loop
    solver->solve(niter);

    copyGridsFromDevice();

    print_results(niter, t_names);

    if (!deallocateArrays())
        return -2;
    return 0;
}

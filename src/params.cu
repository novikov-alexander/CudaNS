#include "header.hpp"

CUDAParameters setupDimensions(int nx2, int ny2, int nz2, int nx, int ny, int nz)
{
    const int blockSizeX = 32;
    const int blockSizeY = 8;

    CUDAParameters params;

    params.blocks = dim3((nx2 + blockSizeX - 1) / blockSizeX, ny2, nz2);
    params.threads = dim3(blockSizeX, 1, 1);

    params.blocks2 = dim3((nx2 + blockSizeX - 1) / blockSizeX, (ny2 + blockSizeY - 1) / blockSizeY);
    params.threads2 = dim3(blockSizeX, blockSizeY);

    params.blockst = dim3((nx + blockSizeY - 1) / blockSizeY, (ny + blockSizeY - 1) / blockSizeY, (nz + blockSizeY - 1) / blockSizeY);
    params.threadst = dim3(blockSizeY, blockSizeY, blockSizeY);

    return params;
}
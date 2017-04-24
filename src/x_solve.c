#include "header.h"

//---------------------------------------------------------------------
// this function performs the solution of the approximate factorization
// step in the x-direction for all five matrix components
// simultaneously. The Thomas algorithm is employed to solve the
// systems for the x-lines. Boundary conditions are non-periodic
//---------------------------------------------------------------------

#define lhs_(x,y,z,m) lhs_[m + (z) * 5 + (y) * 5 * P_SIZE + (x) * 5 * P_SIZE * P_SIZE]
#define lhsm_(x,y,z,m) lhsm_[m + (z) * 5 + (y) * 5 * P_SIZE + (x) * 5 * P_SIZE * P_SIZE]
#define lhsp_(x,y,z,m) lhsp_[m + (z) * 5 + (y) * 5 * P_SIZE + (x) * 5 * P_SIZE * P_SIZE]
#define rhs(x,y,z,m) rhs[m + (z) * 5 + (y) * 5 * P_SIZE + (x) * 5 * P_SIZE * P_SIZE]
#define rho_i(x,y,z) rho_i[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE]
#define us(x,y,z) us[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE]
#define speed(x,y,z) speed[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE]
__global__ void x_solve_kernel_one(double* lhs_, double* lhsp_, double* lhsm_, int nx2, int ny2, int nz2)
{
	int m;

	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int k = threadIdx.z + blockIdx.z * blockDim.z + 1;

	//part 1
	if (k <= nz2 && j <= ny2)
    {        
        for (m = 0; m < 5; m++)
        {
            lhs_(k,j,0,m) = lhs_(k,j,nx2 + 1,m) = 0.0;
            lhsp_(k,j,0,m) = lhsp_(k,j,nx2 + 1,m) = 0.0;
            lhsm_(k,j,0,m) = lhsm_(k,j,nx2 + 1,m) = 0.0;
        }

        lhs_(k,j,0,2) = lhs_(k,j,nx2 + 1,2) = 1.0;
        lhsp_(k,j,0,2) = lhsp_(k,j,nx2 + 1,2) = 1.0;
        lhsm_(k,j,0,2) = lhsm_(k,j,nx2 + 1,2) = 1.0;         
	}
}

__global__ void x_solve_kernel_two(double* lhs_, double* lhsp_, double* lhsm_, double* rhs, double* rho_i, double* us, double* speed, double c3c4, double dx2, double  con43, double  dx5, double c1c5, double dx1, double dttx2, double dttx1, double dxmax, double c2dttx1, double comz1, double comz4, double comz5, double comz6, int nx2, int ny2, int nz2, int nx)
{
	int  i1, i2, m;
	double ru1, rhon1;

	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int k = threadIdx.z + blockIdx.z * blockDim.z + 1;

	//part 2
	if (k <= nz2 && j <= ny2 && i <= nx2)
    {        
        lhs_(k,j,i,0) = 0.0;
        ru1 = c3c4*rho_i(k,j,i - 1);
        rhon1 = max(max(dx2 + con43 * ru1, dx5 + c1c5 * ru1), max(dxmax + ru1, dx1));
        lhs_(k,j,i,1) = -dttx2 * us(k,j,i - 1) - dttx1 * rhon1;

        ru1 = c3c4*rho_i(k,j,i);
        rhon1 = max(max(dx2 + con43 * ru1, dx5 + c1c5 * ru1), max(dxmax + ru1, dx1));
        lhs_(k,j,i,2) = 1.0 + c2dttx1 * rhon1;

        ru1 = c3c4*rho_i(k,j,i + 1);
        rhon1 = max(max(dx2 + con43 * ru1, dx5 + c1c5 * ru1), max(dxmax + ru1, dx1));
        lhs_(k,j,i,3) = dttx2 * us(k,j,i + 1) - dttx1 * rhon1;
        lhs_(k,j,i,4) = 0.0;

        if (i == 1)
        {
            lhs_(k,j,i,2) = lhs_(k,j,i,2) + comz5;
            lhs_(k,j,i,3) = lhs_(k,j,i,3) - comz4;
            lhs_(k,j,i,4) = lhs_(k,j,i,4) + comz1;
        }
        else if (i == 2)
        {
            lhs_(k,j,i,1) = lhs_(k,j,i,1) - comz4;
            lhs_(k,j,i,2) = lhs_(k,j,i,2) + comz6;
            lhs_(k,j,i,3) = lhs_(k,j,i,3) - comz4;
            lhs_(k,j,i,4) = lhs_(k,j,i,4) + comz1;
        }
        else if (i == nx - 3)
        {
            lhs_(k,j,i,0) = lhs_(k,j,i,0) + comz1;
            lhs_(k,j,i,1) = lhs_(k,j,i,1) - comz4;
            lhs_(k,j,i,2) = lhs_(k,j,i,2) + comz6;
            lhs_(k,j,i,3) = lhs_(k,j,i,3) - comz4;
        }
        else if (i == nx - 2)
        {
            lhs_(k,j,i,0) = lhs_(k,j,i,0) + comz1;
            lhs_(k,j,i,1) = lhs_(k,j,i,1) - comz4;
            lhs_(k,j,i,2) = lhs_(k,j,i,2) + comz5;
        }
        else
        {
            lhs_(k,j,i,0) = lhs_(k,j,i,0) + comz1;
            lhs_(k,j,i,1) = lhs_(k,j,i,1) - comz4;
            lhs_(k,j,i,2) = lhs_(k,j,i,2) + comz6;
            lhs_(k,j,i,3) = lhs_(k,j,i,3) - comz4;
            lhs_(k,j,i,4) = lhs_(k,j,i,4) + comz1;

        }

        lhsp_(k,j,i,0) = lhs_(k,j,i,0);
        lhsp_(k,j,i,1) = lhs_(k,j,i,1) - dttx2 * speed(k,j,i - 1);
        lhsp_(k,j,i,2) = lhs_(k,j,i,2);
        lhsp_(k,j,i,3) = lhs_(k,j,i,3) + dttx2 * speed(k,j,i + 1);
        lhsp_(k,j,i,4) = lhs_(k,j,i,4);

        lhsm_(k,j,i,0) = lhs_(k,j,i,0);
        lhsm_(k,j,i,1) = lhs_(k,j,i,1) + dttx2 * speed(k,j,i - 1);
        lhsm_(k,j,i,2) = lhs_(k,j,i,2);
        lhsm_(k,j,i,3) = lhs_(k,j,i,3) - dttx2 * speed(k,j,i + 1);
        lhsm_(k,j,i,4) = lhs_(k,j,i,4);
	}
}

__global__ void x_solve_kernel_three(double* lhs_, double* lhsp_, double* lhsm_, double* rhs, double* rho_i, double* us, double* speed, double c3c4, double dx2, double  con43, double  dx5, double c1c5, double dx1, double dttx2, double dttx1, double dxmax, double c2dttx1, double comz1, double comz4, double comz5, double comz6, int nx2, int ny2, int nz2, int nx)
{
	int  i1, i2, m;
	double ru1, rhon1, fac1, fac2;

	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int k = threadIdx.z + blockIdx.z * blockDim.z + 1;

	//part 3
	if (k <= nz2 && j <= ny2 && i <= nx2)
    {        
		i1 = i;
		i2 = i + 1;
		fac1 = 1.0 / lhs_(k,j,i - 1,2);
		lhs_(k,j,i - 1,3) = fac1 * lhs_(k,j,i - 1,3);
		lhs_(k,j,i - 1,4) = fac1 * lhs_(k,j,i - 1,4);
		for (m = 0; m < 3; m++)
		    rhs(k,j,i - 1,m) = fac1*rhs(k,j,i - 1,m);

		lhs_(k,j,i1,2) = lhs_(k,j,i1,2) - lhs_(k,j,i1,1) * lhs_(k,j,i - 1,3);
		lhs_(k,j,i1,3) = lhs_(k,j,i1,3) - lhs_(k,j,i1,1) * lhs_(k,j,i - 1,4);
		for (m = 0; m < 3; m++)
		    rhs(k,j,i1,m) = rhs(k,j,i1,m) - lhs_(k,j,i1,1) * rhs(k,j,i - 1,m);

		lhs_(k,j,i2,1) = lhs_(k,j,i2,1) - lhs_(k,j,i2,0) * lhs_(k,j,i - 1,3);
		lhs_(k,j,i2,2) = lhs_(k,j,i2,2) - lhs_(k,j,i2,0) * lhs_(k,j,i - 1,4);
		for (m = 0; m < 3; m++)
		    rhs(k,j,i2,m) = rhs(k,j,i2,m) - lhs_(k,j,i2,0) * rhs(k,j,i - 1,m);

		if (i == nx2)
		{
		    fac1 = 1.0 / lhs_(k,j,i1,2);
		    lhs_(k,j,i1,3) = fac1 * lhs_(k,j,i1,3);
		    lhs_(k,j,i1,4) = fac1 * lhs_(k,j,i1,4);
		    for (m = 0; m < 3; m++)
		        rhs(k,j,i1,m) = fac1 * rhs(k,j,i1,m);

		    lhs_(k,j,i2,2) = lhs_(k,j,i2,2) - lhs_(k,j,i2,1) * lhs_(k,j,i1,3);
		    lhs_(k,j,i2,3) = lhs_(k,j,i2,3) - lhs_(k,j,i2,1) * lhs_(k,j,i1,4);
		    for (m = 0; m < 3; m++)
		        rhs(k,j,i2,m) = rhs(k,j,i2,m) - lhs_(k,j,i2,1) * rhs(k,j,i1,m);

		    fac2 = 1.0 / lhs_(k,j,i2,2);
		    for (m = 0; m < 3; m++)
		        rhs(k,j,i2,m) = fac2*rhs(k,j,i2,m);
		}

		m = 3;
		fac1 = 1.0 / lhsp_(k,j,i - 1,2);
		lhsp_(k,j,i - 1,3) = fac1 * lhsp_(k,j,i - 1,3);
		lhsp_(k,j,i - 1,4) = fac1 * lhsp_(k,j,i - 1,4);
		rhs(k,j,i - 1,m) = fac1 * rhs(k,j,i - 1,m);

		lhsp_(k,j,i1,2) = lhsp_(k,j,i1,2) - lhsp_(k,j,i1,1) * lhsp_(k,j,i - 1,3);
		lhsp_(k,j,i1,3) = lhsp_(k,j,i1,3) - lhsp_(k,j,i1,1) * lhsp_(k,j,i - 1,4);
		rhs(k,j,i1,m) = rhs(k,j,i1,m) - lhsp_(k,j,i1,1) * rhs(k,j,i - 1,m);

		lhsp_(k,j,i2,1) = lhsp_(k,j,i2,1) - lhsp_(k,j,i2,0) * lhsp_(k,j,i - 1,3);
		lhsp_(k,j,i2,2) = lhsp_(k,j,i2,2) - lhsp_(k,j,i2,0) * lhsp_(k,j,i - 1,4);
		rhs(k,j,i2,m) = rhs(k,j,i2,m) - lhsp_(k,j,i2,0) * rhs(k,j,i - 1,m);

		m = 4;
		fac1 = 1.0 / lhsm_(k,j,i - 1,2);
		lhsm_(k,j,i - 1,3) = fac1*lhsm_(k,j,i - 1,3);
		lhsm_(k,j,i - 1,4) = fac1*lhsm_(k,j,i - 1,4);
		rhs(k,j,i - 1,m) = fac1*rhs(k,j,i - 1,m);
		lhsm_(k,j,i1,2) = lhsm_(k,j,i1,2) - lhsm_(k,j,i1,1) * lhsm_(k,j,i - 1,3);
		lhsm_(k,j,i1,3) = lhsm_(k,j,i1,3) - lhsm_(k,j,i1,1) * lhsm_(k,j,i - 1,4);
		rhs(k,j,i1,m) = rhs(k,j,i1,m) - lhsm_(k,j,i1,1) * rhs(k,j,i - 1,m);
		lhsm_(k,j,i2,1) = lhsm_(k,j,i2,1) - lhsm_(k,j,i2,0) * lhsm_(k,j,i - 1,3);
		lhsm_(k,j,i2,2) = lhsm_(k,j,i2,2) - lhsm_(k,j,i2,0) * lhsm_(k,j,i - 1,4);
		rhs(k,j,i2,m) = rhs(k,j,i2,m) - lhsm_(k,j,i2,0) * rhs(k,j,i - 1,m);

		if (i == nx2)
		{
		    m = 3;
		    fac1 = 1.0 / lhsp_(k,j,i1,2);
		    lhsp_(k,j,i1,3) = fac1 * lhsp_(k,j,i1,3);
		    lhsp_(k,j,i1,4) = fac1 * lhsp_(k,j,i1,4);
		    rhs(k,j,i1,m) = fac1 * rhs(k,j,i1,m);

		    lhsp_(k,j,i2,2) = lhsp_(k,j,i2,2) - lhsp_(k,j,i2,1) * lhsp_(k,j,i1,3);
		    lhsp_(k,j,i2,3) = lhsp_(k,j,i2,3) - lhsp_(k,j,i2,1) * lhsp_(k,j,i1,4);
		    rhs(k,j,i2,m) = rhs(k,j,i2,m) - lhsp_(k,j,i2,1) * rhs(k,j,i1,m);

		    m = 4;
		    fac1 = 1.0 / lhsm_(k,j,i1,2);
		    lhsm_(k,j,i1,3) = fac1 * lhsm_(k,j,i1,3);
		    lhsm_(k,j,i1,4) = fac1 * lhsm_(k,j,i1,4);
		    rhs(k,j,i1,m) = fac1*rhs(k,j,i1,m);

		    lhsm_(k,j,i2,2) = lhsm_(k,j,i2,2) - lhsm_(k,j,i2,1) * lhsm_(k,j,i1,3);
		    lhsm_(k,j,i2,3) = lhsm_(k,j,i2,3) - lhsm_(k,j,i2,1) * lhsm_(k,j,i1,4);
		    rhs(k,j,i2,m) = rhs(k,j,i2,m) - lhsm_(k,j,i2,1) * rhs(k,j,i1,m);

		    rhs(k,j,i2,3) = rhs(k,j,i2,3) / lhsp_(k,j,i2,2);
		    rhs(k,j,i2,4) = rhs(k,j,i2,4) / lhsm_(k,j,i2,2);

		    for (m = 0; m < 3; m++)
		        rhs(k,j,i1,m) = rhs(k,j,i1,m) - lhs_(k,j,i1,3) * rhs(k,j,i2,m);

		    rhs(k,j,i1,3) = rhs(k,j,i1,3) - lhsp_(k,j,i1,3) * rhs(k,j,i2,3);
		    rhs(k,j,i1,4) = rhs(k,j,i1,4) - lhsm_(k,j,i1,3) * rhs(k,j,i2,4);
		}
	}
}

__global__ void x_solve_kernel_four(double* lhs_, double* lhsp_, double* lhsm_, double* rhs, double* rho_i, double* us, double* speed, double c3c4, double dx2, double  con43, double  dx5, double c1c5, double dx1, double dttx2, double dttx1, double dxmax, double c2dttx1, double comz1, double comz4, double comz5, double comz6, int nx2, int ny2, int nz2, int nx)
{
	int  i1, i2, m, i;

	//int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int k = threadIdx.z + blockIdx.z * blockDim.z + 1;

	//part 4
	if (k <= nz2 && j <= ny2)
	{
		for (i = nx2; i >= 1; i--)
        {
            i1 = i;
            i2 = i + 1;
            for (m = 0; m < 3; m++)
                rhs(k,j,i - 1,m) = rhs(k,j,i - 1,m) - lhs_(k,j,i - 1,3) * rhs(k,j,i1,m) - lhs_(k,j,i - 1,4) * rhs(k,j,i2,m);

            rhs(k,j,i - 1,3) = rhs(k,j,i - 1,3) - lhsp_(k,j,i - 1,3) * rhs(k,j,i1,3) - lhsp_(k,j,i - 1,4) * rhs(k,j,i2,3);
            rhs(k,j,i - 1,4) = rhs(k,j,i - 1,4) - lhsm_(k,j,i - 1,3) * rhs(k,j,i1,4) - lhsm_(k,j,i - 1,4) * rhs(k,j,i2,4);
        }
	}
}

#undef lhs_
#undef lhsp_
#undef lhsm_
#undef rhs
#undef rho_i
#undef us
#undef speed



void x_solve()
{
    int i, j, k, i1, i2, m;
    double ru1, rhon1, fac1, fac2;

	const int size5 = sizeof(double)*P_SIZE*P_SIZE*P_SIZE*5;
	const int size = sizeof(double)*P_SIZE*P_SIZE*P_SIZE;

	dim3 blocks = dim3(nx2 / 32+1, ny2 / 4+1, nz2);
	dim3 threads = dim3(32, 4, 1);
	
	CudaSafeCall(cudaMemcpy(lhs_gpu, lhs_, size5, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(lhsp_gpu, lhsp_, size5, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(lhsm_gpu, lhsm_, size5, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(gpuRhs, rhs, size5, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(gpuRho_i, rho_i, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(gpuUs, us, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(gpuSpeed, speed, size, cudaMemcpyHostToDevice));

    if (timeron) timer_start(t_xsolve);

	x_solve_kernel_one<<<blocks, threads>>>((double*)lhs_gpu, (double*)lhsp_gpu, (double*)lhsm_gpu, nx2, ny2, nz2);
	x_solve_kernel_two<<<blocks, threads>>>((double*)lhs_gpu, (double*)lhsp_gpu, (double*)lhsm_gpu, (double*)gpuRhs, (double*)gpuRho_i, (double*)gpuUs, (double*)gpuSpeed, c3c4, dx2,  con43,  dx5, c1c5, dx1, dttx2, dttx1, dxmax, c2dttx1, comz1, comz4, comz5, comz6, nx2, ny2, nz2, nx);

	CudaSafeCall(cudaMemcpy(rho_i, gpuRho_i, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(us, gpuUs, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(speed, gpuSpeed, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(rhs, gpuRhs, size5, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(lhs_, lhs_gpu, size5, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(lhsp_, lhsp_gpu, size5, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(lhsm_, lhsm_gpu, size5, cudaMemcpyDeviceToHost));

	 for (k = 1; k <= nz2; k++)
    {        
        for (j = 1; j <= ny2; j++)
        {
			for (i = 1; i <= nx2; i++)
            {
                i1 = i;
                i2 = i + 1;
                fac1 = 1.0 / lhs_[k][j][i - 1][2];
                lhs_[k][j][i - 1][3] = fac1 * lhs_[k][j][i - 1][3];
                lhs_[k][j][i - 1][4] = fac1 * lhs_[k][j][i - 1][4];
                for (m = 0; m < 3; m++)
                    rhs[k][j][i - 1][m] = fac1*rhs[k][j][i - 1][m];

                lhs_[k][j][i1][2] = lhs_[k][j][i1][2] - lhs_[k][j][i1][1] * lhs_[k][j][i - 1][3];
                lhs_[k][j][i1][3] = lhs_[k][j][i1][3] - lhs_[k][j][i1][1] * lhs_[k][j][i - 1][4];
                for (m = 0; m < 3; m++)
                    rhs[k][j][i1][m] = rhs[k][j][i1][m] - lhs_[k][j][i1][1] * rhs[k][j][i - 1][m];

                lhs_[k][j][i2][1] = lhs_[k][j][i2][1] - lhs_[k][j][i2][0] * lhs_[k][j][i - 1][3];
                lhs_[k][j][i2][2] = lhs_[k][j][i2][2] - lhs_[k][j][i2][0] * lhs_[k][j][i - 1][4];
                for (m = 0; m < 3; m++)
                    rhs[k][j][i2][m] = rhs[k][j][i2][m] - lhs_[k][j][i2][0] * rhs[k][j][i - 1][m];

                if (i == nx2)
                {
                    fac1 = 1.0 / lhs_[k][j][i1][2];
                    lhs_[k][j][i1][3] = fac1 * lhs_[k][j][i1][3];
                    lhs_[k][j][i1][4] = fac1 * lhs_[k][j][i1][4];
                    for (m = 0; m < 3; m++)
                        rhs[k][j][i1][m] = fac1 * rhs[k][j][i1][m];

                    lhs_[k][j][i2][2] = lhs_[k][j][i2][2] - lhs_[k][j][i2][1] * lhs_[k][j][i1][3];
                    lhs_[k][j][i2][3] = lhs_[k][j][i2][3] - lhs_[k][j][i2][1] * lhs_[k][j][i1][4];
                    for (m = 0; m < 3; m++)
                        rhs[k][j][i2][m] = rhs[k][j][i2][m] - lhs_[k][j][i2][1] * rhs[k][j][i1][m];

                    fac2 = 1.0 / lhs_[k][j][i2][2];
                    for (m = 0; m < 3; m++)
                        rhs[k][j][i2][m] = fac2*rhs[k][j][i2][m];
                }
            
                m = 3;
                fac1 = 1.0 / lhsp_[k][j][i - 1][2];
                lhsp_[k][j][i - 1][3] = fac1 * lhsp_[k][j][i - 1][3];
                lhsp_[k][j][i - 1][4] = fac1 * lhsp_[k][j][i - 1][4];
                rhs[k][j][i - 1][m] = fac1 * rhs[k][j][i - 1][m];

                lhsp_[k][j][i1][2] = lhsp_[k][j][i1][2] - lhsp_[k][j][i1][1] * lhsp_[k][j][i - 1][3];
                lhsp_[k][j][i1][3] = lhsp_[k][j][i1][3] - lhsp_[k][j][i1][1] * lhsp_[k][j][i - 1][4];
                rhs[k][j][i1][m] = rhs[k][j][i1][m] - lhsp_[k][j][i1][1] * rhs[k][j][i - 1][m];

                lhsp_[k][j][i2][1] = lhsp_[k][j][i2][1] - lhsp_[k][j][i2][0] * lhsp_[k][j][i - 1][3];
                lhsp_[k][j][i2][2] = lhsp_[k][j][i2][2] - lhsp_[k][j][i2][0] * lhsp_[k][j][i - 1][4];
                rhs[k][j][i2][m] = rhs[k][j][i2][m] - lhsp_[k][j][i2][0] * rhs[k][j][i - 1][m];

                m = 4;
                fac1 = 1.0 / lhsm_[k][j][i - 1][2];
                lhsm_[k][j][i - 1][3] = fac1*lhsm_[k][j][i - 1][3];
                lhsm_[k][j][i - 1][4] = fac1*lhsm_[k][j][i - 1][4];
                rhs[k][j][i - 1][m] = fac1*rhs[k][j][i - 1][m];
                lhsm_[k][j][i1][2] = lhsm_[k][j][i1][2] - lhsm_[k][j][i1][1] * lhsm_[k][j][i - 1][3];
                lhsm_[k][j][i1][3] = lhsm_[k][j][i1][3] - lhsm_[k][j][i1][1] * lhsm_[k][j][i - 1][4];
                rhs[k][j][i1][m] = rhs[k][j][i1][m] - lhsm_[k][j][i1][1] * rhs[k][j][i - 1][m];
                lhsm_[k][j][i2][1] = lhsm_[k][j][i2][1] - lhsm_[k][j][i2][0] * lhsm_[k][j][i - 1][3];
                lhsm_[k][j][i2][2] = lhsm_[k][j][i2][2] - lhsm_[k][j][i2][0] * lhsm_[k][j][i - 1][4];
                rhs[k][j][i2][m] = rhs[k][j][i2][m] - lhsm_[k][j][i2][0] * rhs[k][j][i - 1][m];

                if (i == nx2)
                {
                    m = 3;
                    fac1 = 1.0 / lhsp_[k][j][i1][2];
                    lhsp_[k][j][i1][3] = fac1 * lhsp_[k][j][i1][3];
                    lhsp_[k][j][i1][4] = fac1 * lhsp_[k][j][i1][4];
                    rhs[k][j][i1][m] = fac1 * rhs[k][j][i1][m];

                    lhsp_[k][j][i2][2] = lhsp_[k][j][i2][2] - lhsp_[k][j][i2][1] * lhsp_[k][j][i1][3];
                    lhsp_[k][j][i2][3] = lhsp_[k][j][i2][3] - lhsp_[k][j][i2][1] * lhsp_[k][j][i1][4];
                    rhs[k][j][i2][m] = rhs[k][j][i2][m] - lhsp_[k][j][i2][1] * rhs[k][j][i1][m];

                    m = 4;
                    fac1 = 1.0 / lhsm_[k][j][i1][2];
                    lhsm_[k][j][i1][3] = fac1 * lhsm_[k][j][i1][3];
                    lhsm_[k][j][i1][4] = fac1 * lhsm_[k][j][i1][4];
                    rhs[k][j][i1][m] = fac1*rhs[k][j][i1][m];

                    lhsm_[k][j][i2][2] = lhsm_[k][j][i2][2] - lhsm_[k][j][i2][1] * lhsm_[k][j][i1][3];
                    lhsm_[k][j][i2][3] = lhsm_[k][j][i2][3] - lhsm_[k][j][i2][1] * lhsm_[k][j][i1][4];
                    rhs[k][j][i2][m] = rhs[k][j][i2][m] - lhsm_[k][j][i2][1] * rhs[k][j][i1][m];

                    rhs[k][j][i2][3] = rhs[k][j][i2][3] / lhsp_[k][j][i2][2];
                    rhs[k][j][i2][4] = rhs[k][j][i2][4] / lhsm_[k][j][i2][2];

                    for (m = 0; m < 3; m++)
                        rhs[k][j][i1][m] = rhs[k][j][i1][m] - lhs_[k][j][i1][3] * rhs[k][j][i2][m];

                    rhs[k][j][i1][3] = rhs[k][j][i1][3] - lhsp_[k][j][i1][3] * rhs[k][j][i2][3];
                    rhs[k][j][i1][4] = rhs[k][j][i1][4] - lhsm_[k][j][i1][3] * rhs[k][j][i2][4];
                }
			}
		}
	}

	//x_solve_kernel_three<<<blocks, threads>>>((double*)lhs_gpu, (double*)lhsp_gpu, (double*)lhsm_gpu, (double*)gpuRhs, (double*)gpuRho_i, (double*)gpuUs, (double*)gpuSpeed, c3c4, dx2,  con43,  dx5, c1c5, dx1, dttx2, dttx1, dxmax, c2dttx1, comz1, comz4, comz5, comz6, nx2, ny2, nz2, nx);

	//CudaSafeCall(cudaMemcpy(rho_i, gpuRho_i, size, cudaMemcpyDeviceToHost));
	//CudaSafeCall(cudaMemcpy(us, gpuUs, size, cudaMemcpyDeviceToHost));
	//CudaSafeCall(cudaMemcpy(speed, gpuSpeed, size, cudaMemcpyDeviceToHost));
//	CudaSafeCall(cudaMemcpy(rhs, gpuRhs, size5, cudaMemcpyDeviceToHost));
//	CudaSafeCall(cudaMemcpy(lhs_, lhs_gpu, size5, cudaMemcpyDeviceToHost));
//	CudaSafeCall(cudaMemcpy(lhsp_, lhsp_gpu, size5, cudaMemcpyDeviceToHost));
	//CudaSafeCall(cudaMemcpy(lhsm_, lhsm_gpu, size5, cudaMemcpyDeviceToHost));

	CudaSafeCall(cudaMemcpy(lhs_gpu, lhs_, size5, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(lhsp_gpu, lhsp_, size5, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(lhsm_gpu, lhsm_, size5, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(gpuRhs, rhs, size5, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(gpuRho_i, rho_i, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(gpuUs, us, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(gpuSpeed, speed, size, cudaMemcpyHostToDevice));

	 for (k = 1; k <= nz2; k++)
    {        
        for (j = 1; j <= ny2; j++)
        {
			for (i = nx2; i >= 1; i--)
		    {
		        i1 = i;
		        i2 = i + 1;
		        for (m = 0; m < 3; m++)
		            rhs[k][j][i - 1][m] = rhs[k][j][i - 1][m] - lhs_[k][j][i - 1][3] * rhs[k][j][i1][m] - lhs_[k][j][i - 1][4] * rhs[k][j][i2][m];

		        rhs[k][j][i - 1][3] = rhs[k][j][i - 1][3] - lhsp_[k][j][i - 1][3] * rhs[k][j][i1][3] - lhsp_[k][j][i - 1][4] * rhs[k][j][i2][3];
		        rhs[k][j][i - 1][4] = rhs[k][j][i - 1][4] - lhsm_[k][j][i - 1][3] * rhs[k][j][i1][4] - lhsm_[k][j][i - 1][4] * rhs[k][j][i2][4];
		    }
		}
	}

	x_solve_kernel_four<<<blocks, threads>>>((double*)lhs_gpu, (double*)lhsp_gpu, (double*)lhsm_gpu, (double*)gpuRhs, (double*)gpuRho_i, (double*)gpuUs, (double*)gpuSpeed, c3c4, dx2,  con43,  dx5, c1c5, dx1, dttx2, dttx1, dxmax, c2dttx1, comz1, comz4, comz5, comz6, nx2, ny2, nz2, nx);

	CudaSafeCall(cudaMemcpy(rho_i, gpuRho_i, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(us, gpuUs, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(speed, gpuSpeed, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(rhs, gpuRhs, size5, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(lhs_, lhs_gpu, size5, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(lhsp_, lhsp_gpu, size5, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(lhsm_, lhsm_gpu, size5, cudaMemcpyDeviceToHost));


    //---------------------------------------------------------------------
    // Do the block-diagonal inversion          
    //---------------------------------------------------------------------
    double r1, r2, r3, r4, r5, t1, t2;
    if (timeron) timer_start(t_ninvr);

    for (k = 1; k <= nz2; k++)
    {
        for (j = 1; j <= ny2; j++)
        {
            for (i = 1; i <= nx2; i++)
            {
                r1 = rhs[k][j][i][0];
                r2 = rhs[k][j][i][1];
                r3 = rhs[k][j][i][2];
                r4 = rhs[k][j][i][3];
                r5 = rhs[k][j][i][4];

                t1 = bt * r3;
                t2 = 0.5 * (r4 + r5);

                rhs[k][j][i][0] = -r2;
                rhs[k][j][i][1] = r1;
                rhs[k][j][i][2] = bt * (r4 - r5);
                rhs[k][j][i][3] = -t1 + t2;
                rhs[k][j][i][4] = t1 + t2;
            }
        }
    }
    if (timeron) timer_stop(t_ninvr);
    if (timeron) timer_stop(t_xsolve);
}

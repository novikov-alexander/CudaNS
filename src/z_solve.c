#include "header.h"

//---------------------------------------------------------------------
// this function performs the solution of the approximate factorization
// step in the z-direction for all five matrix components
// simultaneously. The Thomas algorithm is employed to solve the
// systems for the z-lines. Boundary conditions are non-periodic
//---------------------------------------------------------------------



__global__ void z_solve_kernel_one(double* lhs_, double* lhsp_, double* lhsm_, int nx2, int ny2, int nz2)
{
	int m;

	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int j = threadIdx.y + blockIdx.y * blockDim.y + 1;

	//part 1
	if (j <= ny2 && i <= nx2)
    {        
        for (m = 0; m < 5; m++)
        {
            lhs_(j,i,0,m) = lhs_(j,i,nz2 + 1,m) = 0.0;
            lhsp_(j,i,0,m) = lhsp_(j,i,nz2 + 1,m) = 0.0;
            lhsm_(j,i,0,m) = lhsm_(j,i,nz2 + 1,m) = 0.0;
        }
        lhs_(j,i,0,2) = lhs_(j,i,nz2 + 1,2) = 1.0;
        lhsp_(j,i,0,2) = lhsp_(j,i,nz2 + 1,2) = 1.0;
        lhsm_(j,i,0,2) = lhsm_(j,i,nz2 + 1,2) = 1.0;    
	}
}

__global__ void z_solve_kernel_two(double* lhs_, double* lhsp_, double* lhsm_, double* rho_i, double* ws, double* speed, int nx2, int ny2, int nz2, double c3c4, double dz4, double con43, double dz5, double c1c5, double dzmax, double dz1, double dttz2, double dttz1, double c2dttz1, double comz1, double comz4, double comz5, double comz6)
{
	int m;
	double ru1, rhos1;

	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int k = threadIdx.z + blockIdx.z * blockDim.z + 1;

	if ( i <= nx2 && j <= ny2 && k <= nz2)
	{
        lhs_(j,i,k,0) = 0.0;

        ru1 = c3c4*rho_i(k - 1,j,i);
        rhos1 = max(max(dz4 + con43*ru1, dz5 + c1c5*ru1), max(dzmax + ru1, dz1));
        lhs_(j,i,k,1) = -dttz2 * ws(k - 1,j,i) - dttz1 * rhos1;

        ru1 = c3c4*rho_i(k,j,i);
        rhos1 = max(max(dz4 + con43*ru1, dz5 + c1c5*ru1), max(dzmax + ru1, dz1));
        lhs_(j,i,k,2) = 1.0 + c2dttz1 * rhos1;

        ru1 = c3c4*rho_i(k + 1,j,i);
        rhos1 = max(max(dz4 + con43*ru1, dz5 + c1c5*ru1), max(dzmax + ru1, dz1));
        lhs_(j,i,k,3) = dttz2 * ws(k + 1,j,i) - dttz1 * rhos1;
        lhs_(j,i,k,4) = 0.0;

        if (k == 1)
        {
            lhs_(j,i,k,2) = lhs_(j,i,k,2) + comz5;
            lhs_(j,i,k,3) = lhs_(j,i,k,3) - comz4;
            lhs_(j,i,k,4) = lhs_(j,i,k,4) + comz1;
        }
        else if (k == 2)
        {
            lhs_(j,i,k,1) = lhs_(j,i,k,1) - comz4;
            lhs_(j,i,k,2) = lhs_(j,i,k,2) + comz6;
            lhs_(j,i,k,3) = lhs_(j,i,k,3) - comz4;
            lhs_(j,i,k,4) = lhs_(j,i,k,4) + comz1;
        }
        else if (k == nz2 - 1)
        {
            lhs_(j,i,k,0) = lhs_(j,i,k,0) + comz1;
            lhs_(j,i,k,1) = lhs_(j,i,k,1) - comz4;
            lhs_(j,i,k,2) = lhs_(j,i,k,2) + comz6;
            lhs_(j,i,k,3) = lhs_(j,i,k,3) - comz4;
        }
        else if (k == nz2)
        {
            lhs_(j,i,k,0) = lhs_(j,i,k,0) + comz1;
            lhs_(j,i,k,1) = lhs_(j,i,k,1) - comz4;
            lhs_(j,i,k,2) = lhs_(j,i,k,2) + comz5;
        }
        else
        {
            lhs_(j,i,k,0) = lhs_(j,i,k,0) + comz1;
            lhs_(j,i,k,1) = lhs_(j,i,k,1) - comz4;
            lhs_(j,i,k,2) = lhs_(j,i,k,2) + comz6;
            lhs_(j,i,k,3) = lhs_(j,i,k,3) - comz4;
            lhs_(j,i,k,4) = lhs_(j,i,k,4) + comz1;
        }

        lhsp_(j,i,k,0) = lhs_(j,i,k,0);
        lhsp_(j,i,k,1) = lhs_(j,i,k,1) - dttz2 * speed(k - 1,j,i);
        lhsp_(j,i,k,2) = lhs_(j,i,k,2);
        lhsp_(j,i,k,3) = lhs_(j,i,k,3) + dttz2 * speed(k + 1,j,i);
        lhsp_(j,i,k,4) = lhs_(j,i,k,4);
        lhsm_(j,i,k,0) = lhs_(j,i,k,0);
        lhsm_(j,i,k,1) = lhs_(j,i,k,1) + dttz2 * speed(k - 1,j,i);
        lhsm_(j,i,k,2) = lhs_(j,i,k,2);
        lhsm_(j,i,k,3) = lhs_(j,i,k,3) - dttz2 * speed(k + 1,j,i);
        lhsm_(j,i,k,4) = lhs_(j,i,k,4);
	}
}

__global__ void z_solve_kernel_three(double* lhs_, double* lhsp_, double* lhsm_, double* rhs, double* rho_i, double* ws, double* speed, int nx2, int ny2, int nz2)
{
	int  k1, k2, m;
	double ru1, rhos1, fac1, fac2;

	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int k;

	if (i <= nx2 && j <= ny2)
	{
		for (k = 1; k <= nz2; k++)
        {
            k1 = k;
            k2 = k + 1;

            fac1 = 1.0 / lhs_(j,i,k - 1,2);
            lhs_(j,i,k - 1,3) = fac1 * lhs_(j,i,k - 1,3);
            lhs_(j,i,k - 1,4) = fac1 * lhs_(j,i,k - 1,4);
            for (m = 0; m < 3; m++)
                rhs(k - 1,j,i,m) = fac1 * rhs(k - 1,j,i,m);

            lhs_(j,i,k1,2) = lhs_(j,i,k1,2) - lhs_(j,i,k1,1) * lhs_(j,i,k - 1,3);
            lhs_(j,i,k1,3) = lhs_(j,i,k1,3) - lhs_(j,i,k1,1) * lhs_(j,i,k - 1,4);
            for (m = 0; m < 3; m++)
                rhs(k1,j,i,m) = rhs(k1,j,i,m) - lhs_(j,i,k1,1) * rhs(k - 1,j,i,m);

            lhs_(j,i,k2,1) = lhs_(j,i,k2,1) - lhs_(j,i,k2,0) * lhs_(j,i,k - 1,3);
            lhs_(j,i,k2,2) = lhs_(j,i,k2,2) - lhs_(j,i,k2,0) * lhs_(j,i,k - 1,4);
            for (m = 0; m < 3; m++)
                rhs(k2,j,i,m) = rhs(k2,j,i,m) - lhs_(j,i,k2,0) * rhs(k - 1,j,i,m);

            if (k == nz2)
            {
                fac1 = 1.0 / lhs_(j,i,k1,2);
                lhs_(j,i,k1,3) = fac1 * lhs_(j,i,k1,3);
                lhs_(j,i,k1,4) = fac1 * lhs_(j,i,k1,4);
                for (m = 0; m < 3; m++)
                    rhs(k1,j,i,m) = fac1 * rhs(k1,j,i,m);

                lhs_(j,i,k2,2) = lhs_(j,i,k2,2) - lhs_(j,i,k2,1) * lhs_(j,i,k1,3);
                lhs_(j,i,k2,3) = lhs_(j,i,k2,3) - lhs_(j,i,k2,1) * lhs_(j,i,k1,4);
                for (m = 0; m < 3; m++)
                    rhs(k2,j,i,m) = rhs(k2,j,i,m) - lhs_(j,i,k2,1) * rhs(k1,j,i,m);

                fac2 = 1.0 / lhs_(j,i,k2,2);
                for (m = 0; m < 3; m++)
                    rhs(k2,j,i,m) = fac2 * rhs(k2,j,i,m);
            }

            m = 3;
            fac1 = 1.0 / lhsp_(j,i,k - 1,2);
            lhsp_(j,i,k - 1,3) = fac1 * lhsp_(j,i,k - 1,3);
            lhsp_(j,i,k - 1,4) = fac1 * lhsp_(j,i,k - 1,4);
            rhs(k - 1,j,i,m) = fac1 * rhs(k - 1,j,i,m);

            lhsp_(j,i,k1,2) = lhsp_(j,i,k1,2) - lhsp_(j,i,k1,1) * lhsp_(j,i,k - 1,3);
            lhsp_(j,i,k1,3) = lhsp_(j,i,k1,3) - lhsp_(j,i,k1,1) * lhsp_(j,i,k - 1,4);
            rhs(k1,j,i,m) = rhs(k1,j,i,m) - lhsp_(j,i,k1,1) * rhs(k - 1,j,i,m);

            lhsp_(j,i,k2,1) = lhsp_(j,i,k2,1) - lhsp_(j,i,k2,0) * lhsp_(j,i,k - 1,3);
            lhsp_(j,i,k2,2) = lhsp_(j,i,k2,2) - lhsp_(j,i,k2,0) * lhsp_(j,i,k - 1,4);
            rhs(k2,j,i,m) = rhs(k2,j,i,m) - lhsp_(j,i,k2,0) * rhs(k - 1,j,i,m);

            m = 4;
            fac1 = 1.0 / lhsm_(j,i,k - 1,2);
            lhsm_(j,i,k - 1,3) = fac1 * lhsm_(j,i,k - 1,3);
            lhsm_(j,i,k - 1,4) = fac1 * lhsm_(j,i,k - 1,4);
            rhs(k - 1,j,i,m) = fac1 * rhs(k - 1,j,i,m);

            lhsm_(j,i,k1,2) = lhsm_(j,i,k1,2) - lhsm_(j,i,k1,1) * lhsm_(j,i,k - 1,3);
            lhsm_(j,i,k1,3) = lhsm_(j,i,k1,3) - lhsm_(j,i,k1,1) * lhsm_(j,i,k - 1,4);
            rhs(k1,j,i,m) = rhs(k1,j,i,m) - lhsm_(j,i,k1,1) * rhs(k - 1,j,i,m);

            lhsm_(j,i,k2,1) = lhsm_(j,i,k2,1) - lhsm_(j,i,k2,0) * lhsm_(j,i,k - 1,3);
            lhsm_(j,i,k2,2) = lhsm_(j,i,k2,2) - lhsm_(j,i,k2,0) * lhsm_(j,i,k - 1,4);
            rhs(k2,j,i,m) = rhs(k2,j,i,m) - lhsm_(j,i,k2,0) * rhs(k - 1,j,i,m);

            if (k == nz2)
            {
                m = 3;
                fac1 = 1.0 / lhsp_(j,i,k1,2);
                lhsp_(j,i,k1,3) = fac1 * lhsp_(j,i,k1,3);
                lhsp_(j,i,k1,4) = fac1 * lhsp_(j,i,k1,4);
                rhs(k1,j,i,m) = fac1 * rhs(k1,j,i,m);

                lhsp_(j,i,k2,2) = lhsp_(j,i,k2,2) - lhsp_(j,i,k2,1) * lhsp_(j,i,k1,3);
                lhsp_(j,i,k2,3) = lhsp_(j,i,k2,3) - lhsp_(j,i,k2,1) * lhsp_(j,i,k1,4);
                rhs(k2,j,i,m) = rhs(k2,j,i,m) - lhsp_(j,i,k2,1) * rhs(k1,j,i,m);

                m = 4;
                fac1 = 1.0 / lhsm_(j,i,k1,2);
                lhsm_(j,i,k1,3) = fac1 * lhsm_(j,i,k1,3);
                lhsm_(j,i,k1,4) = fac1 * lhsm_(j,i,k1,4);
                rhs(k1,j,i,m) = fac1 * rhs(k1,j,i,m);

                lhsm_(j,i,k2,2) = lhsm_(j,i,k2,2) - lhsm_(j,i,k2,1) * lhsm_(j,i,k1,3);
                lhsm_(j,i,k2,3) = lhsm_(j,i,k2,3) - lhsm_(j,i,k2,1) * lhsm_(j,i,k1,4);
                rhs(k2,j,i,m) = rhs(k2,j,i,m) - lhsm_(j,i,k2,1) * rhs(k1,j,i,m);

                rhs(k2,j,i,3) = rhs(k2,j,i,3) / lhsp_(j,i,k2,2);
                rhs(k2,j,i,4) = rhs(k2,j,i,4) / lhsm_(j,i,k2,2);

                for (m = 0; m < 3; m++)
                    rhs(k1,j,i,m) = rhs(k1,j,i,m) - lhs_(j,i,k1,3) * rhs(k2,j,i,m);

                rhs(k1,j,i,3) = rhs(k1,j,i,3) - lhsp_(j,i,k1,3) * rhs(k2,j,i,3);
                rhs(k1,j,i,4) = rhs(k1,j,i,4) - lhsm_(j,i,k1,3) * rhs(k2,j,i,4);
            }
        }
	}
}

__global__ void z_solve_kernel_four(double* lhs_, double* lhsp_, double* lhsm_, double* rhs, int nx2, int ny2, int nz2)
{
	int  k1, k2, m;

	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;	
	int k;

	if (j <= ny2 && i <= nx2)
	{
		for (k = nz2; k >= 1; k--)
        {
            k1 = k;
            k2 = k + 1;

            for (m = 0; m < 3; m++)
                rhs(k - 1,j,i,m) = rhs(k - 1,j,i,m) - lhs_(j,i,k - 1,3) * rhs(k1,j,i,m) - lhs_(j,i,k - 1,4) * rhs(k2,j,i,m);

            rhs(k - 1,j,i,3) = rhs(k - 1,j,i,3) - lhsp_(j,i,k - 1,3) * rhs(k1,j,i,3) - lhsp_(j,i,k - 1,4) * rhs(k2,j,i,3);
            rhs(k - 1,j,i,4) = rhs(k - 1,j,i,4) - lhsm_(j,i,k - 1,3) * rhs(k1,j,i,4) - lhsm_(j,i,k - 1,4) * rhs(k2,j,i,4);
        }
    }
}

__global__ void z_solve_inversion(double* rhs, double* us, double* vs, double* ws, double* qs, double* speed, double* u, int nx2, int ny2, int nz2, double bt, double c2iv)
{
	double t1, t2, t3, ac, xvel, yvel, zvel;
    double btuz, ac2u, uzik1, r1, r2, r3, r4, r5;

	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int k = threadIdx.z + blockIdx.z * blockDim.z + 1;

	if (i <= nx2 && j <= ny2 && k <= nz2)
	{
        xvel = us(k,j,i);
        yvel = vs(k,j,i);
        zvel = ws(k,j,i);
        ac = speed(k,j,i);

        ac2u = ac*ac;

        r1 = rhs(k,j,i,0);
        r2 = rhs(k,j,i,1);
        r3 = rhs(k,j,i,2);
        r4 = rhs(k,j,i,3);
        r5 = rhs(k,j,i,4);

        uzik1 = u(k,j,i,0);
        btuz = bt * uzik1;

        t1 = btuz / ac * (r4 + r5);
        t2 = r3 + t1;
        t3 = btuz * (r4 - r5);

        rhs(k,j,i,0) = t2;
        rhs(k,j,i,1) = -uzik1*r2 + xvel*t2;
        rhs(k,j,i,2) = uzik1*r1 + yvel*t2;
        rhs(k,j,i,3) = zvel*t2 + t3;
        rhs(k,j,i,4) = uzik1*(-xvel*r2 + yvel*r1) + qs(k,j,i) * t2 + c2iv*ac2u*t1 + zvel*t3;
	}
}

void z_solve()
{

    int i, j, k, k1, k2, m;
    double ru1, rhos1, fac1, fac2;

    const int size5 = sizeof(double)*P_SIZE*P_SIZE*P_SIZE*5;
	const int size = sizeof(double)*P_SIZE*P_SIZE*P_SIZE;

	dim3 blocks = dim3(nx2 / 32+1, ny2 / 4+1, nz2);
	dim3 threads = dim3(32, 4, 1);

    dim3 blocks2 = dim3(nx2 / 4+1, ny2);
	dim3 threads2 = dim3(4, 1);

    if (timeron) timer_start(t_zsolve);

    
    cudaDeviceSynchronize();
	z_solve_kernel_one<<<blocks2, threads2>>>((double*)lhs_gpu, (double*)lhsp_gpu, (double*)lhsm_gpu, nx2, ny2, nz2);

	cudaDeviceSynchronize();
    z_solve_kernel_two<<<blocks, threads>>>((double*) lhs_gpu, (double*) lhsp_gpu, (double*) lhsm_gpu, (double*) gpuRho_i, (double*) gpuWs, (double*) gpuSpeed, nx2, ny2, nz2, c3c4, dz4, con43, dz5, c1c5, dzmax, dz1, dttz2, dttz1, c2dttz1, comz1, comz4, comz5, comz6);

	cudaDeviceSynchronize();
    z_solve_kernel_three<<<blocks2, threads2>>>((double*) lhs_gpu, (double*) lhsp_gpu, (double*) lhsm_gpu, (double*) gpuRhs, (double*) gpuRho_i, (double*) gpuWs, (double*) gpuSpeed, nx2, ny2, nz2);

	cudaDeviceSynchronize();
    z_solve_kernel_four<<<blocks2, threads2>>>((double*) lhs_gpu, (double*) lhsp_gpu, (double*) lhsm_gpu, (double*) gpuRhs, nx2, ny2, nz2);

    //---------------------------------------------------------------------
    // block-diagonal matrix-vector multiplication                       
    //---------------------------------------------------------------------

    if (timeron) timer_start(t_tzetar);

	z_solve_inversion<<<blocks, threads>>>((double*)gpuRhs, (double*)gpuUs, (double*)gpuVs, (double*)gpuWs, (double*)gpuQs, (double*)gpuSpeed, (double*)gpuU, nx2, ny2, nz2, bt, c2iv);

    if (timeron) timer_stop(t_tzetar);
    if (timeron) timer_stop(t_zsolve);

}

#include "header.h"

//---------------------------------------------------------------------
// block-diagonal matrix-vector multiplication                  
//---------------------------------------------------------------------

#define u(x,y,z,m) u[m + (z) * 5 + (y) * 5 * P_SIZE + (x) * 5 * P_SIZE * P_SIZE]
#define rhs(x,y,z,m) rhs[m + (z) * 5 + (y) * 5 * P_SIZE + (x) * 5 * P_SIZE * P_SIZE]
#define rho_i(x,y,z) rho_i[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE]
#define us(x,y,z) us[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE]
#define vs(x,y,z) vs[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE]
#define ws(x,y,z) ws[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE]
#define qs(x,y,z) qs[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE]
#define speed(x,y,z) speed[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE]

__global__ void xinvr_kernel(double* rhs, double* rho_i, double* us, double* vs, double* ws, double* qs, double* speed, int nx2, int ny2, int nz2, double c2, double bt)
{
    double t1, t2, t3, ac, ru1, uu, vv, ww, r1, r2, r3, r4, r5, ac2inv;

	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int k = threadIdx.z + blockIdx.z * blockDim.z + 1;

	if(i <= nx2 && j <= ny2 && k <= nz2) 
	{
        ru1 = rho_i(k,j,i);
        uu = us(k,j,i);
        vv = vs(k,j,i);
        ww = ws(k,j,i);
        ac = speed(k,j,i);
        ac2inv = ac*ac;

        r1 = rhs(k,j,i,0);
        r2 = rhs(k,j,i,1);
        r3 = rhs(k,j,i,2);
        r4 = rhs(k,j,i,3);
        r5 = rhs(k,j,i,4);

        t1 = c2 / ac2inv * (qs(k,j,i) * r1 - uu*r2 - vv*r3 - ww*r4 + r5);
        t2 = bt * ru1 * (uu * r1 - r2);
        t3 = (bt * ru1 * ac) * t1;

        rhs(k,j,i,0) = r1 - t1;
        rhs(k,j,i,1) = -ru1 * (ww*r1 - r4);
        rhs(k,j,i,2) = ru1 * (vv*r1 - r3);
        rhs(k,j,i,3) = -t2 + t3;
        rhs(k,j,i,4) = t2 + t3;
	}
}

__global__ void add_kernel(double* u, double* rhs, int nx2, int ny2, int nz2)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int k = threadIdx.z + blockIdx.z * blockDim.z + 1;
	if(i <= nx2 && j <= ny2 && k <= nz2)
	{
		for (int m=0; m<5; m++)
		{
			u(k,j,i,m) += rhs(k,j,i,m);
		}
		
	}
}
#undef u
#undef rhs
#undef rho_i
#undef us
#undef vs
#undef ws
#undef qs
#undef speed


void xinvr()
{
	const int size5 = sizeof(double)*P_SIZE*P_SIZE*P_SIZE*5;
	const int size = sizeof(double)*P_SIZE*P_SIZE*P_SIZE;
	dim3 blocks = dim3(nx2 / 32+1, ny2 / 4+1, nz2);
	dim3 threads = dim3(32, 4, 1);

	CudaSafeCall(cudaMemcpy(gpuRhs, rhs, size5, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(gpuRho_i, rho_i, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(gpuUs, us, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(gpuVs, vs, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(gpuWs, ws, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(gpuQs, qs, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(gpuSpeed, speed, size, cudaMemcpyHostToDevice));

    if (timeron) timer_start(t_txinvr);

	xinvr_kernel<<<blocks, threads>>>((double*)gpuRhs, (double*)gpuRho_i, (double*)gpuUs, (double*)gpuVs, (double*)gpuWs, (double*)gpuQs, (double*)gpuSpeed, nx2, ny2, nz2, c2, bt);

    if (timeron) timer_stop(t_txinvr);

	CudaSafeCall(cudaMemcpy(rho_i, gpuRho_i, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(us, gpuUs, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(vs, gpuVs, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(ws, gpuWs, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(qs, gpuQs, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(speed, gpuSpeed, size, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(rhs, gpuRhs, size5, cudaMemcpyDeviceToHost));
}

void add()
{
	const int size = sizeof(double)*P_SIZE*P_SIZE*P_SIZE*5;
	dim3 blocks = dim3(nx2 / 32+1, ny2 / 4+1, nz2);
	dim3 threads = dim3(32, 4, 1);
	CudaSafeCall(cudaMemcpy(gpuU, u, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(gpuRhs, rhs, size, cudaMemcpyHostToDevice));

    if (timeron) timer_start(t_add);

	add_kernel<<<blocks, threads>>>((double*)gpuU, (double*)gpuRhs, nx2, ny2, nz2);
	
	if (timeron) timer_stop(t_add);

	CudaSafeCall(cudaMemcpy(u, gpuU, size, cudaMemcpyDeviceToHost));
}

void adi()
{
    compute_rhs(); //+
    xinvr(); //+
    x_solve();
    y_solve();
    z_solve();
    add(); //+
}

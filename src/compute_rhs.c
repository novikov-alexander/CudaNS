#include <math.h>
#include "header.h"

#define u(x,y,z,m) u[m + (z) * 5 + (y) * 5 * P_SIZE + (x) * 5 * P_SIZE * P_SIZE]
#define rhs(x,y,z,m) rhs[m + (z) * 5 + (y) * 5 * P_SIZE + (x) * 5 * P_SIZE * P_SIZE]
#define forcing(x,y,z,m) forcing[m + (z) * 5 + (y) * 5 * P_SIZE + (x) * 5 * P_SIZE * P_SIZE]
#define rho_i(x,y,z) rho_i[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE]
#define us(x,y,z) us[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE]
#define vs(x,y,z) vs[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE]
#define ws(x,y,z) ws[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE]
#define qs(x,y,z) qs[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE]
#define square(x,y,z) square[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE]
#define speed(x,y,z) speed[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE]

__global__ void compute_rhs_xyz(double* u, double* rhs, double* rho_i, double* us, double* vs, double* ws, double* qs, double* square, double* speed, double* forcing, int nx, int ny, int nz, double c1c2)
{
	int m;
	double rho_inv, aux, uijk, up1, um1;

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;

	if(i < nx && j < ny && k < nz)
	{
		rho_inv = 1.0 / u(k,j,i,0);
		rho_i(k,j,i) = rho_inv; 
		us(k,j,i) = u(k,j,i,1) * rho_inv;
		vs(k,j,i) = u(k,j,i,2) * rho_inv;
		ws(k,j,i) = u(k,j,i,3) * rho_inv;
		square(k,j,i) = 0.5* (u(k,j,i,1) * u(k,j,i,1) + u(k,j,i,2) * u(k,j,i,2) + u(k,j,i,3) * u(k,j,i,3)) * rho_inv;
		qs(k,j,i) = square(k,j,i) * rho_inv;
		aux = c1c2*rho_inv* (u(k,j,i,4) - square(k,j,i));
		speed(k,j,i) = sqrt(aux);

		for (m = 0; m < 5; m++)
		    rhs(k,j,i,m) = forcing(k,j,i,m);
	}
}


__global__ void compute_rhs_x2y2z2(double* u, double* rhs, double* rho_i, double* us, double* vs, double* ws, double* qs, double* square, double* speed, double* forcing, int nx2, int ny2, int nz2, double dx1tx1, double tx2, double dx2tx1, double con43, double c2, double dx3tx1, double xxcon2, double dx4tx1, double dx5tx1, double xxcon3, double xxcon4, double xxcon5, double dssp, double c1, double dy1ty1, double ty2, double dy2ty1, double yycon2, double dy3ty1, double dy4ty1, double dy5ty1, double yycon3, double yycon4, double yycon5, double dz1tz1, double tz2, double zzcon2, double zzcon3, double zzcon4, double zzcon5, double dz2tz1, double dz3tz1, double dz4tz1, double dz5tz1, double dt)
{
	int m;
	double rho_inv, aux, uijk, up1, um1, vijk, vp1, vm1, wijk, wm1, wp1;

	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int k = threadIdx.z + blockIdx.z * blockDim.z + 1;

	//second part
	if(i <= nx2 && j <= ny2 && k <= nz2) 
	{
        uijk = us(k,j,i);
        up1 = us(k,j,i + 1);
        um1 = us(k,j,i - 1);

        rhs(k,j,i,0) = rhs(k,j,i,0) + dx1tx1 *
            (u(k,j,i + 1,0) - 2.0*u(k,j,i,0) + u(k,j,i - 1,0)) -
            tx2 * (u(k,j,i + 1,1) - u(k,j,i - 1,1));

        rhs(k,j,i,1) = rhs(k,j,i,1) + dx2tx1 *
            (u(k,j,i + 1,1) - 2.0*u(k,j,i,1) + u(k,j,i - 1,1)) +
            xxcon2*con43 * (up1 - 2.0*uijk + um1) -
            tx2 * (u(k,j,i + 1,1) * up1 - u(k,j,i - 1,1) * um1 +
            (u(k,j,i + 1,4) - square(k,j,i + 1) -
            u(k,j,i - 1,4) + square(k,j,i - 1)) * c2);

        rhs(k,j,i,2) = rhs(k,j,i,2) + dx3tx1 *
            (u(k,j,i + 1,2) - 2.0*u(k,j,i,2) + u(k,j,i - 1,2)) +
            xxcon2 * (vs(k,j,i + 1) - 2.0*vs(k,j,i) + vs(k,j,i - 1)) -
            tx2 * (u(k,j,i + 1,2) * up1 - u(k,j,i - 1,2) * um1);

        rhs(k,j,i,3) = rhs(k,j,i,3) + dx4tx1 *
            (u(k,j,i + 1,3) - 2.0*u(k,j,i,3) + u(k,j,i - 1,3)) +
            xxcon2 * (ws(k,j,i + 1) - 2.0*ws(k,j,i) + ws(k,j,i - 1)) -
            tx2 * (u(k,j,i + 1,3) * up1 - u(k,j,i - 1,3) * um1);

        rhs(k,j,i,4) = rhs(k,j,i,4) + dx5tx1 *
            (u(k,j,i + 1,4) - 2.0*u(k,j,i,4) + u(k,j,i - 1,4)) +
            xxcon3 * (qs(k,j,i + 1) - 2.0*qs(k,j,i) + qs(k,j,i - 1)) +
            xxcon4 * (up1*up1 - 2.0*uijk*uijk + um1*um1) +
            xxcon5 * (u(k,j,i + 1,4) * rho_i(k,j,i + 1) -
            2.0*u(k,j,i,4) * rho_i(k,j,i) +
            u(k,j,i - 1,4) * rho_i(k,j,i - 1)) -
            tx2 * ((c1*u(k,j,i + 1,4) - c2*square(k,j,i + 1))*up1 -
            (c1*u(k,j,i - 1,4) - c2*square(k,j,i - 1))*um1);

        if (i == 1)
        {
            for (m = 0; m < 5; m++)
                rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (5.0*u(k,j,i,m) - 4.0*u(k,j,i + 1,m) + u(k,j,i + 2,m));
        }
        else if (i == 2)
        {
            for (m = 0; m < 5; m++)
                rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (-4.0*u(k,j,i - 1,m) + 6.0*u(k,j,i,m) - 4.0*u(k,j,i + 1,m) + u(k,j,i + 2,m));
        }
        else if (i == nx2 - 1)
        {
            for (m = 0; m < 5; m++)
                rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (u(k,j,i - 2,m) - 4.0*u(k,j,i - 1,m) + 6.0*u(k,j,i,m) - 4.0*u(k,j,i + 1,m));
        }
        else if (i == nx2)
        {
            for (m = 0; m < 5; m++)
                rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (u(k,j,i - 2,m) - 4.0*u(k,j,i - 1,m) + 5.0*u(k,j,i,m));
        }
        else
        {
            for (m = 0; m < 5; m++)
                rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (u(k,j,i - 2,m) - 4.0*u(k,j,i - 1,m) + 6.0*u(k,j,i,m) - 4.0*u(k,j,i + 1,m) + u(k,j,i + 2,m));
        }
	}


	//third part
	if(i <= nx2 && j <= ny2 && k <= nz2) 
	{
        vijk = vs(k,j,i);
        vp1 = vs(k,j + 1,i);
        vm1 = vs(k,j - 1,i);

        rhs(k,j,i,0) = rhs(k,j,i,0) + dy1ty1 *
            (u(k,j + 1,i,0) - 2.0*u(k,j,i,0) + u(k,j - 1,i,0)) -
            ty2 * (u(k,j + 1,i,2) - u(k,j - 1,i,2));

        rhs(k,j,i,1) = rhs(k,j,i,1) + dy2ty1 *
            (u(k,j + 1,i,1) - 2.0*u(k,j,i,1) + u(k,j - 1,i,1)) +
            yycon2 * (us(k,j + 1,i) - 2.0*us(k,j,i) + us(k,j - 1,i)) -
            ty2 * (u(k,j + 1,i,1) * vp1 - u(k,j - 1,i,1) * vm1);

        rhs(k,j,i,2) = rhs(k,j,i,2) + dy3ty1 *
            (u(k,j + 1,i,2) - 2.0*u(k,j,i,2) + u(k,j - 1,i,2)) +
            yycon2*con43 * (vp1 - 2.0*vijk + vm1) -
            ty2 * (u(k,j + 1,i,2) * vp1 - u(k,j - 1,i,2) * vm1 +
            (u(k,j + 1,i,4) - square(k,j + 1,i) -
            u(k,j - 1,i,4) + square(k,j - 1,i)) * c2);

        rhs(k,j,i,3) = rhs(k,j,i,3) + dy4ty1 *
            (u(k,j + 1,i,3) - 2.0*u(k,j,i,3) + u(k,j - 1,i,3)) +
            yycon2 * (ws(k,j + 1,i) - 2.0*ws(k,j,i) + ws(k,j - 1,i)) -
            ty2 * (u(k,j + 1,i,3) * vp1 - u(k,j - 1,i,3) * vm1);

        rhs(k,j,i,4) = rhs(k,j,i,4) + dy5ty1 *
            (u(k,j + 1,i,4) - 2.0*u(k,j,i,4) + u(k,j - 1,i,4)) +
            yycon3 * (qs(k,j + 1,i) - 2.0*qs(k,j,i) + qs(k,j - 1,i)) +
            yycon4 * (vp1*vp1 - 2.0*vijk*vijk + vm1*vm1) +
            yycon5 * (u(k,j + 1,i,4) * rho_i(k,j + 1,i) -
            2.0*u(k,j,i,4) * rho_i(k,j,i) +
            u(k,j - 1,i,4) * rho_i(k,j - 1,i)) -
            ty2 * ((c1*u(k,j + 1,i,4) - c2*square(k,j + 1,i)) * vp1 -
            (c1*u(k,j - 1,i,4) - c2*square(k,j - 1,i)) * vm1);

        if (j == 1)
        {
            for (m = 0; m < 5; m++)
                rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (5.0*u(k,j,i,m) - 4.0*u(k,j + 1,i,m) + u(k,j + 2,i,m));
        }
        else if (j == 2)
        {
            for (m = 0; m < 5; m++)
                rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (-4.0*u(k,j - 1,i,m) + 6.0*u(k,j,i,m) - 4.0*u(k,j + 1,i,m) + u(k,j + 2,i,m));
        }
        else if (j == ny2 - 1)
        {
            for (m = 0; m < 5; m++)
                rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (u(k,j - 2,i,m) - 4.0*u(k,j - 1,i,m) + 6.0*u(k,j,i,m) - 4.0*u(k,j + 1,i,m));
        }
        else if (j == ny2)
        {
            for (m = 0; m < 5; m++)
                rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (u(k,j - 2,i,m) - 4.0*u(k,j - 1,i,m) + 5.0*u(k,j,i,m));
        }
        else
        {

            for (m = 0; m < 5; m++)
                rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (u(k,j - 2,i,m) - 4.0*u(k,j - 1,i,m) + 6.0*u(k,j,i,m) - 4.0*u(k,j + 1,i,m) + u(k,j + 2,i,m));
        }
  	}

	//forth part
	if(i <= nx2 && j <= ny2 && k <= nz2) 
	{
        wijk = ws(k,j,i);
        wp1 = ws(k + 1,j,i);
        wm1 = ws(k - 1,j,i);

        rhs(k,j,i,0) = rhs(k,j,i,0) + dz1tz1 *	
			(u(k + 1,j,i,0) - 2.0*u(k,j,i,0) + u(k - 1,j,i,0)) -
			tz2 * (u(k + 1,j,i,3) - u(k - 1,j,i,3));

        rhs(k,j,i,1) = rhs(k,j,i,1) + dz2tz1 *
            (u(k + 1,j,i,1) - 2.0*u(k,j,i,1) + u(k - 1,j,i,1)) +
            zzcon2 * (us(k + 1,j,i) - 2.0*us(k,j,i) + us(k - 1,j,i)) -
            tz2 * (u(k + 1,j,i,1) * wp1 - u(k - 1,j,i,1) * wm1);

        rhs(k,j,i,2) = rhs(k,j,i,2) + dz3tz1 *
            (u(k + 1,j,i,2) - 2.0*u(k,j,i,2) + u(k - 1,j,i,2)) +
            zzcon2 * (vs(k + 1,j,i) - 2.0*vs(k,j,i) + vs(k - 1,j,i)) -
            tz2 * (u(k + 1,j,i,2) * wp1 - u(k - 1,j,i,2) * wm1);

        rhs(k,j,i,3) = rhs(k,j,i,3) + dz4tz1 *
            (u(k + 1,j,i,3) - 2.0*u(k,j,i,3) + u(k - 1,j,i,3)) +
            zzcon2*con43 * (wp1 - 2.0*wijk + wm1) -
            tz2 * (u(k + 1,j,i,3) * wp1 - u(k - 1,j,i,3) * wm1 +
            (u(k + 1,j,i,4) - square(k + 1,j,i) -
            u(k - 1,j,i,4) + square(k - 1,j,i)) * c2);

        rhs(k,j,i,4) = rhs(k,j,i,4) + dz5tz1 *
            (u(k + 1,j,i,4) - 2.0*u(k,j,i,4) + u(k - 1,j,i,4)) +
            zzcon3 * (qs(k + 1,j,i) - 2.0*qs(k,j,i) + qs(k - 1,j,i)) +
            zzcon4 * (wp1*wp1 - 2.0*wijk*wijk + wm1*wm1) +
            zzcon5 * (u(k + 1,j,i,4) * rho_i(k + 1,j,i) -
            2.0*u(k,j,i,4) * rho_i(k,j,i) +
            u(k - 1,j,i,4) * rho_i(k - 1,j,i)) -
            tz2 * ((c1*u(k + 1,j,i,4) - c2*square(k + 1,j,i))*wp1 -
            (c1*u(k - 1,j,i,4) - c2*square(k - 1,j,i))*wm1);

        if (k == 1)
        {
            for (m = 0; m < 5; m++)
                rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (5.0*u(k,j,i,m) - 4.0*u(k + 1,j,i,m) + u(k + 2,j,i,m));
        }
        else if (k == 2)
        {
            for (m = 0; m < 5; m++)
                rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (-4.0*u(k - 1,j,i,m) + 6.0*u(k,j,i,m) - 4.0*u(k + 1,j,i,m) + u(k + 2,j,i,m));
        }
        else if (k == nz2 - 1)
        {
            for (m = 0; m < 5; m++)
                rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (u(k - 2,j,i,m) - 4.0*u(k - 1,j,i,m) + 6.0*u(k,j,i,m) - 4.0*u(k + 1,j,i,m));
        }
        else if (k == nz2)
        {
            for (m = 0; m < 5; m++)
                rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (u(k - 2,j,i,m) - 4.0*u(k - 1,j,i,m) + 5.0*u(k,j,i,m));
        }
        else
        {
            for (m = 0; m < 5; m++)
                rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (u(k - 2,j,i,m) - 4.0*u(k - 1,j,i,m) + 6.0*u(k,j,i,m) - 4.0*u(k + 1,j,i,m) + u(k + 2,j,i,m));
        }
	}

	if(i <= nx2 && j <= ny2 && k <= nz2) 
	{
        for (m = 0; m < 5; m++)
            rhs(k,j,i,m) = rhs(k,j,i,m) * dt;
	}
}
#undef u
#undef rhs
#undef forcing
#undef rho_i
#undef us
#undef vs
#undef ws
#undef qs
#undef square
#undef speed


void compute_rhs()
{
	const int size5 = sizeof(double)*P_SIZE*P_SIZE*P_SIZE*5;
	const int size = sizeof(double)*P_SIZE*P_SIZE*P_SIZE;

	dim3 blocks = dim3(nx / 32+1, ny / 4+1, nz);
	dim3 threads = dim3(32, 4, 1);

	CudaSafeCall(cudaMemcpy(gpuU, u, size5, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(gpuRhs, rhs, size5, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(gpuRho_i, rho_i, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(gpuUs, us, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(gpuVs, vs, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(gpuWs, ws, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(gpuQs, qs, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(gpuSquare, square, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(gpuSpeed, speed, size, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(gpuForcing, forcing, size5, cudaMemcpyHostToDevice));

    if (timeron) timer_start(t_rhs);

	compute_rhs_xyz<<<blocks, threads>>>((double*)gpuU, (double*)gpuRhs, (double*)gpuRho_i, (double*)gpuUs, (double*)gpuVs, (double*)gpuWs, (double*)gpuQs,
										 (double*)gpuSquare, (double*)gpuSpeed, (double*)gpuForcing, nx, ny, nz, c1c2);
	
	blocks = dim3(nx2 / 32+1, ny2 / 4+1, nz2);
	threads = dim3(32, 4, 1);

    cudaDeviceSynchronize();
    if (timeron) timer_start(t_rhsx);


	compute_rhs_x2y2z2<<<blocks, threads>>>((double*)gpuU, (double*)gpuRhs, (double*)gpuRho_i, (double*)gpuUs, (double*)gpuVs, (double*)gpuWs, (double*)gpuQs,
									 	(double*)gpuSquare, (double*)gpuSpeed, (double*)gpuForcing, 
									 	 nx2, ny2, nz2, dx1tx1, tx2, dx2tx1, con43, c2, dx3tx1, xxcon2, dx4tx1, dx5tx1, xxcon3, xxcon4, xxcon5, dssp, c1, 
										 dy1ty1, ty2, dy2ty1, yycon2, dy3ty1, dy4ty1, dy5ty1, yycon3, yycon4, yycon5, 
										 dz1tz1, tz2, zzcon2, zzcon3, zzcon4, zzcon5, dz2tz1, dz3tz1, dz4tz1, dz5tz1, dt);

	if (timeron) timer_stop(t_rhsx);
	if (timeron) timer_stop(t_rhs);

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
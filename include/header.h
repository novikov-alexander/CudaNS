#pragma once
#include <stdbool.h>
#include "data_params.h"
#include <cstdio>

#define t_total     1
#define t_rhsx      2
#define t_rhsy      3
#define t_rhsz      4
#define t_rhs       5
#define t_xsolve    6
#define t_ysolve    7
#define t_zsolve    8
#define t_txinvr    9
#define t_pinvr     10
#define t_ninvr     11
#define t_tzetar    12
#define t_add       13
#define t_last      13

#define min(x,y) ((x) < (y) ? (x) : (y))
#define max(x,y) ((x) > (y) ? (x) : (y))

#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

typedef bool logical;

extern int nx2, ny2, nz2, nx, ny, nz;
extern logical timeron;

extern double tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3, 
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

extern double (*u)       [P_SIZE][P_SIZE][P_SIZE];
extern double (*us)      [P_SIZE][P_SIZE];
extern double (*vs)      [P_SIZE][P_SIZE];
extern double (*ws)      [P_SIZE][P_SIZE];
extern double (*qs)      [P_SIZE][P_SIZE];
extern double (*rho_i)   [P_SIZE][P_SIZE];
extern double (*speed)   [P_SIZE][P_SIZE];
extern double (*square)  [P_SIZE][P_SIZE];
extern double (*rhs)     [P_SIZE][P_SIZE][P_SIZE];
extern double (*forcing) [P_SIZE][P_SIZE][P_SIZE];

extern double (*gpuU)	 	[P_SIZE][P_SIZE][P_SIZE];
extern double (*gpuUs)      [P_SIZE][P_SIZE];
extern double (*gpuVs)      [P_SIZE][P_SIZE];
extern double (*gpuWs)      [P_SIZE][P_SIZE];
extern double (*gpuQs)      [P_SIZE][P_SIZE];
extern double (*gpuRho_i)   [P_SIZE][P_SIZE];
extern double (*gpuSpeed)   [P_SIZE][P_SIZE];
extern double (*gpuSquare)  [P_SIZE][P_SIZE];
extern double (*gpuRhs)  	[P_SIZE][P_SIZE][P_SIZE];
extern double (*gpuForcing) [P_SIZE][P_SIZE][P_SIZE];
extern double (*gpuTmp)	 	[P_SIZE][P_SIZE][P_SIZE];

extern double (*lhs_gpu)[P_SIZE][P_SIZE][P_SIZE];
extern double (*lhsp_gpu)[P_SIZE][P_SIZE][P_SIZE];
extern double (*lhsm_gpu)[P_SIZE][P_SIZE][P_SIZE];

extern double (*gpuTmpL)	 	[P_SIZE][P_SIZE][P_SIZE];
extern double (*gpuTmpLm)	 	[P_SIZE][P_SIZE][P_SIZE];
extern double (*gpuTmpLp)	 	[P_SIZE][P_SIZE][P_SIZE];

//-----------------------------------------------------------------------
//initialize functions
void set_constants();
void initialize();
void exact_solution(double xi, double eta, double zeta, double dtemp[5]);
void exact_rhs();
logical inittrace(const char** t_names);
int initparameters(int argc, char **argv, int *niter);
int allocateArrays();
int deallocateArrays();

// main calculations
void adi();
void compute_rhs();
void x_solve();
void y_solve();
void z_solve();

//errors
void error_norm(double rms[5]);
void rhs_norm(double rms[5]);

//verification
void print_results(int niter, double time, logical verified, const char **timers);
void verify(int no_time_steps, logical *verified);

//timers
void timer_clear( int n );
void timer_start( int n );
void timer_stop( int n );
double timer_read( int n );
void wtime( double *);

#define lhs_(x,y,z,m) lhs_[x + (y) * P_SIZE + (z) * P_SIZE * P_SIZE + (m) * P_SIZE * P_SIZE * P_SIZE]
#define lhsm_(x,y,z,m) lhsm_[x + (y) * P_SIZE + (z) * P_SIZE * P_SIZE + (m) * P_SIZE * P_SIZE * P_SIZE]
#define lhsp_(x,y,z,m) lhsp_[x + (y) * P_SIZE + (z) * P_SIZE * P_SIZE + (m) * P_SIZE * P_SIZE * P_SIZE]
#define rhs(x,y,z,m) rhs[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE + (m) * P_SIZE * P_SIZE * P_SIZE]
#define rho_i(x,y,z) rho_i[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE]
#define us(x,y,z) us[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE]
#define ws(x,y,z) ws[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE]
#define vs(x,y,z) vs[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE]
#define qs(x,y,z) qs[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE]
#define speed(x,y,z) speed[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE]
#define u(x,y,z,m) u[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE + (m) * P_SIZE * P_SIZE * P_SIZE]
#define forcing(x,y,z,m) forcing[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE + (m) * P_SIZE * P_SIZE * P_SIZE]
#define square(x,y,z) square[z + (y) * P_SIZE + (x) * P_SIZE * P_SIZE]



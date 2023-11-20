#include "header.hpp"

void xinvr_kernel(double *rhs, double *rho_i, double *us, double *vs, double *ws, double *qs, double *speed, int nx2, int ny2, int nz2, double c2, double bt);

void add_kernel(double *u, double *rhs, int nx2, int ny2, int nz2);

void xinvr();

void add();

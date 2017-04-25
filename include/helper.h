template<typename _T>
bool check3d(_T *host, _T *device)
{
    double (*tmp)[P_SIZE][P_SIZE] = (double (*)[P_SIZE][P_SIZE]) malloc(sizeof(double) * nx * ny * nz);
    SAFE_CALL(cudaMemcpy((double*)tmp, (double*)device, nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost));
    
    bool correct = true;
    double difference = 0;
    for (int k = 0; k <= nz - 1; k++)
    {
        for (int j = 0; j <= ny - 1; j++)
        {
            for (int i = 0; i <= nx - 1; i++)
            {
                difference += fabs(host[k][j][i] - tmp[k][j][i]);
                if(host[k][j][i] != tmp[k][j][i])
                {
                    //printf("error %lf %lf\n", host[k][j][i], tmp[k][j][i]);
                    correct = false;
                }
            }
        }
    }
    difference /= (nx * ny * nz);
    std::cout << "difference: " << difference << std::endl;
    
    free(tmp);
    
    return correct;
}

template<typename _T>
bool check4d(_T *host, _T *device)
{
    double (*tmp)[P_SIZE][P_SIZE][5] = (double (*)[P_SIZE][P_SIZE][5]) malloc(sizeof(double) * nx * ny * nz * 5);
    SAFE_CALL(cudaMemcpy((double*)tmp, (double*)device, nx * ny * nz * 5 * sizeof(double), cudaMemcpyDeviceToHost));
    
    bool correct = true;
    double difference = 0;
    for (int k = 0; k <= nz - 1; k++)
    {
        for (int j = 0; j <= ny - 1; j++)
        {
            for (int i = 0; i <= nx - 1; i++)
            {
                for (int m = 0; m < 5; m++)
                {
                    difference += fabs(host[k][j][i][m] - tmp[k][j][i][m]);
                    if(host[k][j][i][m] != tmp[k][j][i][m])
                    {
                        //printf("error %lf %lf\n", host[k][j][i][m], tmp[k][j][i][m]);
                        correct = false;
                    }
                }
            }
        }
    }
    difference /= (nx * ny * nz * 5);
    std::cout << "difference: " << difference << std::endl;
    
    free(tmp);
    
    return correct;
}

void check()
{
    if(check4d(rhs, rhs_dev))
        printf("correct RHS check!\n");
    else
        printf("error in RHS check!\n");
    
    if(check3d(speed, speed_dev))
        printf("correct SPEED check!\n");
    else
        printf("error in SPEED check!\n");
    
    if(check3d(square, square_dev))
        printf("correct SQUARE check!\n");
    else
        printf("error in SQUARE check!\n");
}

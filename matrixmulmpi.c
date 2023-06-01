#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <string.h>

#define N 4

void printmatrix(double *matrix, char ptext[])
{
    printf("%s\n", ptext);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%.2f\t", matrix[i*N +j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv)
{
    int rank, size, begin, end;
    clock_t startt, endt;
    double totalTime;

    // time init
    startt = clock();

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (N % size != 0)
    {
        printf("Error, la matriz no puede dividirse por el número de procesos, procure que sea par e inferior a:%d", N);
        MPI_Finalize();
        exit(-1);
    }
    begin = rank * N / size;
    end = (rank + 1) * N / size;

    double *a = NULL;
    double *b = NULL;
    double *r = NULL;

    if (rank == 0)
    {
        // Create sub matrixes using pointers
        a = (double *)malloc(N * N * sizeof(double));
        b = (double *)malloc(N * N * sizeof(double));
        r = (double *)malloc(N * N * sizeof(double));
    

        //fill matrixes with random values
        for (int i = 0; i < N*N; i++)
        {
            a[i] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
            b[i] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
            
        }
    }
    // Scatter to distribute all values of matrix a in ranges
    MPI_Scatter(a, N * N / size, MPI_DOUBLE, a+begin*N, N * N / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Broadcast to assign b matrix to every process
    MPI_Bcast(b, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Operation for every submatrix
    for (int i = begin; i < end; i++)
    {
        for (int j = 0; j < N; j++)
        {
            r[(i-begin)*N+j] = 0.0;
            for (int k = 0; k < N; k++)
            {
                r[(i-begin)*N+j] += a[(i-begin)*N+k] * b[k*N+j];
            }
        }
    }

    // Recollect data from processes and get result on R matrix

    MPI_Gather(r, N * N / size, MPI_DOUBLE, r, N * N / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printmatrix(a, "matriz a:");
        printmatrix(b, "matriz b:");
        printmatrix(r, "Resultado");
        // Finish size time
        endt = clock();
        totalTime = ((double)(endt - startt)) / CLOCKS_PER_SEC;
        printf("Fin del programa\nTiempo de ejecución:\n%f\n", totalTime);
    }

    free(a);
    free(b);
    free(r);

    MPI_Finalize();
    
    return 0;
}
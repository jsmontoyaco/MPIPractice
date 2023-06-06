#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <string.h>

#define N 4

void printmatrix(double *matrix, int rows, int columns, char ptext[])
{
    printf("%s\n", ptext);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            printf("%.2f\t", matrix[i*columns +j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv)
{
    int rank, size, begin, end;
    clock_t startt, endt;
    double totalTime;
    double *a, *b, *r, *a_recv, *r_send;
    
    // Time start
    startt = clock();
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Size verification
    if (N % size != 0)
    {
        printf("Error, la matriz no puede dividirse por el número de procesos, procure que sea par e inferior a: %d", N);
        MPI_Finalize();
        exit(-1);
    }
    
    // Delimiters
    begin = rank * N / size;
    end = (rank + 1) * N / size;
    
    // Matrix pieces and b
    a_recv = (double *)malloc(N * (N / size) * sizeof(double));
    r_send = (double *)malloc(N * (N / size) * sizeof(double));
    b = (double *)malloc(N * N * sizeof(double));
    
    // Original matrices
    if (rank == 0)
    {
        a = (double *)malloc(N * N * sizeof(double));
        r = (double *)malloc(N * N * sizeof(double));
        
        for (int i = 0; i < N*N; i++)
        {
            a[i] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
            b[i] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
        }
    }
    
    // Distribute the matrix
    MPI_Scatter(a, N*N/size, MPI_DOUBLE, a_recv, N*N/size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Copy b to every process
    MPI_Bcast(b, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Matrix multiplication by row    
    for (int i = begin; i < end; i++)
    {
        for (int j = 0; j < N; j++)
        {
            r_send[(i-begin)*N + j] = 0.0;
            for (int k = 0; k < N; k++)
            {
                r_send[(i-begin)*N + j] += a_recv[(i-begin)*N + k] * b[j + k*N];
            }
        }
    }
    
    // Recollect results in r matrix
    MPI_Gather(r_send, N * N/size, MPI_DOUBLE, r, N * N/size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Finish
    if (rank == 0)
    {
        // Time end
        endt = clock();
        totalTime = ((double)(endt - startt)) / CLOCKS_PER_SEC;
        printf("Fin del programa\nTiempo de ejecución:\n%f\n", totalTime);
        
        printmatrix(a, N, N, "Matriz a:");
        printmatrix(b, N, N, "Matriz b:");
        printmatrix(r, N, N, "Resultado:");
        free(a);
        free(r);
    }
    
    free(a_recv);
    free(b);
    free(r_send);
    
    MPI_Finalize();
    
    return 0;
}

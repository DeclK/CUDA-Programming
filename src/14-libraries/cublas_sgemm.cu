/* A simple cublas sgemm program, and profiling it.
This is to compare with cutlass sgemm_1.cu, and see 
how much the cutlass can reach the cublas performance.
*/
#include "error.cuh" 
#include <stdio.h>
#include <cublas_v2.h>

void print_matrix(int R, int C, float* A, const char* name);
void timing(cublasHandle_t handle, const float *d_A, const float *d_B, float *d_C, const int M, const int N, const int K);

int main(void)
{
    int M = 5120;
    int K = 5120;
    int N = 4096;
    int MK = M * K;
    int KN = K * N;
    int MN = M * N;

    float *h_A = (float*) malloc(sizeof(float) * MK);
    float *h_B = (float*) malloc(sizeof(float) * KN);
    float *h_C = (float*) malloc(sizeof(float) * MN);
    for (int i = 0; i < MK; i++)
    {
        h_A[i] = i;
    }
    // print_matrix(M, K, h_A, "A");
    for (int i = 0; i < KN; i++)
    {
        h_B[i] = i;
    }
    // print_matrix(K, N, h_B, "B");
    for (int i = 0; i < MN; i++)
    {
        h_C[i] = 0;
    }

    float *g_A, *g_B, *g_C;
    CHECK(cudaMalloc((void **)&g_A, sizeof(float) * MK))
    CHECK(cudaMalloc((void **)&g_B, sizeof(float) * KN))
    CHECK(cudaMalloc((void **)&g_C, sizeof(float) * MN))

    cublasSetVector(MK, sizeof(float), h_A, 1, g_A, 1);
    cublasSetVector(KN, sizeof(float), h_B, 1, g_B, 1);
    cublasSetVector(MN, sizeof(float), h_C, 1, g_C, 1);

    cublasHandle_t handle;
    cublasCreate(&handle);
    // float alpha = 1.0;
    // float beta = 0.0;
    // cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    //     M, N, K, &alpha, g_A, M, g_B, K, &beta, g_C, M);
    timing(handle, g_A, g_B, g_C, M, N, K);
    cublasDestroy(handle);

    cublasGetVector(MN, sizeof(float), g_C, 1, h_C, 1);
    // print_matrix(M, N, h_C, "C = A x B");

    free(h_A);
    free(h_B);
    free(h_C);
    CHECK(cudaFree(g_A))
    CHECK(cudaFree(g_B))
    CHECK(cudaFree(g_C))
    return 0;
}

void timing(cublasHandle_t handle, const float *d_A, const float *d_B, float *d_C, const int M, const int N, const int K)
{
    float alpha = 1.0;
    float beta = 0.0;
    float sum = 0;
    int NUM_REPEATS = 100;
    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);
        
        cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
        
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

        sum += elapsed_time;
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));

    }
    printf("Time = %g ms.\n", sum / NUM_REPEATS);
}


void print_matrix(int R, int C, float* A, const char* name)
{
    printf("%s = \n", name);
    for (int r = 0; r < R; ++r)
    {
        for (int c = 0; c < C; ++c)
        {
            printf("%10.6f", A[c * R + r]);
        }
        printf("\n");
    }
}


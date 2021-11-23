// Matrix addition, CPU version
// gcc matrix_cpu.c -o matrix_cpu -std=c99

#include <stdio.h>

__global__ 
void matrix_add(float *a, float *b, float *c, int *N) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = idy * (*N) + idx;
	c[index] = a[index] + b[index];
}

// QUESTION: How do you calculate the index in the array, using 2-dimensional blocks?
// You have to calculate and offset two transform two-dimensional indices to one-dimensional

int main()
{
    const int N = 5128;
    int* nHandle;

    float* a = new float[N*N];
    float* aHandle;
    float* b = new float[N*N];
    float* bHandle;
    float* c = new float[N*N];
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
	for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
    {
        a[i+j*N] = 10 + i;
        b[i+j*N] = (float)j / N;
    }
    
    int outputSize = N*N*sizeof(float);
    float* outputHandle;
    
    //const int blocksize = 16;
    
    cudaMalloc( (void**)&outputHandle, outputSize);
	cudaMalloc( (void**)&aHandle, outputSize);
	cudaMalloc( (void**)&bHandle, outputSize);
	cudaMalloc( (void**)&nHandle, sizeof(int));
    dim3 dimBlock( N,N );
	dim3 dimGrid( 1, 1 );
    cudaMemcpy( aHandle, a, outputSize, cudaMemcpyHostToDevice );
    cudaMemcpy( bHandle, b, outputSize, cudaMemcpyHostToDevice );
    cudaMemcpy( nHandle, &N, sizeof(N), cudaMemcpyHostToDevice );
    
    cudaEventRecord(start, 0);
    matrix_add<<<dimGrid, dimBlock>>>(aHandle, bHandle, outputHandle, nHandle);
    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);

    cudaMemcpy( c, outputHandle, outputSize, cudaMemcpyDeviceToHost );

    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
	float theTime;
	cudaEventElapsedTime(&theTime, start, stop);

    cudaFree( aHandle );
    cudaFree( bHandle );
    cudaFree( outputHandle );
    cudaFree( nHandle );

    delete[] a;
    delete[] b;
    delete[] c;
    
	// for (int i = 0; i < N; i++)
	// {
	// 	for (int j = 0; j < N; j++)
	// 	{
	// 		printf("%0.2f ", c[i+j*N]);
	// 	}
    //     printf("\n");
    // }

    printf("\nTime consumed: %f\n", theTime);
}

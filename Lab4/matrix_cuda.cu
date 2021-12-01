// Matrix addition, CPU version
// gcc matrix_cpu.c -o matrix_cpu -std=c99

#include <stdio.h>

__global__ 
void matrix_add(float *a, float *b, float *c, int *N) 
{
    int idy = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = blockIdx.y * blockDim.y + threadIdx.y;
    int index = idy * (*N) + idx;
	c[index] = a[index] + b[index];
}

// QUESTION: How do you calculate the index in the array, using 2-dimensional blocks?
// You have to calculate and offset to transform two-dimensional indices to one-dimensional

// QUESTION: What happens if you use too many threads per block?
// The GPU itself has hardware limitations on number of threads per block. The kernel wont run.

// QUESTION: At what data size is the GPU faster than the CPU?
// Somewhere between N=32 and N=64.

// QUESTION: What block size seems like a good choice? Compared to what?
// blocksize of 32 seems to work nicely. Size = 2 is much slower, 256 even faster.
// 512 slower than 256, 256 seems like a good size.
// Since all blocks can't run in parallell, small blocks will result in near sequential execution

// QUESTION: Write down your data size, block size and timing data for the best GPU performance you can get.
// N = 1024, blocksize = 256, time = ~0.005 ms

// QUESTION: How much performance did you lose by making data accesses non-coalesced?
// Around 1 microsecond ...

int main()
{
    const int N = 1024;
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
    
    const int threadDivider = 4;
    
    cudaMalloc( (void**)&outputHandle, outputSize);
	cudaMalloc( (void**)&aHandle, outputSize);
	cudaMalloc( (void**)&bHandle, outputSize);
	cudaMalloc( (void**)&nHandle, sizeof(int));
    dim3 dimBlock( N / threadDivider, N / threadDivider );
	dim3 dimGrid( threadDivider, threadDivider );
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
    
	// for (int i = 0; i < N; i++)
	// {
	// 	for (int j = 0; j < N; j++)
	// 	{
	// 		printf("%0.2f ", c[i+j*N]);
	// 	}
    //     printf("\n");
    // }

    delete[] a;
    delete[] b;
    delete[] c;
    

    printf("\nTime consumed: %f ms, Size: %i\n", theTime, N);
}

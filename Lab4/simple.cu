// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>

// QUESTION: How many cores will simple.cu use, max, as written? How many SMs?
// Original code:
// dim3 dimBlock( blocksize, 1 );
// dim3 dimGrid( 1, 1 );
// It will use one block with 16 cores (blocksize = 16)

// QUESTION: Is the calculated square root identical to what the CPU calculates? Should we assume that this is always the case?
// The GPU will use single precision while CPU can use both single and double precision, depending on the implementation. Might not always be the same.

const int N = 16; 
const int blocksize = 16; 

__global__ 
void simple(float *c) 
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	c[idx] = sqrt((float)c[idx]);
}

int main()
{
	float *c = new float[N];	
	float *cd;
	const int size = N*sizeof(float);

	int sqrtSize = 128;
	float *sqrtsHandle;
	float* sqrts = new float[sqrtSize];
	
	for(size_t i = 0; i < sqrtSize; ++i) {
		sqrts[i] = (float)i;
	}

	cudaMalloc( (void**)&sqrtsHandle, sqrtSize*sizeof(float));

	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( sqrtSize/blocksize );	
	cudaMemcpy( sqrtsHandle, sqrts, sqrtSize*sizeof(float), cudaMemcpyHostToDevice ); 
	simple<<<dimGrid, dimBlock>>>(sqrtsHandle);

	cudaThreadSynchronize();
	cudaMemcpy( sqrts, sqrtsHandle, sqrtSize*sizeof(float), cudaMemcpyDeviceToHost ); 
	cudaFree( sqrtsHandle );
	
	for (int i = 0; i < sqrtSize; i++)
		printf("%f \n", sqrts[i]);
	printf("\n");
	delete[] c;
	delete[] sqrts; 
	printf("done\n");
	return EXIT_SUCCESS;
}

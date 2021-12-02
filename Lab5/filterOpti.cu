// Lab 5, image filters with CUDA.

// Compile with a command-line similar to Lab 4:
// nvcc filter.cu -c -arch=sm_30 -o filter.o
// g++ filter.o milli.c readppm.c -lGL -lm -lcuda -lcudart -L/usr/local/cuda/lib -lglut -o filter
// or (multicore lab)
// nvcc filter.cu -c -arch=sm_20 -o filter.o
// g++ filter.o milli.c readppm.c -lGL -lm -lcuda -L/usr/local/cuda/lib64 -lcudart -lglut -o filter

// 2017-11-27: Early pre-release, dubbed "beta".
// 2017-12-03: First official version! Brand new lab 5 based on the old lab 6.
// Better variable names, better prepared for some lab tasks. More changes may come
// but I call this version 1.0b2.
// 2017-12-04: Two fixes: Added command-lines (above), fixed a bug in computeImages
// that allocated too much memory. b3
// 2017-12-04: More fixes: Tightened up the kernel with edge clamping.
// Less code, nicer result (no borders). Cleaned up some messed up X and Y. b4

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef __APPLE__
  #include <GLUT/glut.h>
  #include <OpenGL/gl.h>
#else
  #include <GL/glut.h>
#endif
#include "readppm.h"
#include "milli.h"

// Use these for setting shared memory size.
#define maxKernelSizeX 10
#define maxKernelSizeY 10
#define blocksizex 20
#define blocksizey 20
__device__ const int BLOCKSIZE = maxKernelSizeX * 3 * maxKernelSizeY;

__global__ void filter(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey)
{ 
    // map from blockIdx to pixel position
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int trueSizeX = maxKernelSizeX; // + kernelsizex/2;
    const int trueSizeY = maxKernelSizeX; // + kernelsizex/2;

	int li = threadIdx.y*blockDim.y + threadIdx.x;
	const int sharedMemSize = (blocksizex + maxKernelSizeX) * 3 * (blocksizey + maxKernelSizeY);
    
    // Saknar överhäng från filterkärnan
	__shared__ unsigned char imagePart[sharedMemSize];
    
    int xx = min(max(x, 0), imagesizex-1);
    int yy = min(max(y, 0), imagesizey-1);

    imagePart[li*3 + 0] = image[((yy)*imagesizex + (xx))*3+0];
    imagePart[li*3 + 1] = image[((yy)*imagesizex + (xx))*3+1];
	imagePart[li*3 + 2] = image[((yy)*imagesizex + (xx))*3+2];
	
    __syncthreads();
    
    // if(threadIdx.x == 0 && threadIdx.y == 0) {
    //     for(int i = 0; i < BLOCKSIZE + maxKernelSizeX + maxKernelSizeY; ++i)
    //         printf("%u \n", imagePart[i]);
    // }

    int dy, dx;
    unsigned int sumx, sumy, sumz;

    int divby = (2*kernelsizex+1)*(2*kernelsizey+1); // Works for box filters only!
	
	if (x < imagesizex && y < imagesizey) // If inside image
	{
	// Filter kernel (simple box filter)
	sumx=0;sumy=0;sumz=0;
    for(dy=-kernelsizey;dy<=kernelsizey;dy++)
		for(dx=-kernelsizex;dx<=kernelsizex;dx++)
		{
			//int i = (li + dx) * 3 + (dy * (blocksizex + kernelsizex) * 3);
			int i = min(max((li + dx) * 3 + (dy * (blocksizex + kernelsizex * 2) * 3), 0), (blocksizex + kernelsizex * 2) * 3 * (blocksizey + kernelsizey * 2) - 3);
			// // Use max and min to avoid branching!
			//int xx = min(max(threadIdx.x + dx, 0), imagesizex-1);
			//int yy = min(max((threadIdx.y*blockDim.y) + dy, 0), imagesizey-1);
			
			// sumx += imagePart[((yy)*imagesizex+(xx))*3+0];
			// sumy += imagePart[((yy)*imagesizex+(xx))*3+1];
            // sumz += imagePart[((yy)*imagesizex+(xx))*3+2];

			sumx += imagePart[i + 0];
			sumy += imagePart[i + 1];
			sumz += imagePart[i + 2];

            //if(threadIdx.x == 0)
				//printf("%u\n", imagePart[li]);
				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

        }
        //__syncthreads();
        out[(y*imagesizex+x)*3+0] = sumx/divby;
        out[(y*imagesizex+x)*3+1] = sumy/divby;
        out[(y*imagesizex+x)*3+2] = sumz/divby;
	}
}

// __global__ void MatrixMultOptimized( float* A, float* B, float* C, int theSize)
// {
// int k, b, gx, gy, gi, bx, by, gia, gib, li;
// // Global index for thread
// gx = blockIdx.x * blockDim.x + threadIdx.x;
// gy = blockIdx.y * blockDim.y + threadIdx.y;
// gi = gy*theSize + gx;
// // Local index for thread
// li = threadIdx.y*blockDim.y + threadIdx.x;
// float sum = 0.0;
// // for all source blocks
// for (b = 0; b < gridDim.x; b++) // We assume that gridDimx and y are equal
// {
// __shared__ float As[BLOCKSIZE*BLOCKSIZE];
// __shared__ float Bs[BLOCKSIZE*BLOCKSIZE];
// bx = blockDim.x*b + threadIdx.x; // modified x for A
// by = blockDim.y*b + threadIdx.y; // modified y for B
// gia = gy*theSize+bx; // resulting global index into A
// gib = by*theSize+gx; // resulting global index into B
// As[li] = A[gia];
// Bs[li] = B[gib];
// __syncthreads(); // Synchronize to make sure all data is loaded
// // Loop in block
// for (k = 0; k < blockDim.x; k++)
// sum += As[threadIdx.y*blockDim.x + k] * Bs[k*blockDim.x + threadIdx.x];
// __syncthreads(); // Synch again so nobody starts loading data before all finish
// }
// C[gi] = sum;


// Global variables for image data

unsigned char *image, *pixels, *dev_bitmap, *dev_input;
unsigned int imagesizey, imagesizex; // Image size

////////////////////////////////////////////////////////////////////////////////
// main computation function
////////////////////////////////////////////////////////////////////////////////
void computeImages(int kernelsizex, int kernelsizey)
{
	if (kernelsizex > maxKernelSizeX || kernelsizey > maxKernelSizeY)
	{
		printf("Kernel size out of bounds!\n");
		return;
	}

	pixels = (unsigned char *) malloc(imagesizex*imagesizey*3);
	cudaMalloc( (void**)&dev_input, imagesizex*imagesizey*3);
	cudaMemcpy( dev_input, image, imagesizey*imagesizex*3, cudaMemcpyHostToDevice );
	cudaMalloc( (void**)&dev_bitmap, imagesizex*imagesizey*3);
    dim3 block(blocksizex + (2 * kernelsizex), blocksizey + (2 * kernelsizey));
    dim3 grid(imagesizex / blocksizex, imagesizey / blocksizey);
	filter<<<grid,block>>>(dev_input, dev_bitmap, imagesizex, imagesizey, kernelsizex, kernelsizey); // Awful load balance
	cudaThreadSynchronize();
//	Check for errors!
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
	cudaMemcpy( pixels, dev_bitmap, imagesizey*imagesizex*3, cudaMemcpyDeviceToHost );
	cudaFree( dev_bitmap );
	cudaFree( dev_input );
}

// Display images
void Draw()
{
// Dump the whole picture onto the screen.	
	glClearColor( 0.0, 0.0, 0.0, 1.0 );
	glClear( GL_COLOR_BUFFER_BIT );

	if (imagesizey >= imagesizex)
	{ // Not wide - probably square. Original left, result right.
		glRasterPos2f(-1, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image );
		glRasterPos2i(0, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE,  pixels);
	}
	else
	{ // Wide image! Original on top, result below.
		glRasterPos2f(-1, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, pixels );
		glRasterPos2i(-1, 0);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image );
	}
	glFlush();
}

// Main program, inits
int main( int argc, char** argv) 
{
	glutInit(&argc, argv);
	glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );

	if (argc > 1)
		image = readppm(argv[1], (int *)&imagesizex, (int *)&imagesizey);
	else
		image = readppm((char *)"maskros512.ppm", (int *)&imagesizex, (int *)&imagesizey);

	if (imagesizey >= imagesizex)
		glutInitWindowSize( imagesizex*2, imagesizey );
	else
		glutInitWindowSize( imagesizex, imagesizey*2 );
	glutCreateWindow("Lab 5");
	glutDisplayFunc(Draw);

	ResetMilli();

	computeImages(2, 2);

// You can save the result to a file like this:
//	writeppm("out.ppm", imagesizey, imagesizex, pixels);

	glutMainLoop();
	return 0;
}

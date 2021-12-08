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
#include <iostream>

// Use these for setting shared memory size.
#define maxKernelSizeX 10
#define maxKernelSizeY 10
#define blocksizex 5
#define blocksizey 5
int filterChoice = -1;
enum choice {
	BOX_LP = 0,
	SEP_LP,
	GAUSSIAN,
	MEDIAN,
};

// QUESTION: How much data did you put in shared memory?
// The whole "block" and it's overlap around the edges

// QUESTION: How much data does each thread copy to shared memory?
// Each pixel copies one pixel into the shared memory

// QUESTION: How did you handle the necessary overlap between the blocks?
// Clever offsets using local+global thread indexing and kernel sizes

// QUESTION: If we would like to increase the block size, 
// about how big blocks would be safe to use in this case? Why?
// Should probably not exceed maxKernelSizeX*maxKernelSizeY

// QUESTION: How much speedup did you get over the naive version? For what filter size?
// Optimised GPU time: ~0.13 ms (5x5 filter size)
// Naive time: ~2.35 ms

// QUESTION: Is your access to global memory coalesced? What should you do to get that?
// The memory (hopefully) is coalesced since the image data is continuous and threads
// access the data in a increasing fashion through offsets

// QUESTION: How much speedup did you get over the non-separated? For what filter size?
// 5x5 compared to 5x1, 1x5: About the same
// 9x9 compared to 9x1, 1x9: Speed is doubled

// QUESTION: Compare the visual result to that of the box filter. 
// Is the image LP-filtered with the weighted kernel noticeably better?
// More detail is present in the gaussian filtered image.

// QUESTION: What was the difference in time to a box filter of the same size (5x5)?
// About twice as slow. Probably due to extra work and memory uploads.

// QUESTION: If you want to make a weighted kernel customizable by weights from the host, how would you deliver the weights to the GPU?
// As we do right now, upload them to the GPU.

// QUESTION: What kind of algorithm did you implement for finding the median?
// Bubble sort ...

// QUESTION: What filter size was best for reducing noise?
// 3x3

__global__ void filterGaussian(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey, unsigned char* gaussianWeights) {
	// map from blockIdx to pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x - (kernelsizex + 2*blockIdx.x*kernelsizex);
	int y = blockIdx.y * blockDim.y + threadIdx.y - (kernelsizey + 2*blockIdx.y*kernelsizey);

	int yy = min(max(y, 0), imagesizey-1);
	int xx = min(max(x, 0), imagesizex-1);

	const int sharedMemSize = (blocksizex + maxKernelSizeX * 2) * 3 * (blocksizey + maxKernelSizeY * 2);
	const int sharedMemSizeLoop = (blocksizex + kernelsizex * 2) * 3 * (blocksizey + kernelsizey * 2);
	__shared__ unsigned char imagePart[sharedMemSize];

	int li = 3 * (threadIdx.y*blockDim.x + threadIdx.x);

	// Access global memory
	imagePart[li + 0] = image[((yy)*imagesizex+(xx))*3+0];
	imagePart[li + 1] = image[((yy)*imagesizex+(xx))*3+1];
	imagePart[li + 2] = image[((yy)*imagesizex+(xx))*3+2];

	__syncthreads();

  int dy, dx;
  unsigned int sumx, sumy, sumz;

  int divby = 16; // Gaussian
	
	if (x < imagesizex && y < imagesizey 
		&& threadIdx.x >= kernelsizex && threadIdx.x < blocksizex + kernelsizex
		&& threadIdx.y >= kernelsizey && threadIdx.y < blocksizey + kernelsizey) // If inside image
	{
// Filter kernel (simple box filter)
	sumx=0;sumy=0;sumz=0;
	int weightsIdx = 0;
	for(dy=-kernelsizey;dy<=kernelsizey;dy++)
		for(dx=-kernelsizex;dx<=kernelsizex;dx++)
		{
			// Use max and min to avoid branching!
			int dd = (dx * 3) + ((blocksizex + 2*kernelsizex) * 3 * dy);
			int i = min(max(li - dd, 0), sharedMemSizeLoop - 1);
			
			sumx += imagePart[i + 0] * gaussianWeights[weightsIdx];
			sumy += imagePart[i + 1] * gaussianWeights[weightsIdx];
			sumz += imagePart[i + 2] * gaussianWeights[weightsIdx++];
		}
	out[(y*imagesizex+x)*3+0] = sumx/divby;
	out[(y*imagesizex+x)*3+1] = sumy/divby;
	out[(y*imagesizex+x)*3+2] = sumz/divby;
	}
}

__global__ void filterMedian(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey)
{
  // map from blockIdx to pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x - (kernelsizex + 2*blockIdx.x*kernelsizex);
	int y = blockIdx.y * blockDim.y + threadIdx.y - (kernelsizey + 2*blockIdx.y*kernelsizey);

	int yy = min(max(y, 0), imagesizey-1);
	int xx = min(max(x, 0), imagesizex-1);

	const int sharedMemSize = (blocksizex + maxKernelSizeX * 2) * 3 * (blocksizey + maxKernelSizeY * 2);
	const int sharedMemSizeLoop = (blocksizex + kernelsizex * 2) * 3 * (blocksizey + kernelsizey * 2);
	__shared__ unsigned char imagePart[sharedMemSize];

	int li = 3 * (threadIdx.y*blockDim.x + threadIdx.x);

	// Access global memory
	imagePart[li + 0] = image[((yy)*imagesizex+(xx))*3+0];
	imagePart[li + 1] = image[((yy)*imagesizex+(xx))*3+1];
	imagePart[li + 2] = image[((yy)*imagesizex+(xx))*3+2];

	__syncthreads();

   	int dy, dx;
   	unsigned int sumx, sumy, sumz;

   	int divby = (2*kernelsizex+1)*(2*kernelsizey+1); // Works for box filters only!
	
	int colorArraySizes = (kernelsizex * 2 + 1) * (kernelsizey * 2 + 1);
	unsigned char r[sharedMemSize];
	unsigned char g[sharedMemSize];
	unsigned char b[sharedMemSize];

	if (x < imagesizex && y < imagesizey 
		&& threadIdx.x >= kernelsizex && threadIdx.x < blocksizex + kernelsizex
		&& threadIdx.y >= kernelsizey && threadIdx.y < blocksizey + kernelsizey) // If inside image
	{
// Filter kernel (simple box filter)
	sumx=0;sumy=0;sumz=0;
	int pixelIdx = 0;
	for(dy=-kernelsizey;dy<=kernelsizey;dy++)
		for(dx=-kernelsizex;dx<=kernelsizex;dx++)	
		{
			// Use max and min to avoid branching!
			int dd = (dx * 3) + ((blocksizex + 2*kernelsizex) * 3 * dy);
			int i = min(max(li - dd, 0), sharedMemSizeLoop - 1);
			
			r[pixelIdx] = imagePart[i + 0];
			g[pixelIdx] = imagePart[i + 1];
			b[pixelIdx++] = imagePart[i + 2];
		}
	unsigned char rMax, rMin, gMax, gMin, bMax, bMin;
	for(int i=0; i < pixelIdx - 1; ++i) {
		for(int j=i+1; j < pixelIdx; ++j ) {
			rMax = max(r[i], r[j]);
			if(rMax != r[j]) {
				rMin = min(r[i], r[j]);
				r[i] = rMin;
				r[j] = rMax;
			}
			
			gMax = max(g[i], g[j]);
			if(gMax != g[j]) {
				gMin = min(g[i], g[j]);
				g[i] = gMin;
				g[j] = gMax;
			}

			bMax = max(b[i], b[j]);
			if(bMax != b[j]) {
				bMin = min(b[i], b[j]);
				b[i] = bMin;
				b[j] = bMax;
			}
		
			// Basically no difference lol
			// if(r[i] > r[j]) {
			// 	unsigned char tmp = r[i];
			// 	r[i] = r[j];
			// 	r[j] = tmp;
			// }
			// if(g[i] > g[j]) {
			// 	unsigned char tmp = g[i];
			// 	g[i] = g[j];
			// 	g[j] = tmp;
			// }
			// if(b[i] > b[j]) {
			// 	unsigned char tmp = b[i];
			// 	b[i] = b[j];
			// 	b[j] = tmp;
			// }
		}
	}
	out[(y*imagesizex+x)*3+0] = r[colorArraySizes/2];
	out[(y*imagesizex+x)*3+1] = g[colorArraySizes/2];
	out[(y*imagesizex+x)*3+2] = b[colorArraySizes/2];
	}
}

__global__ void filter(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey)
{
  // map from blockIdx to pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x - (kernelsizex + 2*blockIdx.x*kernelsizex);
	int y = blockIdx.y * blockDim.y + threadIdx.y - (kernelsizey + 2*blockIdx.y*kernelsizey);

	int yy = min(max(y, 0), imagesizey-1);
	int xx = min(max(x, 0), imagesizex-1);

	const int sharedMemSize = (blocksizex + maxKernelSizeX * 2) * 3 * (blocksizey + maxKernelSizeY * 2);
	const int sharedMemSizeLoop = (blocksizex + kernelsizex * 2) * 3 * (blocksizey + kernelsizey * 2);
	__shared__ unsigned char imagePart[sharedMemSize];

	int li = 3 * (threadIdx.y*blockDim.x + threadIdx.x);

	// Access global memory
	imagePart[li + 0] = image[((yy)*imagesizex+(xx))*3+0];
	imagePart[li + 1] = image[((yy)*imagesizex+(xx))*3+1];
	imagePart[li + 2] = image[((yy)*imagesizex+(xx))*3+2];

	__syncthreads();

  int dy, dx;
  unsigned int sumx, sumy, sumz;

  int divby = (2*kernelsizex+1)*(2*kernelsizey+1); // Works for box filters only!
	
	if (x < imagesizex && y < imagesizey 
		&& threadIdx.x >= kernelsizex && threadIdx.x < blocksizex + kernelsizex
		&& threadIdx.y >= kernelsizey && threadIdx.y < blocksizey + kernelsizey) // If inside image
	{
// Filter kernel (simple box filter)
	sumx=0;sumy=0;sumz=0;
	for(dy=-kernelsizey;dy<=kernelsizey;dy++)
		for(dx=-kernelsizex;dx<=kernelsizex;dx++)	
		{
			// Use max and min to avoid branching!
			int dd = (dx * 3) + ((blocksizex + 2*kernelsizex) * 3 * dy);
			int i = min(max(li - dd, 0), sharedMemSizeLoop - 1);
			
			sumx += imagePart[i + 0];
			sumy += imagePart[i + 1];
			sumz += imagePart[i + 2];
		}
	out[(y*imagesizex+x)*3+0] = sumx/divby;
	out[(y*imagesizex+x)*3+1] = sumy/divby;
	out[(y*imagesizex+x)*3+2] = sumz/divby;
	}
}

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

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	unsigned char* gaussianWeightsHandle;
	unsigned char* gaussianWeights;
	if(filterChoice == GAUSSIAN) {
		gaussianWeights = new unsigned char[kernelsizex*2+1];
		gaussianWeights[0] = 1;
		gaussianWeights[1] = 4;
		gaussianWeights[2] = 6;
		gaussianWeights[3] = 4;
		gaussianWeights[4] = 1;

		cudaMalloc( (void**)&gaussianWeightsHandle, sizeof(unsigned char)*5);
		cudaMemcpy( gaussianWeightsHandle, gaussianWeights, sizeof(unsigned char)*5, cudaMemcpyHostToDevice );
	}

	pixels = (unsigned char *) malloc(imagesizex*imagesizey*3);
	cudaMalloc( (void**)&dev_input, imagesizex*imagesizey*3);
	cudaMemcpy( dev_input, image, imagesizey*imagesizex*3, cudaMemcpyHostToDevice );
	cudaMalloc( (void**)&dev_bitmap, imagesizex*imagesizey*3);
	
	dim3 grid(imagesizex/blocksizex + 1,imagesizey/blocksizey + 1);
	dim3 block(blocksizex + kernelsizex * 2, blocksizey + kernelsizey * 2);
	cudaEventRecord(start, 0);
	if(filterChoice == GAUSSIAN)
		filterGaussian<<<grid,block>>>(dev_input, dev_bitmap, imagesizex, imagesizey, kernelsizex, kernelsizey, gaussianWeightsHandle);
	else if(filterChoice == MEDIAN)
		filterMedian<<<grid,block>>>(dev_input, dev_bitmap, imagesizex, imagesizey, kernelsizex, kernelsizey);
	else
		filter<<<grid,block>>>(dev_input, dev_bitmap, imagesizex, imagesizey, kernelsizex, kernelsizey);
	cudaThreadSynchronize();
	
	if(filterChoice == SEP_LP) {
		dim3 block(blocksizey + kernelsizey * 2, blocksizex + kernelsizex * 2);
		filter<<<grid,block>>>(dev_bitmap, dev_bitmap, imagesizex, imagesizey, kernelsizey, kernelsizex);
	}
	if(filterChoice == GAUSSIAN) {
		dim3 block(blocksizey + kernelsizey * 2, blocksizex + kernelsizex * 2);
		filterGaussian<<<grid,block>>>(dev_bitmap, dev_bitmap, imagesizex, imagesizey, kernelsizey, kernelsizex, gaussianWeightsHandle);
	}
	cudaThreadSynchronize();
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);

	float theTime;
	cudaEventElapsedTime(&theTime, start, stop);
	printf("Elapsed time: %f ms\n", theTime);

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
	printf("Enter filter choice:\n 0: Box low-pass\n 1: Separable low-pass\n 2: Gaussian\n 3: Median\n");
	std::cin >> filterChoice;	
	printf("Choice: %i\n", filterChoice);

	int kernelSize = 1;
	printf("Enter kernel size (1-10)\n");
	std::cin >> kernelSize;

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


	if(filterChoice == BOX_LP)
		computeImages(kernelSize, kernelSize);
	else if(filterChoice == SEP_LP)
		computeImages(kernelSize, 0);
	else if(filterChoice == GAUSSIAN)
		computeImages(kernelSize, 0);
	else if(filterChoice == MEDIAN)
		computeImages(kernelSize, kernelSize);
	else
		printf("Error\n");

// You can save the result to a file like this:
//	writeppm("out.ppm", imagesizey, imagesizex, pixels);

	glutMainLoop();
	return 0;
}

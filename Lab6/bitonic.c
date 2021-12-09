// Laboration in OpenCL. Based on a lab by Jens Ogniewski and Ingemar Ragnemalm 2010-2011.
// Rewritten by Ingemar 2017.

// Compilation line for Linux:
// gcc -std=c99 bitonic.c -o bitonic milli.c CLutilities.c -lOpenCL -I/usr/local/cuda/include/

// C implementation included.
// The OpenCL kernel is just a placeholder.
// Implement Bitonic Merge Sort in OpenCL!

// standard utilities and system includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef __APPLE__
  #include <OpenCL/opencl.h>
  #include <GLUT/glut.h>
  #include <OpenGL/gl.h>
#else
  #include <CL/cl.h>
  #include <GL/glut.h>
#endif
#include "CLutilities.h"
#include "milli.h"

// Size of data!
#define kDataLength 131072
//#define kDataLength 2048
#define MAXPRINTSIZE 16

unsigned int *generateRandomData(unsigned int length)
{
  unsigned int seed;
  struct timeval t_s;
  gettimeofday(&t_s, NULL);
  seed = (unsigned int)t_s.tv_usec;
//  printf("\nseed: %u\n",seed);

  unsigned int *data, i;

  data = (unsigned int *)malloc(length*sizeof(unsigned int));
  if (!data)
  {
    printf("\nerror allocating data.\n\n");
    return NULL;
  }
  srand(seed);
  for (i=0; i<length; i++)
    data[i] = (unsigned int)(rand()%length);
    printf("generateRandomData done.\n\n");
  return data;
}

// ------------ GPU ------------

// Kernel run conveniently packed. Edit as needed, i.e. with more parameters.
// Only ONE array of data.
// __kernel void sort(__global unsigned int *data, const unsigned int length)
void runKernel(cl_kernel kernel, int threads, cl_mem data, unsigned int length, int outerLength, int innerLength)
{
	size_t localWorkSize, globalWorkSize;
	cl_int ciErrNum = CL_SUCCESS;
	
	// Some reasonable number of blocks based on # of threads
	if (threads<512) localWorkSize  = threads;
	else            localWorkSize  = 512;
		globalWorkSize = threads;
	
	// set the args values
	ciErrNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem),  (void *) &data);
	ciErrNum |= clSetKernelArg(kernel, 1, sizeof(cl_uint), (void *) &length);
  ciErrNum |= clSetKernelArg(kernel, 2, sizeof(cl_int), (void *) &outerLength);
  ciErrNum |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void *) &innerLength);
	printCLError(ciErrNum,8);
	
	// Run kernel
	cl_event event;
	ciErrNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, &event);
	printCLError(ciErrNum,9);
	
	// Synch
	clWaitForEvents(1, &event);
	printCLError(ciErrNum,10);
}

static cl_kernel gpgpuSort;

int bitonic_gpu(unsigned int *data, unsigned int length)
{
	cl_int ciErrNum = CL_SUCCESS;
	size_t localWorkSize, globalWorkSize;
	cl_mem io_data;
	printf("GPU sorting.\n");

	io_data = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, length * sizeof(unsigned int), data, &ciErrNum);
	printCLError(ciErrNum,7);

	// ********** RUN THE KERNEL ************
  ResetMilli();
  for (int k = 2; k <= length; k = 2*k) // Outer loop, double size for each step
    for (int j = k >> 1; j > 0; j = j >> 1) // Inner loop, half size for each step
	    runKernel(gpgpuSort, length, io_data, length, k, j);
  printf("GPU %f ms\n", GetSeconds()*1000);

	// Get data
	cl_event event;
	ciErrNum = clEnqueueReadBuffer(commandQueue, io_data, CL_TRUE, 0, length * sizeof(unsigned int), data, 0, NULL, &event);
	printCLError(ciErrNum,11);
	// Synch
	clWaitForEvents(1, &event);
	printCLError(ciErrNum,10);
  
	clReleaseMemObject(io_data);
	return ciErrNum;
}

// ------------ CPU ------------

static void exchange(unsigned int *i, unsigned int *j)
{
	int k;
	k = *i;
	*i = *j;
	*j = k;
}

void bitonic_cpu(unsigned int *data, int N)
{
  unsigned int i,j,k;

  printf("CPU sorting.\n");

  for (k = 2; k <= N; k = 2*k) // Outer loop, double size for each step
  {
    for (j = k >> 1; j > 0; j = j >> 1) // Inner loop, half size for each step
    {
      for (i=0;i<N;i++) // Loop over data
      {
        int ixj=i^j; // Calculate indexing!
        if ((ixj)>i)
        {
          if ((i & k) == 0 && data[i] > data[ixj]) exchange(&data[i], &data[ixj]);
          if ((i & k) != 0 && data[i] < data[ixj]) exchange(&data[i], &data[ixj]);
        }
      }
    }
  }
}

// ------------ main ------------

int main( int argc, char** argv) 
{
  int length = kDataLength; // SIZE OF DATA
  unsigned short int header[2];
  
  // Computed data
  unsigned int *data_cpu, *data_gpu;
  
  // Find a platform and device
  if (initOpenCL()<0)
  {
    closeOpenCL();
    return 1;
  }
  // Load and compile the kernel
  gpgpuSort = compileKernel("bitonic.cl", "bitonic");

  data_cpu = generateRandomData(length);
  data_gpu = (unsigned int *)malloc (length*sizeof(unsigned int));

  if ((!data_cpu)||(!data_gpu))
  {
    printf("\nError allocating data.\n\n");
    return 1;
  }
  
  // Copy to gpu data.
  for(int i=0;i<length;i++)
    data_gpu[i]=data_cpu[i];
  
  ResetMilli();
  bitonic_cpu(data_cpu,length);
  printf("CPU %f ms\n", GetSeconds()*1000);
  
  bitonic_gpu(data_gpu,length);  

  // Print part of result
  for (int i=0;i<MAXPRINTSIZE;i++)
    printf("%d ", data_gpu[i]);
  printf("\n");

  for (int i=0;i<length;i++)
    if (data_cpu[i] != data_gpu[i])
    {
      printf("Wrong value at position %d.\n", i);
      closeOpenCL();
      return(1);
    }
  printf("\nYour sorting looks correct!\n");
  closeOpenCL();
  if (gpgpuSort) clReleaseKernel(gpgpuSort);
  return 0;
}

// QUESTION: Should each thread produce one output or two? Why?
// Since bitonic sort is based on swapping two elements each threads must produce two outputs (two swapped elements in the array)

// QUESTION: How many items can you handle in one workgroup?
// The GPU can handle up to 1024 threads. Each threads handles two items which means if we use 1024 threads in the workgroup 2048 items can be handled in a workgroup

// QUESTION: What problem must be solved when you use more than one workgroup? How did you solve it?
// Since the outer loop takes big steps ( k = 2*k ) it could end up iterating out of a blocks designated memory.
// By doing this work on the host before calling the kernel the GPU gets exactly the memory it needs for the current partition.

// QUESTION: What time do you get? Difference to the CPU? What is the break even size? What can you expect for a parallel CPU version? (Your conclusions here may vary between the labs.)
// Break even: somewhere between kDataLength = 1024 and 2048, around ~1.2 ms
// At kDataLength = 131072: GPU = ~1.5 ms, CPU = ~80 ms. Bitonic sort scales really well in parallell on the GPU.
// A parallell CPU-version should be faster since it scales well. Although it wont reach GPU-times since the number of available threads are much lower.
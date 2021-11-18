/**************************
** TDDD56 Lab 3
***************************
** Author:
** August Ernstsson
**************************/

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <time.h>
#include <iterator>

#include <skepu>

#include "support.h"

unsigned char average_kernel(skepu::Region2D<unsigned char> m, size_t elemPerPx)
{
	float scaling = 1.0 / ((m.oj/elemPerPx*2+1)*(m.oi*2+1));
	float res = 0;
	for (int y = -m.oi; y <= m.oi; ++y)
		for (int x = -m.oj; x <= m.oj; x += elemPerPx)
			res += m(y, x);
	return res * scaling;
}

unsigned char average_kernel_1d(skepu::Region1D<unsigned char> m, size_t elemPerPx)
{
	//printf("M.oi separable: %d \n ", m.oi);
	float scaling = 1.0 / (m.oi/elemPerPx*2+1);
	float res = 0;
	for (int x = -m.oi; x <= m.oi; x += elemPerPx) {
		res += m(x);
	}
	return res * scaling;
}



unsigned char gaussian_kernel(skepu::Region1D<unsigned char> m, const skepu::Vec<float> stencil, size_t elemPerPx)
{
	//printf("M.oi separable: %d \n ", m.oi);
	float scaling = 1.0 / (m.oi / elemPerPx*2+1);
	float res = 0;
	int i = 0;
	for (int x = -m.oi; x <= m.oi; x += elemPerPx) {
		res += m(x)*stencil[i];
		++i;
	}
	return res;
}

// 2.1) Separable average if faster for all backends.
//		Separable Gaussian is faster for all but CUDA and sometimes equal for OpenCL (Might be caused by data upload to GPU)
//		Since separable filter cores are just an array instead of a matrix the data locality will be much better, hence the better performance

int main(int argc, char* argv[])
{
	if (argc < 5)
	{
		std::cout << "Usage: " << argv[0] << " input output radius [backend]\n";
		exit(1);
	}
	
	LodePNGColorType colorType = LCT_RGB;
	std::string inputFileName = argv[1];
	std::string outputFileName = argv[2];
	const int radius = atoi(argv[3]);
	auto spec = skepu::BackendSpec{argv[4]};
	skepu::setGlobalBackendSpec(spec);
	
	// Create the full path for writing the image.
	std::stringstream ss;
	ss << (2 * radius + 1) << "x" << (2 * radius + 1);
	std::string outputFile = outputFileName + ss.str();
	
	// Read the padded image into a matrix. Create the output matrix without padding.
	// Padded version for 2D MapOverlap, non-padded for 1D MapOverlap
	ImageInfo imageInfo;
	skepu::Matrix<unsigned char> inputMatrixPad = ReadAndPadPngFileToMatrix(inputFileName, radius, colorType, imageInfo);
	skepu::Matrix<unsigned char> inputMatrix = ReadPngFileToMatrix(inputFileName, colorType, imageInfo);
	skepu::Matrix<unsigned char> outputMatrix(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);
	// more containers...?
	
	// Original version
	{
		auto conv = skepu::MapOverlap(average_kernel);
		conv.setOverlap(radius, radius  * imageInfo.elementsPerPixel);
	
		auto timeTaken = skepu::benchmark::measureExecTime([&]
		{
			conv(outputMatrix, inputMatrixPad, imageInfo.elementsPerPixel);
		});
	
		WritePngFileMatrix(outputMatrix, outputFile + "-average.png", colorType, imageInfo);
		std::cout << "Time for combined: " << (timeTaken.count() / 10E6) << "\n";
	}
	
	
	// Separable version
	{
		auto conv = skepu::MapOverlap(average_kernel_1d);
		skepu::Matrix<unsigned char> outputMatrixSep(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);
		auto timeTaken = skepu::benchmark::measureExecTime([&]
		{
			conv.setOverlapMode(skepu::Overlap::RowWise);
			conv.setOverlap(radius * imageInfo.elementsPerPixel);
			conv(outputMatrix, inputMatrix, imageInfo.elementsPerPixel);
			
			conv.setOverlapMode(skepu::Overlap::ColWise);
			conv.setOverlap(radius);
			conv(outputMatrixSep, outputMatrix, 1);
		});
		
		WritePngFileMatrix(outputMatrixSep, outputFile + "-separable.png", colorType, imageInfo);
		std::cout << "Time for separable: " << (timeTaken.count() / 10E6) << "\n";
	}
	
	
	// Separable gaussian
	{
		skepu::Vector<float> stencil = sampleGaussian(radius);
			
		auto conv = skepu::MapOverlap(gaussian_kernel);
		skepu::Matrix<unsigned char> outputMatrixSepGaussian(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);
		auto timeTaken = skepu::benchmark::measureExecTime([&]
		{
			conv.setOverlapMode(skepu::Overlap::RowWise);
			conv.setOverlap(radius * imageInfo.elementsPerPixel);
			conv(outputMatrix, inputMatrix, stencil, imageInfo.elementsPerPixel);
			
			conv.setOverlapMode(skepu::Overlap::ColWise);
			conv.setOverlap(radius);
			conv(outputMatrixSepGaussian, outputMatrix, stencil, 1);
		});
	
		WritePngFileMatrix(outputMatrixSepGaussian, outputFile + "-gaussian.png", colorType, imageInfo);
		std::cout << "Time for gaussian: " << (timeTaken.count() / 10E6) << "\n";
	}
	
	
	
	return 0;
}



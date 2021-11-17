#include <skepu>

#include "lodepng.h"

// Information about the image. Used to rebuild the image after filtering.
struct ImageInfo
{
	int width;
	int height;
	int elementsPerPixel;
};

skepu::Matrix<unsigned char>
ReadPngFileToMatrix(
	std::string filePath,
	LodePNGColorType colorType,
	ImageInfo& imageInformation
);

// Reads a file from png and retuns it as a skepu::Matrix. Uses a library called LodePNG.
// Also returns information about the image as an out variable in imageInformation.
skepu::Matrix<unsigned char>
ReadAndPadPngFileToMatrix(
	std::string filePath,
	int kernelRadius,
	LodePNGColorType colorType,
	ImageInfo& imageInformation
);

void
WritePngFileMatrix(
	skepu::Matrix<unsigned char> imageData,
	std::string filePath,
	LodePNGColorType colorType,
	ImageInfo& imageInformation
);

skepu::Vector<float> sampleGaussian(int size);


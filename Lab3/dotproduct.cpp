/**************************
** TDDD56 Lab 3
***************************
** Author:
** August Ernstsson
**************************/

#include <iostream>

#include <skepu>

/* SkePU user functions */


// Unary user function
float square(float a)
{
return a * a;
}
// Binary user function
float mult(float a, float b)
{
return a * b;
}
// User function template
float add(float a, float b)
{
return a + b;
}

int main(int argc, const char* argv[])
{
	if (argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " <input size> <backend>\n";
		exit(1);
	}
	
	const size_t size = std::stoul(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
	//spec.setCPUThreads(8);
	skepu::setGlobalBackendSpec(spec);
	
	
	/* Skeleton instances */
//	auto instance = skepu::Map(userfunction);
// ...
	auto mrDot = skepu::MapReduce<2>(
		[] (float a, float b) { return a * b; },
		[] (float a, float b) { return a + b; }
	);

	auto map = skepu::Map<2>(mult);
	auto reduce = skepu::Reduce(add);
	
	/* SkePU containers */
	skepu::Vector<float> v1(size, 1.0f), v2(size, 2.0f);
	skepu::Vector<float> res(v1.size());
	
	/* Compute and measure time */
	float resComb, resSep;
	
	auto timeComb = skepu::benchmark::measureExecTimeIdempotent([&]
	{
		resComb = mrDot(v1, v2);
	});
	
	auto timeSep = skepu::benchmark::measureExecTimeIdempotent([&]
	{
		map(res, v1, v2);
		resSep = reduce(res);
	});

	// Initial measurements
	// With a vector of size 100,000,000:
	// Separate faster on CPU (CPU Sequential)
	// Combined faster with OpenMP (CPU multithreading)
	// Separate much faster with OpenCL (GPU)
	// Separate much faster with CUDA (GPU)

	// 1.1 MapReduce can reuse the same addresses in the cache for both operations, significantly improving data locality
	// 1.2 Greater control
	// 1.3 Smaller problems are faster on the low latency low throughput CPU, large problems are faster on high latency high throughput GPU
	// 1.4 CPU is the same. OpenMP switches between sep. and comb. OpenCL switched to much faster on comb. CUDA switched to much faster on comb.
	//	   measureExecTimeIdempotent disregards time needed to upload data to GPU => more representative of the true performance

	std::cout << "Time Combined: " << (timeComb.count() / 10E6) << " seconds.\n";
	std::cout << "Time Separate: " << ( timeSep.count() / 10E6) << " seconds.\n";
	
	
	std::cout << "Result Combined: " << resComb << "\n";
	std::cout << "Result Separate: " << resSep  << "\n";
	
	return 0;
}


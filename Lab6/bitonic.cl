/*
 * Placeholder OpenCL kernel
 */

__kernel void bitonic(__global unsigned int *data, const unsigned int length, int outerLength, int innerLength)
{
  unsigned int i = get_global_id(0);

  int ixj = i ^ innerLength; // Calculate indexing!

  if ((ixj) > i)
  {
    if ((i & outerLength) == 0 && data[i] > data[ixj]) {
      unsigned int tmp = data[i];
      data[i] = data[ixj];
      data[ixj] = tmp;
    }
    if ((i & outerLength) != 0 && data[i] < data[ixj]) {
      unsigned int tmp = data[i];
      data[i] = data[ixj];
      data[ixj] = tmp;
    }
  }
}

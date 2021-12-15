/*
 * Placeholder OpenCL kernel
 */

__kernel void find_max(__global unsigned int *data, const unsigned int length)
{ 
  //unsigned int pos = length/2;
  unsigned int val;

  //Something should happen here

  //while(get_global_id(0) < pos) {
    val = max(data[get_global_id(0) * 2], data[get_global_id(0) * 2 + 1]);
    barrier(CLK_LOCAL_MEM_FENCE);
    data[get_global_id(0)] = val;
    //barrier(CLK_LOCAL_MEM_FENCE);
    //pos /= 2;
  //}

  //data[get_global_id(0)] = get_global_id(0);
}

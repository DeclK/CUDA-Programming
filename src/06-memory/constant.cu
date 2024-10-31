// test const memory via func parameter
__global__ void test_const_memory_via_func_param(float *out, const float *in)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    out[idx] = in[idx];
}


struct KernelParams
{
    int size_x;
    int size_y;
    int k;
};

[AutoPyBindCUDA]
[CudaKernel]
void convolve(TensorView<float> input, TensorView<float> output, TensorView<float> kernel, KernelParams input_params)
{
    uint3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    if (globalIdx.x >= output.size(0) ||
        globalIdx.y >= output.size(1))
        return;

    int size_x = input_params.size_x;
    int size_y = input_params.size_y;
    int k = input_params.k;

    float sum_r = 0;
    float sum_g = 0;
    float sum_b = 0;
    int radius = k/2;
    for(int u=0;u<k;++u)
    {
        for(int v=0;v<k;++v)
        {
            int new_x = max(min(globalIdx.x-radius+u,output.size(0)), 0);
            int new_y = max(min(globalIdx.y-radius+v,output.size(1)), 0);
            sum_r += input[new_x,new_y,0] * kernel[u, v];
            sum_g += input[new_x,new_y,1] * kernel[u, v];
            sum_b += input[new_x,new_y,2] * kernel[u, v];
        }
    }
    output[globalIdx.x, globalIdx.y, 0] = sum_r;
    output[globalIdx.x, globalIdx.y, 1] = sum_g;
    output[globalIdx.x, globalIdx.y, 2] = sum_b;
}

[AutoPyBindCUDA]
[CudaKernel]
[Differentiable]
void mse_loss(DiffTensorView<float> input, TensorView<float> reference, DiffTensorView<float> output)
{
    uint3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    if (globalIdx.x >= output.size(0) ||
        globalIdx.y >= output.size(1))
        return;

    float diff_r = input.load(int3(globalIdx.x, globalIdx.y, 0)) - no_diff(reference[globalIdx.x, globalIdx.y, 0]);
    float diff_g = input.load(int3(globalIdx.x, globalIdx.y, 1)) - no_diff(reference[globalIdx.x, globalIdx.y, 1]);
    float diff_b = input.load(int3(globalIdx.x, globalIdx.y, 2)) - no_diff(reference[globalIdx.x, globalIdx.y, 2]);

    float mse_r = diff_r * diff_r;
    float mse_g = diff_g * diff_g;
    float mse_b = diff_b * diff_b;
    output.store(int3(globalIdx.x, globalIdx.y, 0), mse_r);
    output.store( int3(globalIdx.x, globalIdx.y, 1), mse_g);
    output.store(int3(globalIdx.x, globalIdx.y, 2), mse_b);
}

void crop_and_resize_gpu_forward(
    THCudaTensor * image,
    THCudaTensor * boxes,           // [y1, x1, y2, x2]
    THCudaIntTensor * box_index,    // range in [0, batch_size)
    const float extrapolation_value,
    const int crop_height,
    const int crop_width,
    THCudaTensor * crops
);

void crop_and_resize_gpu_backward(
    THCudaTensor * grads,
    THCudaTensor * boxes,      // [y1, x1, y2, x2]
    THCudaIntTensor * box_index,    // range in [0, batch_size)
    THCudaTensor * grads_image // resize to [bsize, c, hc, wc]
);
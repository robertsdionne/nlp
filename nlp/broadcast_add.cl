kernel void BroadcastAdd(
    uint x_dimension, constant int *x_shape, constant int *x_stride, uint x_size, const global float *x_data,
    uint b_dimension, constant int *b_shape, constant int *b_stride, uint b_size, const global float *b_data,
    uint y_dimension, constant int *y_shape, constant int *y_stride, uint y_size, global float *y_data) {
  int i = get_global_id(0), j = get_global_id(1);
  y_data[i * y_stride[0] + j * y_stride[1]] = x_data[i * x_stride[0] + j * x_stride[1]] + b_data[i * b_stride[0]];
}

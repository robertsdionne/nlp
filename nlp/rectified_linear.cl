kernel void RectifiedLinear(
    uint x_dimension, constant int *x_shape, constant int *x_stride, uint x_size, const global float *x_data,
    uint y_dimension, constant int *y_shape, constant int *y_stride, uint y_size, global float *y_data) {
  int index = get_global_id(0);
  y_data[index] = fmax(x_data[index], 0);
}

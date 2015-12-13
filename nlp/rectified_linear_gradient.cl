kernel void RectifiedLinearGradient(
    uint dy_dimension, constant int *dy_shape, constant int *dy_stride, uint dy_size, const global float *dy_data,
    uint x_dimension, constant int *x_shape, constant int *x_stride, uint x_size, const global float *x_data,
    uint dx_dimension, constant int *dx_shape, constant int *dx_stride, uint dx_size, global float *dx_data) {
  int index = get_global_id(0);
  dx_data[index] = dy_data[index] * (x_data[index] > 0);
}

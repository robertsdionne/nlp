kernel void LogisticGradient(
    uint dy_dimension, constant int *dy_shape, constant int *dy_stride, uint dy_size, const global float *dy_data,
    uint y_dimension, constant int *y_shape, constant int *y_stride, uint y_size, const global float *y_data,
    uint dx_dimension, constant int *dx_shape, constant int *dx_stride, uint dx_size, global float *dx_data) {
  int index = get_global_id(0);
  dx_data[index] = dy_data[index] * y_data[index] * (1 - y_data[index]);
}

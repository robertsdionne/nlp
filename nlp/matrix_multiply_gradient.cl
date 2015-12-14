kernel void MatrixMultiplyWGradient(
    uint dy_dimension, constant int *dy_shape, constant int *dy_stride, uint dy_size, const global float *dy_data,
    uint x_dimension, constant int *x_shape, constant int *x_stride, uint x_size, const global float *x_data,
    uint dw_dimension, constant int *dw_shape, constant int *dw_stride, uint dw_size, global float *dw_data) {
  int i = get_global_id(0), k = get_global_id(1);
  float output = 0;
  for (int j = 0; j < x_shape[1]; ++j) {
    output += dy_data[i * dy_stride[0] + j * dy_stride[1]] * x_data[k * x_stride[0] + j * x_stride[1]];
  }
  dw_data[i * dw_stride[0] + k * dw_stride[1]] = output;
}

kernel void MatrixMultiplyXGradient(
    uint dy_dimension, constant int *dy_shape, constant int *dy_stride, uint dy_size, const global float *dy_data,
    uint w_dimension, constant int *w_shape, constant int *w_stride, uint w_size, const global float *w_data,
    uint dx_dimension, constant int *dx_shape, constant int *dx_stride, uint dx_size, global float *dx_data) {
  int k = get_global_id(0), j = get_global_id(1);
  float output = 0;
  for (int i = 0; i < dy_shape[0]; ++i) {
    output += dy_data[i * dy_stride[0] + j * dy_stride[1]] * w_data[i * w_stride[0] + k * w_stride[1]];
  }
  dx_data[k * dx_stride[0] + j * dx_stride[1]] = output;
}

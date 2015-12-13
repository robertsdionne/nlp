kernel void MatrixMultiply(
    uint w_dimension, constant int *w_shape, constant int *w_stride, uint w_size, const global float *w_data,
    uint x_dimension, constant int *x_shape, constant int *x_stride, uint x_size, const global float *x_data,
    uint y_dimension, constant int *y_shape, constant int *y_stride, uint y_size, global float *y_data) {
  int i = get_global_id(0), j = get_global_id(1);
  float output = 0;
  for (int k = 0; k < x_shape[0]; ++k) {
    output += w_data[i * w_stride[0] + k * w_stride[1]] * x_data[k * x_stride[0] + j * x_stride[1]];
  }
  y_data[i * y_stride[0] + j * y_stride[1]] = output;
}

kernel void EmbeddingsGradient(
    uint dy_dimension, constant int *dy_shape, constant int *dy_stride, uint dy_size, const global float *dy_data,
    uint x_dimension, constant int *x_shape, constant int *x_stride, uint x_size, const global int *x_data,
    uint dw_dimension, constant int *dw_shape, constant int *dw_stride, uint dw_size, global float *dw_data) {
  int i = get_global_id(0), k = get_global_id(1);
  float output = 0;
  for (int j = 0; j < x_shape[0]; ++j) {
    output += dy_data[i * dy_stride[0] + j * dy_stride[1]] * (k == x_data[j * x_stride[0]]);
  }
  dw_data[i * dw_stride[0] + k * dw_stride[1]] = output;
}

kernel void BroadcastAddXGradient(
    uint dy_dimension, constant int *dy_shape, constant int *dy_stride, uint dy_size, const global float *dy_data,
    uint dx_dimension, constant int *dx_shape, constant int *dx_stride, uint dx_size, global float *dx_data) {
  int i = get_global_id(0), j = get_global_id(1);
  dx_data[i * dx_stride[0] + j * dx_stride[1]] = dy_data[i * dy_stride[0] + j * dy_stride[1]];
}

kernel void BroadcastAddBGradient(
    uint dy_dimension, constant int *dy_shape, constant int *dy_stride, uint dy_size, const global float *dy_data,
    uint db_dimension, constant int *db_shape, constant int *db_stride, uint db_size, global float *db_data) {
  int i = get_global_id(0);
  float output = 0;
  for (int j = 0; j < dy_shape[1]; ++j) {
    output += dy_data[i * dy_stride[0] + j * dy_stride[1]];
  }
  db_data[i * db_stride[0]] = output;
}

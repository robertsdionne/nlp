cc_library(
  name = 'broadcast_add',
  srcs = [
    'broadcast_add.hpp',
  ],
  deps = [
    ':kernels',
    ':tensor',
    '//opencl',
  ],
  data = [
    'broadcast_add.cl',
  ],
)

cc_library(
  name = 'broadcast_add_gradient',
  srcs = [
    'broadcast_add_gradient.hpp',
  ],
  deps = [
    ':kernels',
    ':tensor',
    '//opencl',
  ],
  data = [
    'broadcast_add_gradient.cl',
  ],
)

cc_library(
  name = 'kernels',
  srcs = [
    'kernels.hpp',
  ],
  deps = [
    ':tensor',
    '//opencl',
  ],
)

cc_library(
  name = 'logistic',
  srcs = [
    'logistic.hpp',
  ],
  deps = [
    ':kernels',
    ':tensor',
    '//opencl',
  ],
  data = [
    'logistic.cl',
  ],
)

cc_library(
  name = 'matrix_multiply',
  srcs = [
    'matrix_multiply.hpp',
  ],
  deps = [
    ':kernels',
    ':tensor',
    '//opencl',
  ],
  data = [
    'matrix_multiply.cl',
  ],
)

cc_library(
  name = 'matrix_multiply_gradient',
  srcs = [
    'matrix_multiply_gradient.hpp',
  ],
  deps = [
    ':kernels',
    ':tensor',
    '//opencl',
  ],
  data = [
    'matrix_multiply_gradient.cl',
  ],
)

cc_binary(
  name = 'nlp',
  srcs = [
    'nlp.cpp',
  ],
  deps = [
    ':broadcast_add',
    ':broadcast_add_gradient',
    ':logistic',
    ':matrix_multiply',
    ':matrix_multiply_gradient',
    ':rectified_linear',
    ':rectified_linear_gradient',
  ],
  linkopts = [
    '-framework OpenCL',
  ],
)

cc_library(
  name = 'rectified_linear',
  srcs = [
    'rectified_linear.hpp',
  ],
  deps = [
    ':kernels',
    ':tensor',
    '//opencl',
  ],
  data = [
    'rectified_linear.cl',
  ],
)

cc_library(
  name = 'rectified_linear_gradient',
  srcs = [
    'rectified_linear_gradient.hpp',
  ],
  deps = [
    ':kernels',
    ':tensor',
    '//opencl',
  ],
  data = [
    'rectified_linear_gradient.cl',
  ],
)

cc_library(
  name = 'tensor',
  srcs = [
    'tensor.hpp',
  ],
  deps = [
    '//opencl',
  ],
)

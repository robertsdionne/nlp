
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

cc_binary(
  name = 'nlp',
  srcs = [
    'nlp.cpp',
  ],
  deps = [
    ':matrix_multiply',
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

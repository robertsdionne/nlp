
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

cc_binary(
  name = 'nlp',
  srcs = [
    'nlp.cpp',
  ],
  deps = [
    ':rectified_linear',
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
  name = 'tensor',
  srcs = [
    'tensor.hpp',
  ],
  deps = [
    '//opencl',
  ],
)

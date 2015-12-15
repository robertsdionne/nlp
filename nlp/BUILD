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

cc_test(
  name = 'broadcast_add_test',
  srcs = [
    'broadcast_add_test.cpp',
  ],
  deps = [
    ':opencl',
    ':broadcast_add',
    ':broadcast_add_gradient',
    ':tensor',
    '//opencl',
    '//third_party:gtest',
  ],
)

cc_library(
  name = 'default_types',
  srcs = [
    'default_types.hpp',
  ],
)

cc_library(
  name = 'embeddings',
  srcs = [
    'embeddings.hpp',
  ],
  deps = [
    ':kernels',
    ':tensor',
    '//opencl',
  ],
  data = [
    'embeddings.cl',
  ],
)

cc_library(
  name = 'embeddings_gradient',
  srcs = [
    'embeddings_gradient.hpp',
  ],
  deps = [
    ':kernels',
    ':tensor',
    '//opencl',
  ],
  data = [
    'embeddings_gradient.cl',
  ],
)

cc_test(
  name = 'embeddings_test',
  srcs = [
    'embeddings_test.cpp',
  ],
  deps = [
    ':opencl',
    ':embeddings',
    ':embeddings_gradient',
    ':tensor',
    '//opencl',
    '//third_party:gtest',
  ],
)

cc_library(
  name = 'kernels',
  srcs = [
    'kernels.hpp',
  ],
  deps = [
    ':default_types',
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
  name = 'logistic_gradient',
  srcs = [
    'logistic_gradient.hpp',
  ],
  deps = [
    ':kernels',
    ':tensor',
    '//opencl',
  ],
  data = [
    'logistic_gradient.cl',
  ],
)

cc_test(
  name = 'logistic_test',
  srcs = [
    'logistic_test.cpp',
  ],
  deps = [
    ':opencl',
    ':logistic',
    ':logistic_gradient',
    ':tensor',
    '//opencl',
    '//third_party:gtest',
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

cc_test(
  name = 'matrix_multiply_test',
  srcs = [
    'matrix_multiply_test.cpp',
  ],
  deps = [
    ':opencl',
    ':matrix_multiply',
    ':matrix_multiply_gradient',
    ':tensor',
    '//opencl',
    '//third_party:gtest',
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
    ':embeddings',
    ':embeddings_gradient',
    ':logistic',
    ':logistic_gradient',
    ':matrix_multiply',
    ':matrix_multiply_gradient',
    ':opencl',
    ':rectified_linear',
    ':rectified_linear_gradient',
  ],
)

cc_library(
  name = 'opencl',
  srcs = [
    'opencl.hpp',
  ],
  deps = [
    '//opencl',
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

cc_test(
  name = 'rectified_linear_test',
  srcs = [
    'rectified_linear_test.cpp',
  ],
  deps = [
    ':opencl',
    ':rectified_linear',
    ':rectified_linear_gradient',
    ':tensor',
    '//opencl',
    '//third_party:gtest',
  ],
)

cc_library(
  name = 'tensor',
  srcs = [
    'tensor.hpp',
  ],
  deps = [
    ':default_types',
    '//opencl',
  ],
)

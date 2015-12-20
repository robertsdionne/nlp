#ifndef NLP_KERNELS_HPP_
#define NLP_KERNELS_HPP_

#include <cassert>
#include <fstream>
#include <iostream>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

#include "nlp/default_types.hpp"
#include "nlp/tensor.hpp"
#include "opencl/cl.hpp"

namespace nlp {

using cl::Context;
using cl::Device;
using cl::Kernel;
using std::string;
using std::vector;

string ReadSourceFile(const string &source_filename) {
  using std::ifstream;
  using std::ios;
  using std::istreambuf_iterator;

  ifstream source_file(source_filename);
  assert(source_file.good());

  string source;

  source_file.seekg(0, ios::end);
  source.reserve(source_file.tellg());
  source_file.seekg(0, ios::beg);

  source.assign(istreambuf_iterator<char>(source_file), istreambuf_iterator<char>());
  return source;
}

Kernel BuildKernel(Context &context, vector<Device> &devices, const string &source_filename, const string &name) {
  using cl::Program;
  using std::make_pair;

  cl_int error;

  auto source = ReadSourceFile(source_filename);
  auto program = Program(context, {
    make_pair(source.c_str(), source.size()),
  }, &error);
  assert(CL_SUCCESS == error);

  if (CL_SUCCESS != program.build(devices)) {
    for (auto &device : devices) {
      string build_log;
      std::cerr << "Build failed for device " << device.getInfo<CL_DEVICE_NAME>() << ": "
          << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device, &error);
      assert(CL_SUCCESS == error);
    }
  }

  auto kernel = Kernel(program, name.c_str(), &error);
  assert(CL_SUCCESS == error);

  return kernel;
}

template <typename F=default_floating_point_type, typename I=default_integer_type>
cl_int SetTensorArg(Kernel &kernel, unsigned int index, const Tensor<F, I> &tensor) {
  assert(CL_SUCCESS == kernel.setArg(5 * index + 0, tensor.shape.size()));
  assert(CL_SUCCESS == kernel.setArg(5 * index + 1, tensor.shape_buffer));
  assert(CL_SUCCESS == kernel.setArg(5 * index + 2, tensor.stride_buffer));
  assert(CL_SUCCESS == kernel.setArg(5 * index + 3, tensor.size()));
  assert(CL_SUCCESS == kernel.setArg(5 * index + 4, tensor.data_buffer));
  return CL_SUCCESS;
}

}  // namespace nlp

#endif  // NLP_KERNELS_HPP_

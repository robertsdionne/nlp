#ifndef NLP_MATRIX_MULTIPLY_GRADIENT_HPP_
#define NLP_MATRIX_MULTIPLY_GRADIENT_HPP_

#include <cassert>

#include "nlp/kernels.hpp"
#include "nlp/tensor.hpp"
#include "opencl/cl.hpp"

namespace nlp {

using cl::CommandQueue;
using cl::Context;
using cl::Device;
using cl::Kernel;
using cl::NDRange;
using cl::NullRange;
using std::vector;

class MatrixMultiplyWGradient {
public:
  MatrixMultiplyWGradient(Context &context, vector<Device> &devices, CommandQueue &command_queue):
      command_queue_(command_queue),
      kernel_(BuildKernel(context, devices, "nlp/matrix_multiply_gradient.cl", "MatrixMultiplyWGradient")) {}

  void operator ()(const Tensor<> &dy, const Tensor<> &x, Tensor<> &dw) {
    assert(CL_SUCCESS == SetTensorArg(kernel_, 0, dy));
    assert(CL_SUCCESS == SetTensorArg(kernel_, 1, x));
    assert(CL_SUCCESS == SetTensorArg(kernel_, 2, dw));
    assert(CL_SUCCESS == command_queue_.enqueueNDRangeKernel(
        kernel_, NullRange, NDRange(dw.shape.at(0), dw.shape.at(1)), NullRange));
  }

private:
  CommandQueue &command_queue_;
  Kernel kernel_;
};

class MatrixMultiplyXGradient {
public:
  MatrixMultiplyXGradient(Context &context, vector<Device> &devices, CommandQueue &command_queue):
      command_queue_(command_queue),
      kernel_(BuildKernel(context, devices, "nlp/matrix_multiply_gradient.cl", "MatrixMultiplyXGradient")) {}

  void operator ()(const Tensor<> &dy, const Tensor<> &w, Tensor<> &dx) {
    assert(CL_SUCCESS == SetTensorArg(kernel_, 0, dy));
    assert(CL_SUCCESS == SetTensorArg(kernel_, 1, w));
    assert(CL_SUCCESS == SetTensorArg(kernel_, 2, dx));
    assert(CL_SUCCESS == command_queue_.enqueueNDRangeKernel(
        kernel_, NullRange, NDRange(dx.shape.at(0), dx.shape.at(1)), NullRange));
  }

private:
  CommandQueue &command_queue_;
  Kernel kernel_;
};

}  // namespace nlp

#endif  // NLP_MATRIX_MULTIPLY_GRADIENT_HPP_

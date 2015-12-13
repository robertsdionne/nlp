#ifndef NLP_RECTIFIED_LINEAR_GRADIENT_HPP_
#define NLP_RECTIFIED_LINEAR_GRADIENT_HPP_

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

class RectifiedLinearGradient {
public:
  RectifiedLinearGradient(Context &context, vector<Device> &devices, CommandQueue &command_queue):
      command_queue_(command_queue),
      kernel_(BuildKernel(context, devices, "nlp/rectified_linear_gradient.cl", "RectifiedLinearGradient")) {}

  void operator ()(const Tensor &dy, const Tensor &x, Tensor &dx) {
    assert(CL_SUCCESS == SetTensorArg(kernel_, 0, dy));
    assert(CL_SUCCESS == SetTensorArg(kernel_, 1, x));
    assert(CL_SUCCESS == SetTensorArg(kernel_, 2, dx));
    assert(CL_SUCCESS == command_queue_.enqueueNDRangeKernel(kernel_, NullRange, NDRange(dx.data.size()), NullRange));
  }

private:
  CommandQueue &command_queue_;
  Kernel kernel_;
};

}  // namespace nlp

#endif  // NLP_RECTIFIED_LINEAR_GRADIENT_HPP_

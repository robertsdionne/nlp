#ifndef NLP_BROADCAST_ADD_GRADIENT_HPP_
#define NLP_BROADCAST_ADD_GRADIENT_HPP_

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

class BroadcastAddXGradient {
public:
  BroadcastAddXGradient(Context &context, vector<Device> &devices, CommandQueue &command_queue):
      command_queue_(command_queue),
      kernel_(BuildKernel(context, devices, "nlp/broadcast_add_gradient.cl", "BroadcastAddXGradient")) {}

  void operator ()(const Tensor<> &dy, Tensor<> &dx) {
    assert(CL_SUCCESS == SetTensorArg(kernel_, 0, dy));
    assert(CL_SUCCESS == SetTensorArg(kernel_, 1, dx));
    assert(CL_SUCCESS == command_queue_.enqueueNDRangeKernel(
        kernel_, NullRange, NDRange(dx.shape.at(0), dx.shape.at(1)), NullRange));
  }

private:
  CommandQueue &command_queue_;
  Kernel kernel_;
};

class BroadcastAddBGradient {
public:
  BroadcastAddBGradient(Context &context, vector<Device> &devices, CommandQueue &command_queue):
      command_queue_(command_queue),
      kernel_(BuildKernel(context, devices, "nlp/broadcast_add_gradient.cl", "BroadcastAddBGradient")) {}

  void operator ()(const Tensor<> &dy, Tensor<> &db) {
    assert(CL_SUCCESS == SetTensorArg(kernel_, 0, dy));
    assert(CL_SUCCESS == SetTensorArg(kernel_, 1, db));
    assert(CL_SUCCESS == command_queue_.enqueueNDRangeKernel(kernel_, NullRange, NDRange(db.shape.at(0)), NullRange));
  }

private:
  CommandQueue &command_queue_;
  Kernel kernel_;
};

}  // namespace nlp

#endif  // NLP_BROADCAST_ADD_GRADIENT_HPP_

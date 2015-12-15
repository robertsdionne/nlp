#ifndef NLP_BROADCAST_ADD_HPP_
#define NLP_BROADCAST_ADD_HPP_

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

class BroadcastAdd {
public:
  BroadcastAdd(Context &context, vector<Device> &devices, CommandQueue &command_queue):
      command_queue_(command_queue),
      kernel_(BuildKernel(context, devices, "nlp/broadcast_add.cl", "BroadcastAdd")) {}

  void operator ()(const Tensor<> &x, const Tensor<> &b, Tensor<> &y) {
    assert(CL_SUCCESS == SetTensorArg(kernel_, 0, x));
    assert(CL_SUCCESS == SetTensorArg(kernel_, 1, b));
    assert(CL_SUCCESS == SetTensorArg(kernel_, 2, y));
    assert(CL_SUCCESS == command_queue_.enqueueNDRangeKernel(
        kernel_, NullRange, NDRange(y.shape.at(0), y.shape.at(1)), NullRange));
  }

private:
  CommandQueue &command_queue_;
  Kernel kernel_;
};

}  // namespace nlp

#endif  // NLP_BROADCAST_ADD_HPP_

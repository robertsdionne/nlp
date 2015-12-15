#ifndef NLP_EMBEDDINGS_GRADIENT_HPP_
#define NLP_EMBEDDINGS_GRADIENT_HPP_

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

class EmbeddingsGradient {
public:
  EmbeddingsGradient(Context &context, vector<Device> &devices, CommandQueue &command_queue):
      command_queue_(command_queue),
      kernel_(BuildKernel(context, devices, "nlp/embeddings_gradient.cl", "EmbeddingsGradient")) {}

  void operator ()(const Tensor<> &dy, const Tensor<int> &x, Tensor<> &dw) {
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

}  // namespace nlp

#endif  // NLP_EMBEDDINGS_GRADIENT_HPP_

#ifndef NLP_TENSOR_HPP_
#define NLP_TENSOR_HPP_

#include <ostream>
#include <vector>

#include "opencl/cl.hpp"

namespace nlp {

using cl::Buffer;
using cl::CommandQueue;
using cl::Context;

using std::ostream;
using std::ostream_iterator;
using std::vector;

struct Tensor {
  void Allocate(Context &context) {
    shape_buffer = Buffer(context, CL_MEM_USE_HOST_PTR, shape.size() * sizeof(int), shape.data());
    stride_buffer = Buffer(context, CL_MEM_USE_HOST_PTR, stride.size() * sizeof(int), stride.data());
    data_buffer = Buffer(context, CL_MEM_USE_HOST_PTR, data.size() * sizeof(float), data.data());
  }

  void Read(CommandQueue &command_queue) {
    command_queue.enqueueReadBuffer(data_buffer, true, 0, data.size() * sizeof(float), data.data());
  }

  vector<int> shape, stride;
  vector<float> data;
  Buffer shape_buffer, stride_buffer, data_buffer;
};

template <typename F> ostream &operator <<(ostream &out, const vector<F> &v) {
  out << "{";
  if (!v.empty()) {
    copy(begin(v), end(v) - 1, ostream_iterator<F>(out, ", "));
    copy(end(v) - 1, end(v), ostream_iterator<F>(out));
  }
  out << "}";
  return out;
}

ostream &operator <<(ostream &out, const Tensor &tensor) {
  return out << "Tensor(" << tensor.shape << ", " << tensor.data << ")";
}

}  // namespace nlp

#endif  // NLP_TENSOR_HPP_

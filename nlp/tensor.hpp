#ifndef NLP_TENSOR_HPP_
#define NLP_TENSOR_HPP_

#include <cxxabi.h>
#include <ostream>
#include <vector>

#include "nlp/default_types.hpp"
#include "opencl/cl.hpp"

namespace nlp {

using cl::Buffer;
using cl::CommandQueue;
using cl::Context;

using std::ostream;
using std::ostream_iterator;
using std::vector;

template <typename F = default_floating_point_type, typename I = default_integer_type>
struct Tensor {
  void Allocate(Context &context) {
    shape_buffer = Buffer(context, CL_MEM_USE_HOST_PTR, shape.size() * sizeof(I), shape.data());
    stride_buffer = Buffer(context, CL_MEM_USE_HOST_PTR, stride.size() * sizeof(I), stride.data());
    data_buffer = Buffer(context, CL_MEM_USE_HOST_PTR, data.size() * sizeof(F), data.data());
  }

  Tensor<F> &Read(CommandQueue &command_queue) {
    command_queue.enqueueReadBuffer(data_buffer, true, 0, data.size() * sizeof(F), data.data());
    return *this;
  }

  vector<I> shape, stride;
  vector<F> data;
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

#define DESCRIBE_T(T) abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr)

template <typename F = default_floating_point_type, typename I = default_integer_type>
ostream &operator <<(ostream &out, const Tensor<F, I> &tensor) {
  return out << "Tensor<"
      << DESCRIBE_T(F) << ", " << DESCRIBE_T(I)
      << ">(" << tensor.shape << ", " << tensor.data << ")";
}

}  // namespace nlp

#endif  // NLP_TENSOR_HPP_

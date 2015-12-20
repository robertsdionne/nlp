#ifndef NLP_FEEDFORWARD_MODEL_HPP_
#define NLP_FEEDFORWARD_MODEL_HPP_

#include "nlp/broadcast_add.hpp"
#include "nlp/broadcast_add_gradient.hpp"
#include "nlp/embeddings.hpp"
#include "nlp/embeddings_gradient.hpp"
#include "nlp/logistic.hpp"
#include "nlp/logistic_gradient.hpp"
#include "nlp/matrix_multiply.hpp"
#include "nlp/matrix_multiply_gradient.hpp"
#include "nlp/rectified_linear.hpp"
#include "nlp/rectified_linear_gradient.hpp"
#include "nlp/tensor.hpp"
#include "opencl/cl.hpp"

namespace nlp {

using cl::CommandQueue;
using cl::Context;
using cl::Device;

using std::vector;

class FeedforwardModel {
public:
  FeedforwardModel(Context &context, vector<Device> &devices, CommandQueue &command_queue):
      context_(context),
      command_queue_(command_queue),
      embeddings(context, devices, command_queue),
      embeddings_gradient(context, devices, command_queue),
      matrix_multiply(context, devices, command_queue),
      matrix_multiply_w_gradient(context, devices, command_queue),
      matrix_multiply_x_gradient(context, devices, command_queue),
      broadcast_add(context, devices, command_queue),
      broadcast_add_x_gradient(context, devices, command_queue),
      broadcast_add_b_gradient(context, devices, command_queue),
      rectified_linear(context, devices, command_queue),
      rectified_linear_gradient(context, devices, command_queue),
      logistic(context, devices, command_queue),
      logistic_gradient(context, devices, command_queue) {}

  virtual ~FeedforwardModel() = default;

  void operator ()(const Tensor<int> &sentence,
      Tensor<> &word_vectors,
      Tensor<> &w0, Tensor<> &b0,
      Tensor<> &w1, Tensor<> &b1,
      Tensor<> &temporary_x,
      Tensor<> &temporary_h0,
      Tensor<> &temporary_h1,
      Tensor<> &temporary_y0,
      Tensor<> &temporary_h2,
      Tensor<> &temporary_h3,
      Tensor<> &y,
      Tensor<> &word_vectors_gradient,
      Tensor<> &w0_gradient, Tensor<> &b0_gradient,
      Tensor<> &w1_gradient, Tensor<> &b1_gradient,
      Tensor<> &temporary_y1_gradient,
      Tensor<> &temporary_h3_gradient,
      Tensor<> &temporary_h2_gradient,
      Tensor<> &temporary_y0_gradient,
      Tensor<> &temporary_h1_gradient,
      Tensor<> &temporary_h0_gradient,
      Tensor<> &temporary_x_gradient) {
    embeddings(word_vectors, sentence, temporary_x);

    matrix_multiply(w0, temporary_x, temporary_h0);
    broadcast_add(temporary_h0, b0, temporary_h1);
    rectified_linear(temporary_h1, temporary_y0);

    matrix_multiply(w1, temporary_y0, temporary_h2);
    broadcast_add(temporary_h2, b1, temporary_h3);
    logistic(temporary_h3, y);

    logistic_gradient(temporary_y1_gradient, y, temporary_h3_gradient);
    broadcast_add_x_gradient(temporary_h3_gradient, temporary_h2_gradient);
    broadcast_add_b_gradient(temporary_h3_gradient, b1_gradient);
    matrix_multiply_w_gradient(temporary_h2_gradient, temporary_y0, w1_gradient);
    matrix_multiply_x_gradient(temporary_h2_gradient, w1, temporary_y0_gradient);

    rectified_linear_gradient(temporary_y0_gradient, temporary_h1, temporary_h1_gradient);
    broadcast_add_x_gradient(temporary_h1_gradient, temporary_h0_gradient);
    broadcast_add_b_gradient(temporary_h1_gradient, b0_gradient);
    matrix_multiply_w_gradient(temporary_h0_gradient, temporary_x, w0_gradient);
    matrix_multiply_x_gradient(temporary_h0_gradient, w0, temporary_x_gradient);

    embeddings_gradient(temporary_x_gradient, sentence, word_vectors_gradient);
  }

private:
  Context &context_;
  CommandQueue &command_queue_;
  Embeddings embeddings;
  EmbeddingsGradient embeddings_gradient;
  MatrixMultiply matrix_multiply;
  MatrixMultiplyWGradient matrix_multiply_w_gradient;
  MatrixMultiplyXGradient matrix_multiply_x_gradient;
  BroadcastAdd broadcast_add;
  BroadcastAddXGradient broadcast_add_x_gradient;
  BroadcastAddBGradient broadcast_add_b_gradient;
  RectifiedLinear rectified_linear;
  RectifiedLinearGradient rectified_linear_gradient;
  Logistic logistic;
  LogisticGradient logistic_gradient;
};

}  // namespace nlp

#endif  // NLP_FEEDFORWARD_MODEL_HPP_

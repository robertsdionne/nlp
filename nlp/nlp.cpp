#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "nlp/broadcast_add.hpp"
#include "nlp/broadcast_add_gradient.hpp"
#include "nlp/embeddings.hpp"
#include "nlp/embeddings_gradient.hpp"
#include "nlp/feedforward_model.hpp"
#include "nlp/logistic.hpp"
#include "nlp/logistic_gradient.hpp"
#include "nlp/matrix_multiply.hpp"
#include "nlp/matrix_multiply_gradient.hpp"
#include "nlp/opencl.hpp"
#include "nlp/rectified_linear.hpp"
#include "nlp/rectified_linear_gradient.hpp"
#include "nlp/tensor.hpp"
#include "opencl/cl.hpp"

using namespace nlp;

int main(int argument_count, char *arguments[]) {
  using nlp::BroadcastAdd;
  using nlp::BroadcastAddXGradient;
  using nlp::BroadcastAddBGradient;
  using nlp::Embeddings;
  using nlp::EmbeddingsGradient;
  using nlp::FeedforwardModel;
  using nlp::Logistic;
  using nlp::LogisticGradient;
  using nlp::MatrixMultiply;
  using nlp::MatrixMultiplyWGradient;
  using nlp::MatrixMultiplyXGradient;
  using nlp::RectifiedLinear;
  using nlp::RectifiedLinearGradient;
  using nlp::Tensor;

  using std::cout;
  using std::endl;
  using std::find_if;
  using std::get;
  using std::string;
  using std::vector;

  auto opencl = SetupOpenCl();
  auto context = get<0>(opencl);
  auto devices = get<1>(opencl);
  auto command_queue = get<2>(opencl);

  auto feedforward_model = FeedforwardModel(context, devices, command_queue);

  auto n = 9, d = 4, v = 5;

  auto sentence = Tensor<int>{{n}, {1}, {
    0, 1, 2, 1, 2, 3, 2, 3, 4,
  }};

  auto word_vectors = Tensor<>{{d, v}, {v, 1}, {
    -0.01, 0.02, 0.03, 0.04, -0.05,
    -0.02, 0.03, 0.04, 0.05, -0.06,
    -0.03, 0.04, 0.05, -0.06, 0.07,
    -0.04, 0.05, 0.06, -0.07, 0.08,
  }},
  w0 = Tensor<>{{d, d}, {d, 1}, {
    -0.01, 0.02, -0.03, 0.04,
    -0.02, 0.03, -0.04, 0.05,
    0.03, 0.04, -0.05, 0.06,
    0.04, 0.05, -0.06, 0.07,
  }},
  b0 = Tensor<>{{d}, {1}, {
    0.01,
    -0.02,
    -0.03,
    0.04,
  }},
  w1 = Tensor<>{{d, d}, {d, 1}, {
    -0.01, 0.02, 0.03, -0.04,
    0.02, -0.03, 0.04, -0.05,
    0.03, -0.04, 0.05, -0.06,
    -0.04, 0.05, 0.06, -0.07,
  }},
  b1 = Tensor<>{{d}, {1}, {
    -0.01,
    0.02,
    -0.03,
    0.04,
  }},
  y = Tensor<>{{4, 9}, {9, 1}, vector<float>(d * n)};

  auto word_vectors_gradient = Tensor<>{{4, 5}, {5, 1}, vector<float>(d * v)},
      w0_gradient = Tensor<>{{4, 4}, {4, 1}, vector<float>(d * d)},
      b0_gradient = Tensor<>{{4}, {1}, vector<float>(d)},
      w1_gradient = Tensor<>{{4, 4}, {4, 1}, vector<float>(d * d)},
      b1_gradient = Tensor<>{{4}, {1}, vector<float>(d)};

  sentence.Allocate(context, command_queue);
  word_vectors.Allocate(context, command_queue);
  w0.Allocate(context, command_queue);
  b0.Allocate(context, command_queue);
  w1.Allocate(context, command_queue);
  b1.Allocate(context, command_queue);
  y.Allocate(context, command_queue);

  word_vectors_gradient.Allocate(context, command_queue);
  w0_gradient.Allocate(context, command_queue);
  b0_gradient.Allocate(context, command_queue);
  w1_gradient.Allocate(context, command_queue);
  b1_gradient.Allocate(context, command_queue);

  auto temporary_x = Tensor<>{{d, n}, {n, 1}};
  auto temporary_h0 = Tensor<>{{d, n}, {n, 1}};
  auto temporary_h1 = Tensor<>{{d, n}, {n, 1}};
  auto temporary_y0 = Tensor<>{{d, n}, {n, 1}};
  auto temporary_h2 = Tensor<>{{d, n}, {n, 1}};
  auto temporary_h3 = Tensor<>{{d, n}, {n, 1}};

  temporary_x.Temporary(context, command_queue);
  temporary_h0.Temporary(context, command_queue);
  temporary_h1.Temporary(context, command_queue);
  temporary_y0.Temporary(context, command_queue);
  temporary_h2.Temporary(context, command_queue);
  temporary_h3.Temporary(context, command_queue);

  auto temporary_y1_gradient = Tensor<>{{d, n}, {n, 1}, {
    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
    0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
    0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11,
    0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12,
  }};
  auto temporary_h3_gradient = Tensor<>{{d, n}, {n, 1}};
  auto temporary_h2_gradient = Tensor<>{{d, n}, {n, 1}};
  auto temporary_y0_gradient = Tensor<>{{d, n}, {n, 1}};
  auto temporary_h1_gradient = Tensor<>{{d, n}, {n, 1}};
  auto temporary_h0_gradient = Tensor<>{{d, n}, {n, 1}};
  auto temporary_x_gradient = Tensor<>{{d, n}, {n, 1}};

  temporary_y1_gradient.Allocate(context, command_queue);
  temporary_h3_gradient.Temporary(context, command_queue);
  temporary_h2_gradient.Temporary(context, command_queue);
  temporary_y0_gradient.Temporary(context, command_queue);
  temporary_h1_gradient.Temporary(context, command_queue);
  temporary_h0_gradient.Temporary(context, command_queue);
  temporary_x_gradient.Temporary(context, command_queue);

  for (auto i = 0; i < 1000; ++i) {
    feedforward_model(
        sentence, word_vectors,
        w0, b0, w1, b1,
        temporary_x,
        temporary_h0,
        temporary_h1,
        temporary_y0,
        temporary_h2,
        temporary_h3,
        y, word_vectors_gradient,
        w0_gradient, b0_gradient, w1_gradient, b1_gradient,
        temporary_y1_gradient,
        temporary_h3_gradient,
        temporary_h2_gradient,
        temporary_y0_gradient,
        temporary_h1_gradient,
        temporary_h0_gradient,
        temporary_x_gradient);
  }

  cout << y.Read(command_queue) << endl;
  cout << word_vectors_gradient.Read(command_queue) << endl;
  cout << w0_gradient.Read(command_queue) << endl;
  cout << b0_gradient.Read(command_queue) << endl;
  cout << w1_gradient.Read(command_queue) << endl;
  cout << b1_gradient.Read(command_queue) << endl;

  return 0;
}

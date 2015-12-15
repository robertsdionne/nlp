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

  return 0;
}

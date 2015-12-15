#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "nlp/opencl.hpp"
#include "nlp/embeddings.hpp"
#include "nlp/embeddings_gradient.hpp"
#include "nlp/tensor.hpp"
#include "opencl/cl.hpp"

namespace nlp {

using std::cout;
using std::endl;
using std::get;
using std::vector;

TEST(Embeddings, Run) {
  auto opencl = SetupOpenCl();
  auto context = get<0>(opencl);
  auto devices = get<1>(opencl);
  auto command_queue = get<2>(opencl);

  auto embeddings = Embeddings(context, devices, command_queue);
  auto embeddings_gradient = EmbeddingsGradient(context, devices, command_queue);

  auto w = Tensor<>{{4, 3}, {3, 1}, {
    1, 2, 3,
    2, 3, 4,
    3, 4, 5,
    4, 5, 6,
  }},
  y = Tensor<>({{4, 9}, {9, 1}, vector<float>(4 * 9)});

  auto x = Tensor<int>{{9}, {1}, {
    0, 0, 0, 1, 1, 1, 2, 2, 2,
  }};

  w.Allocate(context);
  x.Allocate(context);
  y.Allocate(context);

  embeddings(w, x, y);

  cout << "w = " << w << endl
      << "x = " << x << endl
      << "Embeddings(w, x) = " << y.Read(command_queue) << endl << endl;

  auto dy = Tensor<>{{4, 9}, {9, 1}, {
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1,
  }}, dw = Tensor<>{{4, 3}, {3, 1}, vector<float>(4 * 3)},
  dx = Tensor<>{{3, 2}, {2, 1}, vector<float>(3 * 2)};

  dy.Allocate(context);
  dw.Allocate(context);
  dx.Allocate(context);

  embeddings_gradient(dy, x, dw);

  cout << "dy = " << dy << endl
      << "EmbeddingsGradient(dy, x) = " << dw.Read(command_queue) << endl << endl;
}

}  // namespace nlp

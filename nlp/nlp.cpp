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

  {
    auto broadcast_add = BroadcastAdd(context, devices, command_queue);
    auto broadcast_add_x_gradient = BroadcastAddXGradient(context, devices, command_queue);
    auto broadcast_add_b_gradient = BroadcastAddBGradient(context, devices, command_queue);

    auto x = Tensor<>{{4, 3}, {3, 1}, {
      1, 2, 3,
      2, 3, 4,
      3, 4, 5,
      4, 5, 6,
    }},
    b = Tensor<>{{4}, {1}, {
      1,
      2,
      3,
      4,
    }},
    y = Tensor<>{{4, 3}, {3, 1}, vector<float>(4 * 3)};

    x.Allocate(context);
    b.Allocate(context);
    y.Allocate(context);

    broadcast_add(x, b, y);

    cout << "x = " << x << endl
        << "b = " << b << endl
        << "BroadcastAdd(x, b) = " << y.Read(command_queue) << endl << endl;

    auto dy = Tensor<>{{4, 3}, {3, 1}, {
      1, 1, 1,
      1, 1, 1,
      1, 1, 1,
      1, 1, 1,
    }},
    dx = Tensor<>{{4, 3}, {3, 1}, vector<float>(4 * 3)},
    db = Tensor<>{{4}, {1}, vector<float>(4)};

    dy.Allocate(context);
    dx.Allocate(context);
    db.Allocate(context);

    broadcast_add_x_gradient(dy, dx);
    broadcast_add_b_gradient(dy, db);

    cout << "dy = " << dy << endl
        << "BroadcastAddXGradient(dy) = " << dx.Read(command_queue) << endl
        << "BroadcastAddBGradient(dy) = " << db.Read(command_queue) << endl << endl;
  }

  {
    auto logistic = Logistic(context, devices, command_queue);
    auto logistic_gradient = LogisticGradient(context, devices, command_queue);

    auto x = Tensor<>{{4, 3}, {3, 1}, {
      1, 2, 3,
      2, 3, 4,
      3, 4, 5,
      4, 5, 6,
    }},
    y = Tensor<>{{4, 3}, {3, 1}, vector<float>(4 * 3)};

    x.Allocate(context);
    y.Allocate(context);

    logistic(x, y);

    cout << "x = " << x << endl
        << "Logistic(x) = " << y.Read(command_queue) << endl << endl;

    auto dy = Tensor<>{{4, 3}, {3, 1}, {
      1, 1, 1,
      1, 1, 1,
      1, 1, 1,
      1, 1, 1,
    }},
    dx = Tensor<>{{4, 3}, {3, 1}, vector<float>(4 * 3)};

    dy.Allocate(context);
    dx.Allocate(context);

    logistic_gradient(dy, y, dx);

    cout << "dy = " << dy << endl
        << "LogisticGradient(dy, y) = " << dx.Read(command_queue) << endl << endl;
  }

  {
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

  return 0;
}

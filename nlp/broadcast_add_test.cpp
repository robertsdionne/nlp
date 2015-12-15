#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "nlp/opencl.hpp"
#include "nlp/broadcast_add.hpp"
#include "nlp/broadcast_add_gradient.hpp"
#include "nlp/tensor.hpp"
#include "opencl/cl.hpp"

namespace nlp {

using std::cout;
using std::endl;
using std::get;
using std::vector;

TEST(BroadcastAdd, Run) {
  auto opencl = SetupOpenCl();
  auto context = get<0>(opencl);
  auto devices = get<1>(opencl);
  auto command_queue = get<2>(opencl);

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

}  // namespace nlp

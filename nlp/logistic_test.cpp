#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "nlp/opencl.hpp"
#include "nlp/logistic.hpp"
#include "nlp/logistic_gradient.hpp"
#include "nlp/tensor.hpp"
#include "opencl/cl.hpp"

namespace nlp {

using std::cout;
using std::endl;
using std::get;
using std::vector;

TEST(Logistic, Run) {
  auto opencl = SetupOpenCl();
  auto context = get<0>(opencl);
  auto devices = get<1>(opencl);
  auto command_queue = get<2>(opencl);

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

}  // namespace nlp

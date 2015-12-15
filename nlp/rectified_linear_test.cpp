#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "nlp/opencl.hpp"
#include "nlp/rectified_linear.hpp"
#include "nlp/rectified_linear_gradient.hpp"
#include "nlp/tensor.hpp"
#include "opencl/cl.hpp"

namespace nlp {

using std::cout;
using std::endl;
using std::get;
using std::vector;

TEST(RectifiedLinear, Run) {
  auto opencl = SetupOpenCl();
  auto context = get<0>(opencl);
  auto devices = get<1>(opencl);
  auto command_queue = get<2>(opencl);

  auto rectified_linear = RectifiedLinear(context, devices, command_queue);
  auto rectified_linear_gradient = RectifiedLinearGradient(context, devices, command_queue);

  auto x = Tensor<>{{5, 5}, {5, 1}, {
    1, -2, 3, -4, 5,
    -2, 3, -4, 5, -6,
    3, -4, 5, -6, 7,
    -4, 5, -6, 7, -8,
    5, -6, 7, -8, 9,
  }},
  y = Tensor<>{{5, 5}, {5, 1}, vector<float>(5 * 5)};

  x.Allocate(context);
  y.Allocate(context);

  rectified_linear(x, y);

  cout << "x = " << x << endl
      << "RectifiedLinear(x) = " << y.Read(command_queue) << endl << endl;

  auto dy = Tensor<>{{5, 5}, {5, 1}, {
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
  }},
  dx = Tensor<>{{5, 5}, {5, 1}, vector<float>(5 * 5)};

  dy.Allocate(context);
  dx.Allocate(context);

  rectified_linear_gradient(dy, x, dx);

  cout << "dy = " << dy << endl
     << "RectifiedLinearGradient(dy, x) = " << dx.Read(command_queue) << endl << endl;
}

}  // namespace nlp

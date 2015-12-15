#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "nlp/opencl.hpp"
#include "nlp/matrix_multiply.hpp"
#include "nlp/matrix_multiply_gradient.hpp"
#include "nlp/tensor.hpp"
#include "opencl/cl.hpp"

namespace nlp {

using std::cout;
using std::endl;
using std::get;
using std::vector;

TEST(MatrixMultiply, Run) {
  auto opencl = SetupOpenCl();
  auto context = get<0>(opencl);
  auto devices = get<1>(opencl);
  auto command_queue = get<2>(opencl);

  auto matrix_multiply = MatrixMultiply(context, devices, command_queue);
  auto matrix_multiply_w_gradient = MatrixMultiplyWGradient(context, devices, command_queue);
  auto matrix_multiply_x_gradient = MatrixMultiplyXGradient(context, devices, command_queue);

  auto w = Tensor<>{{4, 3}, {3, 1}, {
    1, 2, 3,
    2, 3, 4,
    3, 4, 5,
    4, 5, 6,
  }},
  x = Tensor<>{{3, 2}, {2, 1}, {
    1, 2,
    2, 3,
    3, 4,
  }},
  y = Tensor<>({{4, 2}, {2, 1}, vector<float>(4 * 2)});

  w.Allocate(context);
  x.Allocate(context);
  y.Allocate(context);

  matrix_multiply(w, x, y);

  cout << "w = " << w << endl
      << "x = " << x << endl
      << "MatrixMultiply(w, x) = " << y.Read(command_queue) << endl << endl;

  auto dy = Tensor<>{{4, 2}, {2, 1}, {
    1, 1,
    1, 1,
    1, 1,
    1, 1,
  }}, dw = Tensor<>{{4, 3}, {3, 1}, vector<float>(4 * 3)},
  dx = Tensor<>{{3, 2}, {2, 1}, vector<float>(3 * 2)};

  dy.Allocate(context);
  dw.Allocate(context);
  dx.Allocate(context);

  matrix_multiply_w_gradient(dy, x, dw);
  matrix_multiply_x_gradient(dy, w, dx);

  cout << "dy = " << dy << endl
      << "MatrixMultiplyWGradient(dy, x) = " << dw.Read(command_queue) << endl
      << "MatrixMultiplyWGradient(dy, w) = " << dx.Read(command_queue) << endl << endl;
}

}  // namespace nlp

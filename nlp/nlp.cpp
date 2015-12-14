#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "nlp/broadcast_add.hpp"
#include "nlp/broadcast_add_gradient.hpp"
#include "nlp/matrix_multiply.hpp"
#include "nlp/matrix_multiply_gradient.hpp"
#include "nlp/rectified_linear.hpp"
#include "nlp/rectified_linear_gradient.hpp"
#include "nlp/tensor.hpp"
#include "opencl/cl.hpp"

using namespace nlp;

int main(int argument_count, char *arguments[]) {
  using cl::CommandQueue;
  using cl::Context;
  using cl::Device;
  using cl::Platform;

  using nlp::BroadcastAdd;
  using nlp::BroadcastAddXGradient;
  using nlp::BroadcastAddBGradient;
  using nlp::MatrixMultiply;
  using nlp::MatrixMultiplyWGradient;
  using nlp::MatrixMultiplyXGradient;
  using nlp::RectifiedLinear;
  using nlp::RectifiedLinearGradient;
  using nlp::Tensor;

  using std::cout;
  using std::endl;
  using std::find_if;
  using std::string;
  using std::vector;

  vector<Platform> platforms;
  Platform::get(&platforms);
  assert(platforms.size() >= 1);

  auto platform = platforms.at(0);
  vector<Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

  auto nvidia_gpu = find_if(devices.begin(), devices.end(), [] (Device &device) {
    return string("NVIDIA\0", 7) == device.getInfo<CL_DEVICE_VENDOR>();
  });
  assert(nvidia_gpu != devices.end());
  devices = {*nvidia_gpu};

  cl_context_properties context_properties[] = {
    CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform()), 0,
  };
  auto context = Context(devices, context_properties);
  auto command_queue = CommandQueue(context, *nvidia_gpu);

  {
    auto rectified_linear = RectifiedLinear(context, devices, command_queue);
    auto rectified_linear_gradient = RectifiedLinearGradient(context, devices, command_queue);

    auto x = Tensor{{5, 5}, {5, 1}, {
      1, -2, 3, -4, 5,
      -2, 3, -4, 5, -6,
      3, -4, 5, -6, 7,
      -4, 5, -6, 7, -8,
      5, -6, 7, -8, 9,
    }},
    y = Tensor{{5, 5}, {5, 1}, vector<float>(5 * 5)};

    x.Allocate(context);
    y.Allocate(context);

    rectified_linear(x, y);

    cout << "x = " << x << endl
        << "RectifiedLinear(x) = " << y.Read(command_queue) << endl << endl;

    auto dy = Tensor{{5, 5}, {5, 1}, {
      1, 1, 1, 1, 1,
      1, 1, 1, 1, 1,
      1, 1, 1, 1, 1,
      1, 1, 1, 1, 1,
      1, 1, 1, 1, 1,
    }},
    dx = Tensor{{5, 5}, {5, 1}, vector<float>(5 * 5)};

    dy.Allocate(context);
    dx.Allocate(context);

    rectified_linear_gradient(dy, x, dx);

    cout << "dy = " << dy << endl
       << "RectifiedLinearGradient(dy, x) = " << dx.Read(command_queue) << endl << endl;
  }

  {
    auto matrix_multiply = MatrixMultiply(context, devices, command_queue);
    auto matrix_multiply_w_gradient = MatrixMultiplyWGradient(context, devices, command_queue);
    auto matrix_multiply_x_gradient = MatrixMultiplyXGradient(context, devices, command_queue);

    auto w = Tensor{{4, 3}, {3, 1}, {
      1, 2, 3,
      2, 3, 4,
      3, 4, 5,
      4, 5, 6,
    }},
    x = Tensor{{3, 2}, {2, 1}, {
      1, 2,
      2, 3,
      3, 4,
    }},
    y = Tensor({{4, 2}, {2, 1}, vector<float>(4 * 2)});

    w.Allocate(context);
    x.Allocate(context);
    y.Allocate(context);

    matrix_multiply(w, x, y);

    cout << "w = " << w << endl
        << "x = " << x << endl
        << "MatrixMultiply(w, x) = " << y.Read(command_queue) << endl << endl;

    auto dy = Tensor{{4, 2}, {2, 1}, {
      1, 1,
      1, 1,
      1, 1,
      1, 1,
    }}, dw = Tensor{{4, 3}, {3, 1}, vector<float>(4 * 3)},
    dx = Tensor{{3, 2}, {2, 1}, vector<float>(3 * 2)};

    dy.Allocate(context);
    dw.Allocate(context);
    dx.Allocate(context);

    matrix_multiply_w_gradient(dy, x, dw);
    matrix_multiply_x_gradient(dy, w, dx);

    cout << "dy = " << dy << endl
        << "MatrixMultiplyWGradient(dy, x) = " << dw.Read(command_queue) << endl
        << "MatrixMultiplyWGradient(dy, w) = " << dx.Read(command_queue) << endl << endl;
  }

  {
    auto broadcast_add = BroadcastAdd(context, devices, command_queue);
    auto broadcast_add_x_gradient = BroadcastAddXGradient(context, devices, command_queue);
    auto broadcast_add_b_gradient = BroadcastAddBGradient(context, devices, command_queue);

    auto x = Tensor{{4, 3}, {3, 1}, {
      1, 2, 3,
      2, 3, 4,
      3, 4, 5,
      4, 5, 6,
    }},
    b = Tensor{{4}, {1}, {
      1,
      2,
      3,
      4,
    }},
    y = Tensor{{4, 3}, {3, 1}, vector<float>(4 * 3)};

    x.Allocate(context);
    b.Allocate(context);
    y.Allocate(context);

    broadcast_add(x, b, y);

    cout << "x = " << x << endl
        << "b = " << b << endl
        << "BroadcastAdd(x, b) = " << y.Read(command_queue) << endl << endl;

    auto dy = Tensor{{4, 3}, {3, 1}, {
      1, 1, 1,
      1, 1, 1,
      1, 1, 1,
      1, 1, 1,
    }},
    dx = Tensor{{4, 3}, {3, 1}, vector<float>(4 * 3)},
    db = Tensor{{4}, {1}, vector<float>(4)};

    dy.Allocate(context);
    dx.Allocate(context);
    db.Allocate(context);

    broadcast_add_x_gradient(dy, dx);
    broadcast_add_b_gradient(dy, db);

    cout << "dy = " << dy << endl
        << "BroadcastAddXGradient(dy) = " << dx.Read(command_queue) << endl
        << "BroadcastAddBGradient(dy) = " << db.Read(command_queue) << endl << endl;
  }

  return 0;
}

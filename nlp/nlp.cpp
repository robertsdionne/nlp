#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "nlp/rectified_linear.hpp"
#include "nlp/rectified_linear_gradient.hpp"
#include "opencl/cl.hpp"

int main(int argument_count, char *arguments[]) {
  using cl::CommandQueue;
  using cl::Context;
  using cl::Device;
  using cl::Platform;

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

  auto rectified_linear = RectifiedLinear(context, devices, command_queue);
  auto rectified_linear_gradient = RectifiedLinearGradient(context, devices, command_queue);

  auto x = Tensor{{5, 5}, {5, 1}, {
    1, -2, 3, -4, 5,
    -2, 3, -4, 5, -6,
    3, -4, 5, -6, 7,
    -4, 5, -6, 7, -8,
    5, -6, 7, -8, 9,
  }}, y = Tensor{{5, 5}, {5, 1}, vector<float>(5 * 5)};

  x.Allocate(context);
  y.Allocate(context);

  rectified_linear(x, y);

  cout << y.Read(command_queue) << endl;

  auto dy = Tensor{{5, 5}, {5, 1}, {
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
  }}, dx = Tensor{{5, 5}, {5, 1}, vector<float>(5 * 5)};

  dy.Allocate(context);
  dx.Allocate(context);

  rectified_linear_gradient(dy, x, dx);

  cout << dx.Read(command_queue) << endl;

  return 0;
}

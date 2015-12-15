#ifndef NLP_OPENCL_HPP_
#define NLP_OPENCL_HPP_

#include <string>
#include <tuple>
#include <vector>

#include "opencl/cl.hpp"

namespace nlp {

using cl::CommandQueue;
using cl::Context;
using cl::Device;
using cl::Platform;

using std::string;
using std::tuple;
using std::vector;

tuple<Context, vector<Device>, CommandQueue> SetupOpenCl() {
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

  return make_tuple(context, devices, command_queue);
}


}  // namespace nlp

#endif  // NLP_OPENCL_HPP_

#include <torch/extension.h>

torch::Tensor add_one(torch::Tensor x) {
  return x + 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_one", &add_one, "Add one to a tensor");
}


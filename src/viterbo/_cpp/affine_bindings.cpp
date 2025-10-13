#include <torch/extension.h>

#include "affine_ops.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("affine_scale_shift", &affine_scale_shift,
        "Elementwise affine transform applying scale then shift.",
        py::arg("x"), py::arg("scale"), py::arg("shift"));
}

#pragma once

#include <torch/extension.h>

torch::Tensor affine_scale_shift(torch::Tensor x, double scale, double shift);

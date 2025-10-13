#include "affine_ops.h"

torch::Tensor affine_scale_shift(torch::Tensor x, double scale, double shift) {
  torch::Tensor out = x.mul(scale);
  out.add_(shift);
  return out;
}

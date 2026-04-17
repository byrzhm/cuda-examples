#include <torch/extension.h>

void launch_square (torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE (TORCH_EXTENSION_NAME, m) {
    m.def ("square", &launch_square, "Square elements (CUDA)");
}

// Acknowledgments: https://github.com/tarashakhurana/4d-occ-forecasting
// Modified by Haisong Liu

#include <string>
#include <torch/extension.h>
#include <vector>

/*
 * CUDA forward declarations
 */

std::vector<torch::Tensor> render_forward_cpu(torch::Tensor sigma,
                                               torch::Tensor origin,
                                               torch::Tensor points,
                                               torch::Tensor tindex,
                                               const std::vector<int> grid,
                                               std::string phase_name);

std::vector<torch::Tensor>
render_cpu(torch::Tensor sigma, torch::Tensor origin, torch::Tensor points,
            torch::Tensor tindex, std::string loss_name);

torch::Tensor init_cpu(torch::Tensor points, torch::Tensor tindex,
                        const std::vector<int> grid);

/*
 * C++ interface
 */

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor>
render_forward(torch::Tensor sigma, torch::Tensor origin, torch::Tensor points,
               torch::Tensor tindex, const std::vector<int> grid,
               std::string phase_name) {
  CHECK_CONTIGUOUS(sigma);
  CHECK_CONTIGUOUS(origin);
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(tindex);
  return render_forward_cpu(sigma, origin, points, tindex, grid, phase_name);
}


std::vector<torch::Tensor> render(torch::Tensor sigma, torch::Tensor origin,
                                  torch::Tensor points, torch::Tensor tindex,
                                  std::string loss_name) {
  CHECK_CONTIGUOUS(sigma);
  CHECK_CONTIGUOUS(origin);
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(tindex);
  return render_cpu(sigma, origin, points, tindex, loss_name);
}

torch::Tensor init(torch::Tensor points, torch::Tensor tindex,
                   const std::vector<int> grid) {
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(tindex);
  return init_cpu(points, tindex, grid);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &init, "Initialize");
  m.def("render", &render, "Render");
  m.def("render_forward", &render_forward, "Render (forward pass only)");
}

#include <torch/extension.h>
#include <ATen/ATen.h>
using namespace at;

#include <vector>

#define WITHIN_BOUNDS(x, y, H, W) (x >= 0 && x < W && y >= 0 && y < H)

template <typename scalar_t>
static void convolve_patch(
    TensorAccessor<scalar_t,3> input,
    TensorAccessor<scalar_t,3> weight,
    scalar_t *dst,
    int kH, int kW,
    int u, int v,
    int shiftU, int shiftV){
  const int C = input1.size(0);
  const int iH = input1.size(1);
  const int iW = input1.size(2);
  for (int c=0; c<C; ++c){
    for (int i=0; i<kH; ++i){
      int i1 = u + i;
      int i2 = i1 + shiftU;
      if WITHIN_BOUNDS(i1, i2, iH, iH){
        for (int j=0; j<kW; ++j){
          int j1 = v + j;
          int j2 = j1 + shiftV;
          if WITHIN_BOUNDS(j1, j2, iW, iW){
            scalar_t v1 = input1[c][i1][j1];
            scalar_t v2 = input2[c][i2][j2];
            *dst += v1 * v2;
          }
        }
      }
    }
  }
}


torch::Tensor max_convolution2d_cpp_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int kH, int kW,
    int padH, int padW) {

  const auto batch_size = input.size(0);
  const auto C = input.size(1);
  const auto iH = input.size(2);
  const auto iW = input.size(3);

  const auto oH = (iH + 2 * padH) / kH;
  const auto oW = (iW + 2 * padW) / kW;
  auto output = at::zeros({batch_size, C, oH, oW}, input.options());

  int n, c, h, w;
  #pragma omp parallel for private(n, c, h, w) collapse(2)
    for (n = 0; n < batch_size; ++n) {
      for(c = 0; c < C; ++c){
          AT_DISPATCH_FLOATING_TYPES(input.type(), "max_convolution2d_forward_cpp", ([&] {
            auto input_acc = input.accessor<scalar_t, 4>();
            auto weight_acc = weight.accessor<scalar_t, 4>();
            auto output_acc = output.accessor<scalar_t, 5>();
            for (h = 0; h < oH; ++h) {
              for (w = 0; w < oW; ++w) {
                convolve_patch(input_acc[n],
                               weight_acc,
                                &output_acc[n][c][h][w],
                                kH, kW,
                                -padH + h * dH,
                                -padW + w * dW,
                                (ph - patchRadH)  * dilation_patchH,
                                (pw - patchRadW)  * dilation_patchW);
              }
            }
          }));
      }
    }
  return output;
}

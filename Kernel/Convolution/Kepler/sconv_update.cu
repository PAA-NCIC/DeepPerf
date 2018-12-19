#include "sconv.h"

bool update(const float *I, float *F, const float *O,
  unsigned int N, unsigned int C, unsigned int K,
  unsigned int D, unsigned int H, unsigned int W,
  unsigned int R, unsigned int S, unsigned int T,
  unsigned int M, unsigned int P, unsigned int Q,
  unsigned int str_d, unsigned int str_h, unsigned int str_w,
  unsigned int pad_d, unsigned int pad_h, unsigned int pad_w) {
  float alpha = 1.0f;
  unsigned int DHW, WN, HW, HWN, DHWN, CRST, RST, RS;
  unsigned int PQ, QN, PQN, MPQN;
  unsigned int magic_HW, magic_W;
  unsigned int shift_HW, shift_W;
  unsigned int magic_RST, magic_RS, magic_S;
  unsigned int shift_RST, shift_RS, shift_S;
  unsigned int magic_PQu, shift_PQu;
  unsigned int magic_Qu, shift_Qu;
  unsigned int grid_P = 1;
  unsigned int grid_Q = 1;
  unsigned int grid_PQ = grid_P * grid_Q;
  unsigned int grid_PQM = grid_PQ * M;
  // input
  WN = W * N;
  HW = H * W;
  HWN = H * WN;
  DHW = D * HW;
  DHWN = D * HWN;
  // filter
  RS = R * S;
  RST = T * RS;
  CRST = C * RS;
  // output
  QN = Q * N;
  PQN = P * QN;
  MPQN = M * PQN;
  // magic numbers
  magic32(CRST, RST, magic_RST, shift_RST);
  magic32(RST + 32, RS, magic_RS, shift_RS);
  magic32(RS + 32, S, magic_S, shift_S);
  magic32(DHW, HW, magic_HW, shift_HW);
  magic32(HW, W, magic_W, shift_W);
  magic32(grid_PQM, grid_PQ, magic_PQu, shift_PQu);
  magic32(grid_PQ, grid_Q, magic_Qu, shift_Qu);
  std::cout << "CRST: " << CRST << std::endl;
  // test param set up
  float *test_param;
  cudaError_t cuda_error;
  cuda_error = cudaMalloc((void**)&test_param, sizeof(float) * 1024);
  cudaMemset(test_param, 0, sizeof(float) * 1024);
  void *args[43] = {&test_param, &F, &I, &O, &alpha,
    &N, &K, &D, &H, &W, &WN, &HWN, &DHWN,
    &C, &CRST,
    &RST, &magic_RST, &shift_RST,
    &RS, &magic_RS, &shift_RS,
    &S, &magic_S, &shift_S,
    &pad_d, &pad_h, &pad_w,
    &str_d, &str_h, &str_w,
    &P, &Q, &PQ, &QN, &PQN, &MPQN,
    &magic_Qu, &shift_Qu,
    &magic_PQu, &shift_PQu,
    &grid_P, &grid_Q, &grid_PQ};
  int gridX = grid_PQM;
  int gridY = CRST / 128 + (CRST % 128 != 0);
  int gridZ = K / 128 + (K % 128 != 0);
  std::string kernel_name = "sconv_update_C128_K128";
  CUresult res = cuLaunchKernel(nervana_kernels[kernel_name], gridX, gridY, gridZ, 256, 1, 1,
    0, 0, args, NULL);
  if (res != CUDA_SUCCESS) {
    std::cerr << "Line " << __LINE__ << " error launching kernel " << kernel_name << " " << res << std::endl;
    return false;
  }
  cuCtxSynchronize();
  float* h_test = (float *)malloc(sizeof(float) * 256);
  cuda_error = cudaMemcpy(h_test, test_param, sizeof(float) * 256, cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess) {
    std::cerr << "Line " << __LINE__ << " memcpy error: " << cuda_error << std::endl;
    exit(1);
  }
  for (int i = 0; i < 256; ++i) {
    std::cout << h_test[i] << " ";
  }
  std::cout << std::endl;
  // free test_param
  free(h_test);
  return true;
}

int main() {
  cudaFree(0);
  float *d_I, *d_F, *d_O;
  unsigned int N = 128, C = 3, K = 128, D = 1, H = 224, W = 224, T = 1, R = 11, S = 11;
  unsigned int str_d = 1, str_h = 4, str_w = 4;
  unsigned int pad_d = 0, pad_h = 3, pad_w = 3;
  unsigned int M, P, Q;
  cudaError_t cuda_error;
  M = (D - T + 2 * pad_d) / str_d + 1;
  P = (H - R + 2 * pad_h) / str_h + 1;
  Q = (W - S + 2 * pad_w) / str_w + 1;
  float *h_O = (float *)malloc(K * M * P * Q * N * sizeof(float));
  for (int i = 0; i < K * M * P * Q * N; ++i) {
    h_O[i] = 1;
  }
  float *h_I = (float *)malloc(C * D * H * W * N * sizeof(float));
  for (int i = 0; i < C * D * H * W * N; ++i) {
    h_I[i] = 1;
  }
  float* h_F = (float *)malloc(sizeof(float) * C * R * S * T * K);
  // device memory
  cudaMalloc((void**)&d_I, sizeof(float) * C * D * H * W * N);
  cudaMalloc((void**)&d_F, sizeof(float) * C * R * S * T * K);
  cudaMalloc((void**)&d_O, sizeof(float) * K * M * P * Q * N);
  // memcpy h_I, h_O
  cudaMemcpy(d_I, h_I, sizeof(float) * C * D * H * W * N,
    cudaMemcpyHostToDevice);
  cudaMemcpy(d_O, h_O, sizeof(float) * K * M * P * Q * N,
    cudaMemcpyHostToDevice);
  // load kernels 
  if (!load_kernels("./")) {
    std::cerr << "Couldn't load all kernels" << std::endl;
    exit(1);
  }
  // launch kernel
  if (!update(d_I, d_F, d_O, N, C, K, D, H, W, R, S, T, M, P, Q, str_d, str_h, str_w, pad_d, pad_h, pad_w)) {
    std::cerr << "Launch error" << std::endl;
    exit(1);
  }
  // output
  std::cout << "result" << std::endl;
  cuda_error = cudaMemcpy(h_F, d_F, sizeof(float) * C * R * S * T * K, cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess) {
    std::cerr << "Line " << __LINE__ << " memcpy error: " << cuda_error << std::endl;
    exit(1);
  }
  for (int i = 0; i < 128; ++i) {
    std::cout << h_F[i] << " ";
  }
  std::cout << std::endl;
  // free memory
  free(h_O);
  free(h_I);
  free(h_F);
  cudaFree(d_I);
  cudaFree(d_F);
  cudaFree(d_O);
  // run successfully
  std::cout << "finish" << std::endl;
  return 0;
}

#include "sconv.h"

bool bprop_C128_N128(float *I, const float *F, const float *O,
  unsigned int N, unsigned int C, unsigned int K,
  unsigned int D, unsigned int H, unsigned int W,
  unsigned int R, unsigned int S, unsigned int T,
  unsigned int M, unsigned int P, unsigned int Q,
  unsigned int str_d, unsigned int str_h, unsigned int str_w,
  unsigned int pad_d, unsigned int pad_h, unsigned int pad_w) {
  float alpha = 1.0f;
  unsigned int DHW, WN, HW, HWN, DHWN, CRST, RST, RS;
  unsigned int MPQ, PQ, QN, PQN, MPQN;
  unsigned int magic_HW, magic_W;
  unsigned int shift_HW, shift_W;
  unsigned int magic_RST, magic_RS, magic_S;
  unsigned int shift_RST, shift_RS, shift_S;
  unsigned int magic_Q, shift_Q, magic_PQ, shift_PQ;
  unsigned int magic_str_w, magic_str_h, magic_str_d;
  unsigned int shift_str_w, shift_str_h, shift_str_d;
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
  PQ = P * Q;
  MPQ = M * P * Q;
  // magic numbers
  magic32(MPQ, PQ, magic_PQ, shift_PQ);
  magic32(PQ, Q, magic_Q, shift_Q);
  magic32(CRST, RST, magic_RST, shift_RST);
  magic32(RST + 32, RS, magic_RS, shift_RS);
  magic32(RS + 32, S, magic_S, shift_S);
  magic32(W + S - pad_w - 2, str_w, magic_str_w, shift_str_w);
  magic32(H + R - pad_h - 2, str_h, magic_str_h, shift_str_h);
  magic32(D + T - pad_d - 2, str_d, magic_str_d, shift_str_d);
  magic32(DHW, HW, magic_HW, shift_HW);
  magic32(HW, W, magic_W, shift_W);
  // test param set up
  float *test_param;
  cudaError_t cuda_error;
  cuda_error = cudaMalloc((void**)&test_param, sizeof(float) * 1024);
  cudaMemset(test_param, 0, sizeof(float) * 1024);
  void *args[45] = {
    &test_param, &I, &O, &F, &alpha,
    &N, &C, &M, &P, &Q, &QN, &PQN, &MPQN,
    &K, &CRST, &RST,
    &RS, &magic_RS, &shift_RS,
    &S, &magic_S, &shift_S,
    &pad_d, &pad_h, &pad_w,
    &str_d, &str_h, &str_w,
    &W, &HW, &WN, &HWN, &DHWN,
    &magic_W, &shift_W,
    &magic_HW, &shift_HW,
    &R, &T,
    &magic_str_w, &shift_str_w,
    &magic_str_h, &shift_str_h,
    &magic_str_d, &shift_str_d};
  int gridDWH = D * W * H;
  int gridX = gridDWH;
  int gridY = C / 128 + (C % 128 != 0);
  int gridZ = N / 128 + (N % 128 != 0);
  std::string kernel_name = "sconv_bprop_C128_N128";
  CUresult res = cuLaunchKernel(nervana_kernels[kernel_name], gridX, gridY, gridZ, 256, 1, 1,
    128 * 8 * 4 + RST * 4 * 2 + 8, 0, args, NULL);
  if (res != CUDA_SUCCESS) {
    std::cerr << "Line " << __LINE__ << " error launching kernel " << kernel_name << " " << res << std::endl;
    return false;
  }
  cuCtxSynchronize();
  float* h_test = (float *)malloc(sizeof(float) * 128);
  for (int i = 0; i < 128; ++i) {
    std::cout << h_test[i] << " ";
  }
  std::cout << std::endl;
  cuda_error = cudaMemcpy(h_test, test_param, sizeof(float) * 128, cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess) {
    std::cerr << "Line " << __LINE__ << " memcpy error: " << cuda_error << std::endl;
    exit(1);
  }
  for (int i = 0; i < 128; ++i) {
    std::cout << h_test[i] << " ";
  }
  std::cout << std::endl;
  // free test_param
  free(h_test);
  return true;
}

bool bprop_C64_N64(float *I, const float *F, const float *O,
  unsigned int N, unsigned int C, unsigned int K,
  unsigned int D, unsigned int H, unsigned int W,
  unsigned int R, unsigned int S, unsigned int T,
  unsigned int M, unsigned int P, unsigned int Q,
  unsigned int str_d, unsigned int str_h, unsigned int str_w,
  unsigned int pad_d, unsigned int pad_h, unsigned int pad_w) {
  float alpha = 1.0f;
  unsigned int DHW, WN, HW, HWN, DHWN, CRST, RST, RS;
  unsigned int MPQ, PQ, QN, PQN, MPQN;
  unsigned int magic_HW, magic_W;
  unsigned int shift_HW, shift_W;
  unsigned int magic_RST, magic_RS, magic_S;
  unsigned int shift_RST, shift_RS, shift_S;
  unsigned int magic_Q, shift_Q, magic_PQ, shift_PQ;
  unsigned int magic_str_w, magic_str_h, magic_str_d;
  unsigned int shift_str_w, shift_str_h, shift_str_d;
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
  PQ = P * Q;
  MPQ = M * P * Q;
  // magic numbers
  magic32(MPQ, PQ, magic_PQ, shift_PQ);
  magic32(PQ, Q, magic_Q, shift_Q);
  magic32(CRST, RST, magic_RST, shift_RST);
  magic32(RST + 32, RS, magic_RS, shift_RS);
  magic32(RS + 32, S, magic_S, shift_S);
  magic32(W + S - pad_w - 2, str_w, magic_str_w, shift_str_w);
  magic32(H + R - pad_h - 2, str_h, magic_str_h, shift_str_h);
  magic32(D + T - pad_d - 2, str_d, magic_str_d, shift_str_d);
  magic32(DHW, HW, magic_HW, shift_HW);
  magic32(HW, W, magic_W, shift_W);
  // test param set up
  float *test_param;
  cudaError_t cuda_error;
  cuda_error = cudaMalloc((void**)&test_param, sizeof(float) * 1024);
  cudaMemset(test_param, 0, sizeof(float) * 1024);
  void *args[45] = {
    &test_param, &I, &O, &F, &alpha,
    &N, &C, &M, &P, &Q, &QN, &PQN, &MPQN,
    &K, &CRST, &RST,
    &RS, &magic_RS, &shift_RS,
    &S, &magic_S, &shift_S,
    &pad_d, &pad_h, &pad_w,
    &str_d, &str_h, &str_w,
    &W, &HW, &WN, &HWN, &DHWN,
    &magic_W, &shift_W,
    &magic_HW, &shift_HW,
    &R, &T,
    &magic_str_w, &shift_str_w,
    &magic_str_h, &shift_str_h,
    &magic_str_d, &shift_str_d};
  int gridDWH = D * W * H;
  int gridX = gridDWH;
  int gridY = C / 64 + (C % 64 != 0);
  int gridZ = N / 64 + (N % 64 != 0);
  std::string kernel_name = "sconv_bprop_C64_N64";
  CUresult res = cuLaunchKernel(nervana_kernels[kernel_name], gridX, gridY, gridZ, 64, 1, 1,
    0, 0, args, NULL);
  if (res != CUDA_SUCCESS) {
    std::cerr << "Line " << __LINE__ << " error launching kernel " << kernel_name << " " << res << std::endl;
    return false;
  }
  cuCtxSynchronize();
  float* h_test = (float *)malloc(sizeof(float) * 64);
  for (int i = 0; i < 64; ++i) {
    std::cout << h_test[i] << " ";
  }
  std::cout << std::endl;
  cuda_error = cudaMemcpy(h_test, test_param, sizeof(float) * 64, cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess) {
    std::cerr << "Line " << __LINE__ << " memcpy error: " << cuda_error << std::endl;
    exit(1);
  }
  for (int i = 0; i < 64; ++i) {
    std::cout << h_test[i] << " ";
  }
  std::cout << std::endl;
  // free test_param
  free(h_test);
  return true;
}

bool bprop_C1_N64(float *I, const float *F, const float *O,
  unsigned int N, unsigned int C, unsigned int K,
  unsigned int D, unsigned int H, unsigned int W,
  unsigned int R, unsigned int S, unsigned int T,
  unsigned int M, unsigned int P, unsigned int Q,
  unsigned int str_d, unsigned int str_h, unsigned int str_w,
  unsigned int pad_d, unsigned int pad_h, unsigned int pad_w) {
  float alpha = 1.0f;
  unsigned int WN, HWN, DHWN, CRST, RST, RS;
  unsigned int MPQ, PQ, QN, PQN, MPQN;
  unsigned int magic_RST, magic_RS, magic_S;
  unsigned int shift_RST, shift_RS, shift_S;
  unsigned int magic_Q, shift_Q, magic_PQ, shift_PQ;
  unsigned int magic_str_w, magic_str_h, magic_str_d;
  unsigned int shift_str_w, shift_str_h, shift_str_d;
  unsigned int CRST32, MPQN32;
  // input
  WN = W * N;
  HWN = H * WN;
  DHWN = D * HWN;
  // filter
  RS = R * S;
  RST = T * RS;
  CRST = C * RS;
  // output
  QN = Q * N;
  PQN = P * QN;
  MPQN = M * PQN;
  PQ = P * Q;
  MPQ = M * PQ;
  // special
  CRST32 = 32 * CRST;
  MPQN32 = 32 * MPQN;
  // magic numbers
  magic32(MPQ, PQ, magic_PQ, shift_PQ);
  magic32(PQ, Q, magic_Q, shift_Q);
  magic32(CRST, RST, magic_RST, shift_RST);
  magic32(RST + 32, RS, magic_RS, shift_RS);
  magic32(RS + 32, S, magic_S, shift_S);
  magic32(W + S - pad_w - 2, str_w, magic_str_w, shift_str_w);
  magic32(H + R - pad_h - 2, str_h, magic_str_h, shift_str_h);
  magic32(D + T - pad_d - 2, str_d, magic_str_d, shift_str_d);
  // test param set up
  float *test_param;
  cudaError_t cuda_error;
  cuda_error = cudaMalloc((void**)&test_param, sizeof(float) * 1024);
  cudaMemset(test_param, 0, sizeof(float) * 1024);
  void *args[41] = {
    &test_param, &I, &O, &F, &alpha,
    &N, &K, &D, &H, &W, &WN, &HWN, &DHWN,
    &C, &CRST,
    &RST, &magic_RST, &shift_RST,
    &RS, &magic_RS, &shift_RS,
    &S, &magic_S, &shift_S,
    &pad_d, &pad_h, &pad_w,
    &str_d, &str_h, &str_w,
    &Q, &PQ, &QN, &PQN, &MPQN,
    &magic_Q, &shift_Q,
    &magic_PQ, &shift_PQ,
    &CRST32,
    &MPQN32};
  int gridMPQ = MPQ;
  int gridX = gridMPQ;
  int gridY = CRST / 32 + (CRST % 32 != 0);
  int gridZ = N / 64 + (N % 64 != 0);
  const std::string kernel_name = "sconv_bprop_C1_N64";
  CUresult res = cuLaunchKernel(nervana_kernels[kernel_name], gridX, gridY, gridZ, 32, 1, 1,
    0, 0, args, NULL);
  if (res != CUDA_SUCCESS) {
    std::cerr << "Line " << __LINE__ << " error launching kernel " << kernel_name << " " << res << std::endl;
    return false;
  }
  cuCtxSynchronize();
  float* h_test = (float *)malloc(sizeof(float) * 32);
  cuda_error = cudaMemcpy(h_test, test_param, sizeof(float) * 32, cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess) {
    std::cerr << "Line " << __LINE__ << " memcpy error: " << cuda_error << std::endl;
    exit(1);
  }
  for (int i = 0; i < 32; ++i) {
    std::cout << h_test[i] << " ";
  }
  std::cout << std::endl;
  // free test_param
  free(h_test);
  return true;
}

int main(int argc, char** argv) {
  // init
  cudaFree(0);
  // params
  float *d_I, *d_F, *d_O;
  unsigned int N = 128, C = 192, K = 192, D = 1, H = 13, W = 13, T = 1, R = 12, S = 12;
  unsigned int str_d = 1, str_h = 1, str_w = 1;
  unsigned int pad_d = 0, pad_h = 0, pad_w = 0;
  unsigned int M, P, Q;
  cudaError_t cuda_error;
  // 32, 64, or 128
  if (argc > 1) {
    C = atoi(argv[1]);
  }   
  M = (D - T + 2 * pad_d) / str_d + 1;
  P = (H - R + 2 * pad_h) / str_h + 1;
  Q = (W - S + 2 * pad_w) / str_w + 1;
  float *h_O = (float *)malloc(K * M * P * Q * N * sizeof(float));
  for (int i = 0; i < K * M * P * Q * N; ++i) {
    h_O[i] = 1;
  }
  float *h_F = (float *)malloc(K * R * S * T * C * sizeof(float));
  for (int i = 0; i < K * C * R * S * T; ++i) {
    h_F[i] = 1;
  }
  float* h_I = (float *)malloc(sizeof(float) * C * D * H * W * N);
  // device memory
  cudaMalloc((void**)&d_I, sizeof(float) * C * D * H * W * N);
  cudaMalloc((void**)&d_F, sizeof(float) * K * R * S * T * C * 2);
  cudaMalloc((void**)&d_O, sizeof(float) * K * M * P * Q * N);
  // memcpy h_O, h_F
  cudaMemcpy(d_O, h_O, sizeof(float) * M * P * Q * K * N,
    cudaMemcpyHostToDevice);
  cudaMemcpy(d_F, h_F, sizeof(float) * K * R * S * T * C,
    cudaMemcpyHostToDevice);
  // load kernels 
  if (!load_kernels("./")) {
    std::cerr << "Couldn't load all kernels" << std::endl;
    exit(1);
  }
  if (C % 64 != 0) {
    // launch kernel C1
    if (!bprop_C1_N64(d_I, d_F, d_O, N, C, K, D, H, W, R, S, T, M, P, Q, str_d, str_h, str_w, pad_d, pad_h, pad_w)) {
      std::cerr << "Launch error C1" << std::endl;
      exit(1);
    }
  } else {
    // launch kernel C64
    if (C <= 64) {
      if (!bprop_C64_N64(d_I, d_F, d_O, N, C, K, D, H, W, R, S, T, M, P, Q, str_d, str_h, str_w, pad_d, pad_h, pad_w)) {
        std::cerr << "Launch error C64" << std::endl;
        exit(1);
      }
    } else {
      if (!bprop_C128_N128(d_I, d_F, d_O, N, C, K, D, H, W, R, S, T, M, P, Q, str_d, str_h, str_w, pad_d, pad_h, pad_w)) {
        std::cerr << "Launch error C128" << std::endl;
        exit(1);
      }
    }
  }
  // output
  std::cout << "result" << std::endl;
  cuda_error = cudaMemcpy(h_I, d_I, sizeof(float) * C * D * H * W * N, cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess) {
    std::cerr << "Line " << __LINE__ << " memcpy error: " << cuda_error << std::endl;
    exit(1);
  }
  for (int i = 0; i < 128; ++i) {
    std::cout << h_I[i] << " ";
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

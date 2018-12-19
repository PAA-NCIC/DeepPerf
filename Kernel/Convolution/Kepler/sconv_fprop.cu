#include "sconv.h"

bool fprop_K64_N64(const float *I, const float *F, float *O,
  unsigned int N, unsigned int C, unsigned int K,
  unsigned int D, unsigned int H, unsigned int W,
  unsigned int R, unsigned int S, unsigned int T,
  unsigned int M, unsigned int P, unsigned int Q,
  unsigned int str_d, unsigned int str_h, unsigned int str_w,
  unsigned int pad_d, unsigned int pad_h, unsigned int pad_w) {
	std::string kernel_name = "sconv_fprop_K64_N64";
  float alpha = 1.0f;
  unsigned int WN, HWN, DHWN, KRST, RST, RS, PQ, QN, PQM, PQN, MPQN;
  unsigned int magic_RS, magic_S;
  unsigned int shift_RS, shift_S;
  unsigned int magic_Q, shift_Q, magic_PQ, shift_PQ;
  // input
  WN = W * N;
  HWN = H * WN;
  DHWN = D * HWN;
  // filter
  RS = R * S;
  RST = T * RS;
  KRST = K * RST;
  // output
  QN = Q * N;
  PQ = P * Q;
  PQM = PQ * M;
  PQN = P * QN;
  MPQN = M * PQN;
  // magic numbers
  magic32(PQ, Q, magic_Q, shift_Q);
  magic32(PQM, PQ, magic_PQ, shift_PQ);
  magic32(RST + 32, RS, magic_RS, shift_RS);
  magic32(RS + 32, S, magic_S, shift_S);
  // test param set up
  float *test_param;
  cudaError_t cuda_error;
  cuda_error = cudaMalloc((void**)&test_param, sizeof(float) * 1024);
  cudaMemset(test_param, 0, sizeof(float) * 1024);
  // arguments
  void *args[37] = {&test_param, &O, &I, &F, &alpha,
    &N, &K, &D, &H, &W, &WN, &HWN, &DHWN,
    &C, &KRST, &RST, &RS, &magic_RS, &shift_RS, &S, &magic_S, &shift_S,
    &pad_d, &pad_h, &pad_w, &str_d, &str_h, &str_w,
    &Q, &PQ, &QN, &PQN, &MPQN, &magic_Q, &shift_Q, &magic_PQ, &shift_PQ};
  int gridMPQ = M * P * Q;
  int gridX = gridMPQ;
  int gridY = K / 64 + (K % 64 != 0);
  int gridZ = N / 64 + (N % 64 != 0);
  CUresult res = cuLaunchKernel(nervana_kernels[kernel_name],
    gridX, gridY, gridZ, 64, 1, 1, 64 * 8 * 4 + RST * 4 * 2 + 8, 0, args, NULL);
  if (res != CUDA_SUCCESS) {
    std::cerr << "Line " << __LINE__ << " error launching kernel " << kernel_name << " " << res << std::endl;
    return false;
  }
  cuCtxSynchronize();
  // output test_param
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

bool fprop_K128_N128(const float *I, const float *F, float *O,
  unsigned int N, unsigned int C, unsigned int K,
  unsigned int D, unsigned int H, unsigned int W,
  unsigned int R, unsigned int S, unsigned int T,
  unsigned int M, unsigned int P, unsigned int Q,
  unsigned int str_d, unsigned int str_h, unsigned int str_w,
  unsigned int pad_d, unsigned int pad_h, unsigned int pad_w) {
	std::string kernel_name = "sconv_fprop_K128_N128";
  float alpha = 1.0f;
  unsigned int WN, HWN, DHWN, KRST, RST, RS, PQ, QN, PQM, PQN, MPQN;
  unsigned int magic_RS, magic_S;
  unsigned int shift_RS, shift_S;
  unsigned int magic_Q, shift_Q, magic_PQ, shift_PQ;
  // input
  WN = W * N;
  HWN = H * WN;
  DHWN = D * HWN;
  // filter
  RS = R * S;
  RST = T * RS;
  KRST = K * RST;
  // output
  QN = Q * N;
  PQ = P * Q;
  PQM = PQ * M;
  PQN = P * QN;
  MPQN = M * PQN;
  // magic numbers
  magic32(PQ, Q, magic_Q, shift_Q);
  magic32(PQM, PQ, magic_PQ, shift_PQ);
  magic32(RST + 32, RS, magic_RS, shift_RS);
  magic32(RS + 32, S, magic_S, shift_S);
  // test param set up
  float *test_param;
  cudaError_t cuda_error;
  cuda_error = cudaMalloc((void**)&test_param, sizeof(float) * 1024);
  cudaMemset(test_param, 0, sizeof(float) * 1024);
  // arguments
  void *args[37] = {&test_param, &O, &I, &F, &alpha,
    &N, &K, &D, &H, &W, &WN, &HWN, &DHWN,
    &C, &KRST, &RST, &RS, &magic_RS, &shift_RS, &S, &magic_S, &shift_S,
    &pad_d, &pad_h, &pad_w, &str_d, &str_h, &str_w,
    &Q, &PQ, &QN, &PQN, &MPQN, &magic_Q, &shift_Q, &magic_PQ, &shift_PQ};
  int gridMPQ = M * P * Q;
  int gridX = gridMPQ;
  int gridY = K / 128 + (K % 128 != 0);
  int gridZ = N / 128 + (N % 128 != 0);
  CUresult res = cuLaunchKernel(nervana_kernels[kernel_name],
    gridX, gridY, gridZ, 256, 1, 1, 128 * 8 * 4 + RST * 4 * 2 + 8, 0, args, NULL);
  if (res != CUDA_SUCCESS) {
    std::cerr << "Line " << __LINE__ << " error launching kernel " << kernel_name << " " << res << std::endl;
    return false;
  }
  cuCtxSynchronize();
  // output test_param
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

int main() {
  // init
  cudaFree(0);
  // params
  float *d_I, *d_F, *d_O;
  unsigned int N = 128, C = 1, K = 128, D = 1, H = 5, W = 5, T = 1, R = 5, S = 5;
  unsigned int str_d = 1, str_h = 1, str_w = 1;
  unsigned int pad_d = 0, pad_h = 0, pad_w = 0;
  unsigned int M, P, Q;
  cudaError_t cuda_error;
  M = (D - T + 2 * pad_d) / str_d + 1;
  P = (H - R + 2 * pad_h) / str_h + 1;
  Q = (W - S + 2 * pad_w) / str_w + 1;
  // host memory
  float *h_I = (float *)malloc(C * D * H * W * N * sizeof(float));
  for (int i = 0; i < C * D * H * W; ++i) {
    for (int j = 0; j < N; ++j) {
      h_I[i * N + j] = j;
    }
  }
  float *h_F = (float *)malloc(C * R * S * T * K * sizeof(float));
  for (int i = 0; i < C * R * S * T * K; ++i) {
    h_F[i] = 1;
  }
  float* h_O = (float *)malloc(sizeof(float) * K * M * P * Q * N);
  // device memory
  cudaMalloc((void**)&d_I, sizeof(float) * C * D * H * W * N);
  cudaMalloc((void**)&d_F, sizeof(float) * C * R * S * T * K);
  cudaMalloc((void**)&d_O, sizeof(float) * K * M * P * Q * N);
  // memcpy h_I, h_F
  cudaMemcpy(d_I, h_I, sizeof(float) * C * D * H * W * N,
    cudaMemcpyHostToDevice);
  cudaMemcpy(d_F, h_F, sizeof(float) * C * R * S * T * K,
    cudaMemcpyHostToDevice);
  // load kernels 
  if (!load_kernels("./")) {
    std::cerr << "Couldn't load all kernels" << std::endl;
    exit(1);
  }
  // launch kernel
	if (K <= 64) {
		if (!fprop_K64_N64(d_I, d_F, d_O, N, C, K, D, H, W, R, S, T, M, P, Q, str_d, str_h, str_w, pad_d, pad_h, pad_w)) {
			std::cerr << "Launch error" << std::endl;
		}
	} else {
		if (!fprop_K128_N128(d_I, d_F, d_O, N, C, K, D, H, W, R, S, T, M, P, Q, str_d, str_h, str_w, pad_d, pad_h, pad_w)) {
			std::cerr << "Launch error" << std::endl;
		}
	}
  // output
  std::cout << "Result" << std::endl;
  cuda_error = cudaMemcpy(h_O, d_O, sizeof(float) * K * M * P * Q * N, cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess) {
    std::cerr << "Line " << __LINE__ << " memcpy error: " << cuda_error << std::endl;
    exit(1);
  }
  for (int i = 0; i < 100; ++i) {
    std::cout << h_O[i] << " ";
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

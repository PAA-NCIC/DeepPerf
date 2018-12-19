extern "C"
__global__ void sconv_bprop_C1_N64 (
    float* param_test,
    float* param_I,
    const float*  param_F,
    const float*  param_E,
    float param_alpha,
    int param_N,
    int param_K,
    int param_D,
    int param_H,
    int param_W,
    int param_WN,
    int param_HWN,
    int param_DHWN,
    int param_C,
    int param_CRST,
    int param_RST,
    int param_magic_RST,
    int param_shift_RST,
    int param_RS,
    int param_magic_RS,
    int param_shift_RS,
    int param_S,
    int param_magic_S,
    int param_shift_S,
    int param_pad_d,
    int param_pad_h,
    int param_pad_w,
    int param_str_d,
    int param_str_h,
    int param_str_w,
    int param_Q,
    int param_PQ,
    int param_QN,
    int param_PQN,
    int param_MPQN,
    int param_magic_Q,
    int param_shift_Q,
    int param_magic_PQ,
    int param_shift_PQ,
    int param_CRST8,
    int param_MPQN8) {
      __shared__ float shared[64 * 8 * 4 * 2];

      int tid = threadIdx.x;

      shared[tid] = 1;

      *param_I = shared[31 - tid];
      *param_test = shared[31 - tid];
    }

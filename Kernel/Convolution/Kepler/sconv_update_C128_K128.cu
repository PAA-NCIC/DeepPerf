extern "C"
__global__ void sconv_update_C128_K128 (
    float* param_test,
    float* param_F,
    const float* param_I,
    const float* param_E,
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
    int param_P,
    int param_Q,
    int param_PQ,
    int param_QN,
    int param_PQN,
    int param_MPQN,
    int param_magic_Q,
    int param_shift_Q,
    int param_magic_PQ,
    int param_shift_PQ,
    int param_part_P,
    int param_part_Q,
    int param_part_PQ) {
      __shared__ float share[(128 * 16 + 32) * 4 + 6];

      int tid = threadIdx.x;

      share[tid] = 1;

      *param_F = share[255 - tid];
      *param_test = share[255 - tid];
    }

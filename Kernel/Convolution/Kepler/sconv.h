#include <vector>
#include <string>
#include <map>
#include <cuda.h>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <math.h>

std::map<std::string, CUfunction> nervana_kernels;
std::vector<CUmodule> nervana_modules;

int len_d2b(int n) {
  int i, j = 0;
  i = n;
  while (i) {
    i /= 2;
    j++;
  }
  return j;
}

void magic32(unsigned int nmax, unsigned int d, unsigned int& m, unsigned int& p) {
  long nc = ((nmax + 1) / d) * d - 1;
  long nbits = len_d2b(nmax);
  std::cout << "nbits " << nbits << std::endl;
  for(p = 0; p < 2 * nbits + 1; p++) {   
    if(pow(2, p) > nc * (d - 1 - (long)(pow(2, p) - 1) % d)) {
      m = (pow(2, p) + d - 1 -(long)(pow(2, p) - 1) % d) / d;
      std::cout << "m " << m << std::endl;
      std::cout << "p " << p << std::endl;
      return;
    }   
  }   
  return;
}

void magic64(unsigned int d, unsigned int& magic, unsigned int& shift) {
  // 3 is a special case that only ends up in the high bits
  // if the nmax is 0xffffffff
  // we can't use 0xffffffff for all cases as some return a 33 bit
  // magic number
  unsigned long nmax;
  if(d == 3)
    nmax = 0xffffffff;
  else
    nmax = 0x7fffffff;
  magic32(nmax, d, magic, shift);
  if(magic != 1)
    shift -= 32;
}

bool load_kernels(const char* const base_path_cstr) {
    //better would be a vector<string>, but there is a bug in nvcc that prevents this
    // (bug report filed)
    const int NUM_KERNELS = 6;
    std::string names[NUM_KERNELS] = {
        "sconv_fprop_K64_N64",
        "sconv_fprop_K128_N128",
        "sconv_bprop_C128_N128",
        "sconv_bprop_C64_N64",
        "sconv_bprop_C1_N64",
        "sconv_update_C128_K128"
    };

    std::string base_path(base_path_cstr);

    for (int i = 0; i < NUM_KERNELS; ++i) {
      std::string kernel = names[i];
        if (nervana_kernels.count(kernel) > 0)
            continue;

        CUmodule module;

        std::string path = base_path + kernel + std::string(".cubin");
        CUresult res = cuModuleLoad(&module, path.c_str());

        if (res != CUDA_SUCCESS) {
            std::cerr << "Failed to load: " << kernel << " " << res << std::endl;
            return false;
        }

        nervana_modules.push_back(module);

        CUfunction function;
        res = cuModuleGetFunction(&function, module, kernel.c_str());
        if (res != CUDA_SUCCESS) {
            std::cerr << "Failed to extract: " << kernel << " " << res << std::endl;
            return false;
        }

        nervana_kernels.insert(std::make_pair(kernel, function));
    }

    return true;
}

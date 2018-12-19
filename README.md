# DeepPerf

DeepPerf is developed to understand GPU microarchitectural features and improve performance for compute-intensive kernels. The methodology relies on a reverse engineering approach to crack the GPU ISA encodings in order to build a GPU assembler. An assembly microbenchmark suite correlates microarchitectural features with their performance factors to uncover instruction-level and memory hierarchy preferences.
We use SGEMM and Convolution as examples to show the ways to achieve bare-metal performance tuning. In your deep learning framework, you could use directly these sass code to speed up the performance.

The toolchain is an attempt to automatically crack different GPU ISA encodings and build an assembler adaptively for the purpose of performance enhancements to applications on GPUs.
There are three directories in this folder, which consists of three major steps to optimize a cuda code in the assembly level. All the tools cover three recent NVIDIA GPU architecture, Kepler, Maxwell and Pascal.



#include <iostream>

int main(){
  int dev = 0;
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, dev);
  std::cout << "GPU Device" << dev << ": " << devProp.name << std::endl;
  std::cout << "Num SM: " << devProp.multiProcessorCount << std::endl;
  std::cout << "Shared Memory Per Block: " << devProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
  std::cout << "Max Threads Per Block: " << devProp.maxThreadsPerBlock << std::endl;
  std::cout << "Max Threads Per SM: " << devProp.maxThreadsPerMultiProcessor << std::endl;
  std::cout << "Max Warps Per SM: " << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
}
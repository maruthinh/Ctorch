cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(fmnist)

set(CMAKE_PREFIX_PATH "/mnt/external_data/maruthi/local_softwares/libtorch/share/cmake/Torch/")
find_package(Torch REQUIRED)

include_directories(include)
add_subdirectory(src)
add_subdirectory(app)
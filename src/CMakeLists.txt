project(opt-lib CXX)

# explicitly set cmake policy for find_package imports
cmake_policy(SET CMP0144 NEW)

# Require version 3.4 to ensure compatibility with CUDA
find_package(Eigen3 3.4 REQUIRED)

add_library(opt INTERFACE)
target_include_directories(opt INTERFACE .)
target_link_libraries(opt INTERFACE Eigen3::Eigen GREEN::UTILS)

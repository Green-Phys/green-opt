project(opt-lib CXX)

# Require version 3.4 to ensure compatibility with CUDA
find_package(Eigen3 3.4 REQUIRED)

add_library(opt INTERFACE)
target_include_directories(opt INTERFACE .)
target_link_libraries(opt INTERFACE Eigen3::Eigen GREEN::UTILS)

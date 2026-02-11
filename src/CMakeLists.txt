project(opt-lib CXX)

# Require version 3.4 to ensure compatibility with CUDA
find_package(Eigen3 REQUIRED)
if(Eigen3_VERSION VERSION_LESS 3.4)
  message(FATAL_ERROR "Eigen3 version ${Eigen3_VERSION} is too old. Need at least 3.4")
endif()

add_library(opt INTERFACE)
target_include_directories(opt INTERFACE .)
target_link_libraries(opt INTERFACE Eigen3::Eigen GREEN::UTILS)

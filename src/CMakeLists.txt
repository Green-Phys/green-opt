project(opt-lib CXX)

# Use NO_MODULE to ensure Eigen3's CMake package configuration file is used instead of a FindEigen3.cmake module.
# Require version 3.3 to ensure compatibility with features introduced in that version.
find_package(Eigen3 3.3 NO_MODULE REQUIRED)

add_library(opt INTERFACE)
target_include_directories(opt INTERFACE .)
target_link_libraries(opt INTERFACE Eigen3::Eigen GREEN::UTILS)

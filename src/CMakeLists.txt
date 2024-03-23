project(opt-lib CXX)

find_package(Eigen3 REQUIRED)

add_library(opt INTERFACE)
target_include_directories(opt INTERFACE .)
target_link_libraries(opt INTERFACE Eigen3::Eigen GREEN::UTILS)
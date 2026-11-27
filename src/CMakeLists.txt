project(opt-lib CXX)

# Require version 3.4 or later to ensure compatibility with CUDA.
# Don't pass the version to find_package: Eigen's version file treats a
# "3.4" request as "3.4.x only" and would reject Eigen 5.
find_package(Eigen3 REQUIRED)
if (Eigen3_VERSION VERSION_LESS 3.4)
    message(FATAL_ERROR "green-opt requires Eigen >= 3.4 (found ${Eigen3_VERSION})")
endif ()

add_library(opt INTERFACE)
target_include_directories(opt INTERFACE .)
target_link_libraries(opt INTERFACE Eigen3::Eigen GREEN::UTILS)

project(opt_test)

Include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0 # or a later release_
)

FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

add_executable(opt_test opt_test.cpp)
target_link_libraries(opt_test
        PRIVATE
        Catch2::Catch2WithMain
        GREEN::OPT)

include(CTest)
include(Catch)
catch_discover_tests(opt_test)
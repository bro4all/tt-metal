cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_test_binary_llk)

add_executable(metal_example_test_binary_llk)
target_sources(metal_example_test_binary_llk PRIVATE main.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_test_binary_llk PUBLIC TT::Metalium)

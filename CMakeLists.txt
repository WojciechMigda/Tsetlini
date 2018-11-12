cmake_minimum_required(VERSION 3.5)

project(TsetlinMachineToolkit VERSION 1.0.0 LANGUAGES CXX)

add_library(tsetlin
    src/tsetlin.cpp
    src/tsetlin_config.cpp
    src/tsetlin_state.cpp
)

target_include_directories(tsetlin
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(tsetlin PRIVATE -Werror -Wall -march=native)

target_compile_features(tsetlin PRIVATE cxx_auto_type) # 11
target_compile_features(tsetlin PRIVATE cxx_generic_lambdas) # 14



enable_testing()
add_subdirectory(tests)

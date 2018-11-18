cmake_minimum_required(VERSION 3.5)

project(TsetlinMachineToolkit VERSION 1.0.0 LANGUAGES CXX)

add_library(tsetlin
    src/tsetlin.cpp
    src/tsetlin_config.cpp
    src/tsetlin_state.cpp
)

add_library(pytsetlin SHARED
    src/c_api.cpp
)

target_include_directories(tsetlin
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(pytsetlin
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(tsetlin PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Werror -Wall -march=native -std=c++17>)
target_compile_options(pytsetlin PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Werror -Wall -march=native -std=c++17>)
set_property(TARGET tsetlin PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(pytsetlin tsetlin)


add_subdirectory(examples)

enable_testing()
add_subdirectory(tests)

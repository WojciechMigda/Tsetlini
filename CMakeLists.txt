cmake_minimum_required(VERSION 3.5)

project(TsetlinMachineToolkit VERSION 1.0.0 LANGUAGES CXX)

add_library(tsetlin_static
    src/tsetlin.cpp
    src/tsetlin_config.cpp
    src/tsetlin_state.cpp
)

add_library(tsetlin SHARED
    src/c_api.cpp
)

target_include_directories(tsetlin_static
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(tsetlin
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(tsetlin_static PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Werror -Wall -march=native -std=c++17>)
target_compile_options(tsetlin PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Werror -Wall -march=native -std=c++17>)
set_property(TARGET tsetlin_static PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(tsetlin tsetlin_static)


add_subdirectory(examples)

enable_testing()
add_subdirectory(tests)

# Create target 'install' for installing libraries.
install(TARGETS tsetlin tsetlin_static DESTINATION lib/tsetlin)
install(DIRECTORY include/ DESTINATION include/tsetlin)

add_executable(noisy-xor
    src/main.cpp
)

target_include_directories(noisy-xor
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

target_link_libraries(noisy-xor tsetlin)
target_compile_options(noisy-xor PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Wall -march=native -std=c++17>)

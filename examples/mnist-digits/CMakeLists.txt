add_executable(mnist-digits
    src/main.cpp
)

target_include_directories(mnist-digits
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

target_link_libraries(mnist-digits tktsetlin_static)
target_compile_options(mnist-digits PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Wall -march=native -std=c++17>)

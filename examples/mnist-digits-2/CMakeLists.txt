add_executable(mnist-digits-2
    src/main.cpp
)

target_include_directories(mnist-digits-2
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

target_link_libraries(mnist-digits-2 tsetlini_static)
target_compile_options(mnist-digits-2 PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Wall -Werror -march=${MARCH} -std=c++17>)

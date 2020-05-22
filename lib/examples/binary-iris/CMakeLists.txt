add_executable(binary-iris
    src/main.cpp
)

target_include_directories(binary-iris
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

target_link_libraries(binary-iris tsetlini_static)
target_compile_options(binary-iris PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Wall -march=${MARCH} -std=c++17>)

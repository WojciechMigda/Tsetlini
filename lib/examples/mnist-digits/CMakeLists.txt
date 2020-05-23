add_executable(mnist-digits
    src/main.cpp
)

target_include_directories(mnist-digits
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

target_link_libraries(mnist-digits tsetlini_static)
target_compile_definitions(mnist-digits PRIVATE DATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/src)

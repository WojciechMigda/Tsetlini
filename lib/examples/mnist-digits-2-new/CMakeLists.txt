add_executable(mnist-digits-2-new
    src/main.cpp
)

target_include_directories(mnist-digits-2-new
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib/include>
)

target_link_libraries(mnist-digits-2-new tsetlini_static)

add_executable(mnist-digits-partial
    src/main.cpp
)

target_include_directories(mnist-digits-partial
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib/include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib/src>
)

target_link_libraries(mnist-digits-partial tsetlini_static)

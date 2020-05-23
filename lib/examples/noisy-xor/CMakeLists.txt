add_executable(noisy-xor
    src/main.cpp
)

target_include_directories(noisy-xor
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

target_link_libraries(noisy-xor tsetlini_static)

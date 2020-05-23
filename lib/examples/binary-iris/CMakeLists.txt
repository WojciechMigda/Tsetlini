add_executable(binary-iris
    src/main.cpp
)

target_include_directories(binary-iris
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

target_link_libraries(binary-iris tsetlini_static)

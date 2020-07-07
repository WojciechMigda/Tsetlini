add_executable(imdb
    src/main.cpp
)

target_include_directories(imdb
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib/include>
)

target_link_libraries(imdb tsetlini_static)

add_executable(breast-cancer
    src/main.cpp
)

target_include_directories(breast-cancer
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib/include>
)

target_link_libraries(breast-cancer tsetlini_static)

add_executable(california-housing
    src/main.cpp
)

target_include_directories(california-housing
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib/include>
)

target_link_libraries(california-housing tsetlini_static)

add_executable(california-housing-bitwise
    src/main-bitwise.cpp
)

target_include_directories(california-housing-bitwise
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib/include>
)

target_link_libraries(california-housing-bitwise tsetlini_static)

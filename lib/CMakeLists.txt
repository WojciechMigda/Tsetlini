set( CMAKE_CXX_STANDARD 17 )

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


include(FindOpenMP)
if(NOT OpenMP_FOUND)
    message(FATAL_ERROR "OpenMP for C++ is required but is missing")
endif()

set(SOURCES
    src/tsetlini.cpp
    src/tsetlini_params.cpp
    src/tsetlini_state.cpp
    src/tsetlini_state_json.cpp
)

add_library(tsetlini_static src/dummy.cpp)
target_sources( tsetlini_static PRIVATE
  ${SOURCES}
)


add_library(tsetlini SHARED src/dummy.cpp)
target_sources( tsetlini PRIVATE
  ${SOURCES}
)

target_include_directories(tsetlini_static
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(tsetlini
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(tsetlini_static PRIVATE ${OpenMP_CXX_FLAGS})
target_compile_options(tsetlini PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(tsetlini_static PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(tsetlini PRIVATE ${OpenMP_CXX_FLAGS})

add_subdirectory(examples)

enable_testing()
add_subdirectory(tests)

# Unit tests (future?)
add_executable( utests "utests/dummy.cpp" )
target_sources( utests PRIVATE
  "utests/main.cpp"
  "utests/suite_aligned_array.cpp"
)


# Create target 'install' for installing libraries.
install(TARGETS tsetlini tsetlini_static DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/tsetlini)

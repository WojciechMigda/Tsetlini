cmake_minimum_required(VERSION 3.5)

project(Tsetlini VERSION 1.0.0 LANGUAGES CXX)

if(NOT MARCH)
  set(MARCH "native" CACHE STRING "-march flag for the compiler")
endif()

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

add_library(tsetlini_static ${SOURCES})

add_library(tsetlini SHARED ${SOURCES})

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

target_compile_options(tsetlini_static PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Werror -Wall -march=${MARCH} -std=c++17> ${OpenMP_CXX_FLAGS})
target_compile_options(tsetlini PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Werror -Wall -march=${MARCH} -std=c++17> ${OpenMP_CXX_FLAGS})
target_link_libraries(tsetlini_static PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(tsetlini PRIVATE ${OpenMP_CXX_FLAGS})
set_property(TARGET tsetlini_static PROPERTY POSITION_INDEPENDENT_CODE ON)

add_subdirectory(examples)

enable_testing()
add_subdirectory(tests)

# Create target 'install' for installing libraries.
install(TARGETS tsetlini tsetlini_static DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/tsetlini)

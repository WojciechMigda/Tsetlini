cmake_minimum_required(VERSION 3.5)

project(TsetlinMachineToolkit VERSION 1.0.0 LANGUAGES CXX)

if(NOT MARCH)
  set(MARCH "native" CACHE STRING "-march flag for the compiler")
endif()

include(FindOpenMP)
if(NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "OpenMP for C++ is required but is missing")
endif()

set(SOURCES
    src/tsetlin.cpp
    src/tsetlin_params.cpp
    src/tsetlin_state.cpp
    src/tsetlin_state_json.cpp
)

add_library(tktsetlin_static ${SOURCES})

add_library(tktsetlin SHARED ${SOURCES})

target_include_directories(tktsetlin_static
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(tktsetlin
    PUBLIC 
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(tktsetlin_static PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Werror -Wall -march=${MARCH} -std=c++17> ${OpenMP_CXX_FLAGS})
target_compile_options(tktsetlin PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Werror -Wall -march=${MARCH} -std=c++17> ${OpenMP_CXX_FLAGS})
target_link_libraries(tktsetlin_static PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(tktsetlin PRIVATE ${OpenMP_CXX_FLAGS})
set_property(TARGET tktsetlin_static PROPERTY POSITION_INDEPENDENT_CODE ON)

add_subdirectory(examples)

enable_testing()
add_subdirectory(tests)

# Create target 'install' for installing libraries.
install(TARGETS tktsetlin tktsetlin_static DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/tktsetlin)

# https://github.com/pabloariasal/modern-cmake-sample/blob/master/libjsonutils/test/CMakeLists.txt

include(GoogleTest)

# Download and unpack googletest at configure time
configure_file(${CMAKE_SOURCE_DIR}/cmake/GoogleTest-CMakeLists.txt.in ${CMAKE_BINARY_DIR}/googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
                 ${CMAKE_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)


# Now simply link against gtest or gtest_main as needed. Eg
add_executable(tsetlin-test
    src/main.cpp
    src/test_config.cpp
    src/test_state.cpp
#    src/test_aligned_array.cpp
)

target_include_directories(tsetlin-test
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(tsetlin-test gtest_main tsetlin)
target_compile_options(tsetlin-test PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Wall -march=native -std=c++17>)

gtest_discover_tests(tsetlin-test)
add_test(NAME monolithic COMMAND tsetlin-test)

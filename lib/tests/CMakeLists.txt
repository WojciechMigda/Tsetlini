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
add_executable(tsetlini-test
    src/main.cpp
    src/test_algo_classic.cpp
    src/test_classifier.cpp
    src/test_classifier_state_json.cpp
    src/test_params.cpp
    src/test_bit_matrix.cpp
    src/test_numeric_matrix.cpp
)

target_include_directories(tsetlini-test
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib/include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/lib/src
)

target_compile_options(tsetlini-test PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(tsetlini-test gtest_main tsetlini_static ${OpenMP_CXX_FLAGS})

gtest_discover_tests(tsetlini-test)
add_test(NAME monolithic COMMAND tsetlini-test)

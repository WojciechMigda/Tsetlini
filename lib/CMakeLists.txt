set( CMAKE_CXX_STANDARD 17 )

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

cmake_policy(SET CMP0067 NEW) # to preserve CXX standard for check_include_file_cxx
include(CheckIncludeFileCXX)
check_include_file_cxx("optional" HAS_CXX_OPTIONAL)
if(NOT HAS_CXX_OPTIONAL)
    message(FATAL_ERROR "C++ <optional> header is required but is missing")
endif()
check_include_file_cxx("variant" HAS_CXX_VARIANT)
if(NOT HAS_CXX_VARIANT)
    message(FATAL_ERROR "C++ <variant> header is required but is missing")
endif()

include(FindOpenMP)
if(NOT OpenMP_FOUND)
    message(FATAL_ERROR "OpenMP for C++ is required but is missing")
endif()

################################################################################

set(SOURCES
    src/estimator_state.cpp
    src/estimator_state_cache.cpp
    src/loss_fn.cpp
    src/ta_state.cpp
    src/tsetlini.cpp
    src/tsetlini_params.cpp
    src/tsetlini_state.cpp
    src/tsetlini_state_json.cpp
    src/box_muller_approx.cpp
)

add_library( tsetlini_ObjLib OBJECT src/dummy.cpp )
target_sources( tsetlini_ObjLib PRIVATE ${SOURCES} )

target_include_directories(tsetlini_ObjLib
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options( tsetlini_ObjLib PRIVATE ${OpenMP_CXX_FLAGS})

target_compile_features(tsetlini_ObjLib PRIVATE cxx_rvalue_references)

################################################################################

add_library( tsetlini SHARED src/dummy.cpp )
target_sources( tsetlini PRIVATE
  $<TARGET_OBJECTS:tsetlini_ObjLib>
)

target_include_directories( tsetlini
    PUBLIC
        $<TARGET_PROPERTY:tsetlini_ObjLib,INTERFACE_INCLUDE_DIRECTORIES> )
target_compile_features( tsetlini
    PUBLIC $<TARGET_PROPERTY:tsetlini_ObjLib,INTERFACE_COMPILE_FEATURES> )

target_link_libraries(tsetlini PRIVATE ${OpenMP_CXX_FLAGS})

################################################################################

add_library( tsetlini_static src/dummy.cpp )
target_sources( tsetlini_static PRIVATE
  $<TARGET_OBJECTS:tsetlini_ObjLib>
)

target_include_directories( tsetlini_static
    PUBLIC
        $<TARGET_PROPERTY:tsetlini_ObjLib,INTERFACE_INCLUDE_DIRECTORIES> )
target_compile_features( tsetlini_static
    PUBLIC $<TARGET_PROPERTY:tsetlini_ObjLib,INTERFACE_COMPILE_FEATURES> )

target_link_libraries(tsetlini_static PRIVATE ${OpenMP_CXX_FLAGS})

################################################################################

add_subdirectory(examples)

################################################################################

enable_testing()
add_subdirectory(utests)

################################################################################

# Create target 'install' for installing libraries.
install(TARGETS tsetlini tsetlini_static DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/tsetlini)

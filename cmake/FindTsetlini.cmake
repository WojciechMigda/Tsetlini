include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)

find_path(Tsetlini_INCLUDE_DIR tsetlini.hpp PATH_SUFFIXES tsetlini)
find_library(Tsetlini_LIBRARY tsetlini libtsetlini)

find_package_handle_standard_args(Tsetlini
   DEFAULT_MSG
    Tsetlini_INCLUDE_DIR
    Tsetlini_LIBRARY
)

mark_as_advanced(Tsetlini_LIBRARY Tsetlini_INCLUDE_DIR)

if(Tsetlini_FOUND)
  set(Tsetlini_LIBRARIES    ${Tsetlini_LIBRARY})
  set(Tsetlini_INCLUDE_DIRS ${Tsetlini_INCLUDE_DIR})
endif()

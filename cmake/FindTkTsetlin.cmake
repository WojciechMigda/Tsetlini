include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)

find_path(TkTsetlin_INCLUDE_DIR tsetlin.hpp PATH_SUFFIXES tktsetlin)
find_library(TkTsetlin_LIBRARY tktsetlin libtktsetlin)

find_package_handle_standard_args(TkTsetlin
   DEFAULT_MSG
    TkTsetlin_INCLUDE_DIR
    TkTsetlin_LIBRARY
)

mark_as_advanced(TkTsetlin_LIBRARY TkTsetlin_INCLUDE_DIR)

if(TkTsetlin_FOUND)
  set(TkTsetlin_LIBRARIES    ${TkTsetlin_LIBRARY})
  set(TkTsetlin_INCLUDE_DIRS ${TkTsetlin_INCLUDE_DIR})
endif()

function(add_ut)
    set( options TEST_NAME )
    set( oneValueArgs EXE_NAME SOURCE )
    set( multiValueArgs "" )
    cmake_parse_arguments( parsed "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    if( parsed_UNPARSED_ARGUMENTS )
        message( FATAL_ERROR "[add_ut] Bad arguments: ${parsed_UNPARSED_ARGUMENTS}" )
    endif()

    if( NOT parsed_TEST_NAME )
        set( parsed_TEST_NAME ${parsed_EXE_NAME} )
    endif()

    add_executable( ${parsed_EXE_NAME} "src/dummy.cpp" )
    target_sources( ${parsed_EXE_NAME} PRIVATE ${parsed_SOURCE} )
    target_include_directories( ${parsed_EXE_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include )
    target_link_libraries( ${parsed_EXE_NAME} tsetlini_static )
    add_test( ${parsed_TEST_NAME} ${parsed_EXE_NAME} )
endfunction()


add_ut( EXE_NAME test_aligned_array     SOURCE "src/test_aligned_array.cpp" )
add_ut( EXE_NAME test_bit_matrix        SOURCE "src/test_bit_matrix.cpp" )
add_ut( EXE_NAME test_numeric_matrix    SOURCE "src/test_numeric_matrix.cpp" )
add_ut( EXE_NAME test_classifier_params SOURCE "src/test_classifier_params.cpp" )
add_ut( EXE_NAME test_regressor_params  SOURCE "src/test_regressor_params.cpp" )

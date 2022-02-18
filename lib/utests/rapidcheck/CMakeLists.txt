# https://github.com/emil-e/rapidcheck
# commit: 08b5058 on Dec 20, 2021

# code below is copied from rapidcheck main CMakeLists.txt
# with the only modification being explicitly declaring
# built library as STATIC

add_library(rapidcheck STATIC
  src/BeforeMinimalTestCase.cpp
  src/Check.cpp
  src/Classify.cpp
  src/GenerationFailure.cpp
  src/Log.cpp
  src/Random.cpp
  src/Show.cpp
  src/detail/Any.cpp
  src/detail/Assertions.cpp
  src/detail/Base64.cpp
  src/detail/Configuration.cpp
  src/detail/DefaultTestListener.cpp
  src/detail/FrequencyMap.cpp
  src/detail/ImplicitParam.cpp
  src/detail/LogTestListener.cpp
  src/detail/MapParser.cpp
  src/detail/MulticastTestListener.cpp
  src/detail/ParseException.cpp
  src/detail/Platform.cpp
  src/detail/Property.cpp
  src/detail/PropertyContext.cpp
  src/detail/ReproduceListener.cpp
  src/detail/Results.cpp
  src/detail/Serialization.cpp
  src/detail/StringSerialization.cpp
  src/detail/TestMetadata.cpp
  src/detail/TestParams.cpp
  src/detail/Testing.cpp
  src/gen/Numeric.cpp
  src/gen/Text.cpp
  src/gen/detail/ExecHandler.cpp
  src/gen/detail/GenerationHandler.cpp
  src/gen/detail/Recipe.cpp
  src/gen/detail/ScaleInteger.cpp
)

target_include_directories(rapidcheck PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include>  # <prefix>/include
)

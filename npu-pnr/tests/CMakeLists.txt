include(FetchContent)

find_package(ortools CONFIG REQUIRED)
find_package(SQLite3 REQUIRED)

find_package(GTest QUIET)
if(NOT GTest_FOUND)
  message(STATUS "GoogleTest not found; fetching from source")
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(gtest_build_tests OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

set(NPU_PNR_TEST_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}
)

set(NPU_PNR_TEST_COMMON_LIBRARIES
  npu_pnr
  ortools::ortools
)
if(TARGET SQLite::SQLite3)
  list(APPEND NPU_PNR_TEST_COMMON_LIBRARIES SQLite::SQLite3)
else()
  list(APPEND NPU_PNR_TEST_COMMON_LIBRARIES ${SQLite3_LIBRARIES})
endif()

set(NPU_PNR_TEST_RUNTIME_DIRS)
foreach(runtime_target IN ITEMS ortools::ortools GTest::gtest GTest::gtest_main)
  if(TARGET ${runtime_target})
    list(APPEND NPU_PNR_TEST_RUNTIME_DIRS "$<TARGET_FILE_DIR:${runtime_target}>")
  endif()
endforeach()

foreach(sqlite_library IN LISTS SQLite3_LIBRARIES)
  if(IS_ABSOLUTE "${sqlite_library}")
    get_filename_component(sqlite_runtime_dir "${sqlite_library}" DIRECTORY)
    list(APPEND NPU_PNR_TEST_RUNTIME_DIRS "${sqlite_runtime_dir}")
  endif()
endforeach()

list(REMOVE_DUPLICATES NPU_PNR_TEST_RUNTIME_DIRS)

# Share common test configuration without wrapping add_executable(...).
add_library(npu_pnr_test_support INTERFACE)
target_compile_features(npu_pnr_test_support INTERFACE cxx_std_20)
target_include_directories(npu_pnr_test_support INTERFACE
  ${NPU_PNR_TEST_INCLUDE_DIRS}
)
target_link_libraries(npu_pnr_test_support INTERFACE
  ${NPU_PNR_TEST_COMMON_LIBRARIES}
)

file(GLOB_RECURSE UNIT_TEST_SOURCES
  CONFIGURE_DEPENDS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*_test.cpp"
)
list(FILTER UNIT_TEST_SOURCES EXCLUDE REGEX "^regression/")
list(SORT UNIT_TEST_SOURCES)

if(NOT UNIT_TEST_SOURCES)
  message(FATAL_ERROR
    "No unit test sources matching '*_test.cpp' were found under ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

add_executable(unit_tests ${UNIT_TEST_SOURCES})
target_link_libraries(unit_tests PRIVATE
  npu_pnr_test_support
  GTest::gtest_main
)

set(NPU_PNR_REGRESSION_MANIFEST
  "${CMAKE_CURRENT_SOURCE_DIR}/regression_cases.toml"
)

add_executable(regression_tests
  regression/main.cpp
)
target_link_libraries(regression_tests PRIVATE
  npu_pnr_test_support
  GTest::gtest
)
target_compile_definitions(regression_tests PRIVATE
  NPU_PNR_DEFAULT_REGRESSION_MANIFEST=\"${NPU_PNR_REGRESSION_MANIFEST}\"
)

foreach(test_target IN ITEMS unit_tests regression_tests)
  # Linux-only: keep tests runnable from the build tree by setting the build
  # rpath explicitly with CMake properties instead of raw linker flags.
  set_target_properties(${test_target} PROPERTIES
    BUILD_RPATH "${NPU_PNR_TEST_RUNTIME_DIRS}"
    BUILD_WITH_INSTALL_RPATH FALSE
  )
endforeach()

gtest_add_tests(TARGET unit_tests TEST_LIST UNIT_TEST_CASES)

# Intentionally no CTest integration for regression_tests yet.

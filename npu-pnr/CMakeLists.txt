cmake_minimum_required(VERSION 3.15...4.0)

project(
  NPUPnR
  VERSION 0.1
  DESCRIPTION "NPU Place and Route Flow"
  LANGUAGES CXX)

# Ensure -std=c++xx instead of -std=g++xx
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

option(ENABLE_GPROF "Enable GNU Profiler gprof" OFF)
if(ENABLE_GPROF)
  message(STATUS "GNU Profiler gprof enabled (force setting CMAKE_BUILD_TYPE to Debug)")
  set(CMAKE_BUILD_TYPE "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
endif()

include(CTest)
include(FetchContent)

add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(tools)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

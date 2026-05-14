add_executable(pnr pnr.cpp)
target_compile_features(pnr PRIVATE cxx_std_20)
target_link_libraries(pnr PRIVATE npu_pnr)
target_include_directories(pnr PUBLIC ${PROJECT_SOURCE_DIR}/external_libs)

# add_executable(router router.cpp)
# target_compile_features(router PRIVATE cxx_std_20)
# target_link_libraries(router PRIVATE npu_pnr)
# target_include_directories(router PUBLIC ${PROJECT_SOURCE_DIR}/external_libs)

# add_executable(debugger debugger.cpp)
# target_compile_features(debugger PRIVATE cxx_std_20)
# target_link_libraries(debugger PRIVATE npu_pnr)
# target_include_directories(debugger PUBLIC ${PROJECT_SOURCE_DIR}/external_libs)

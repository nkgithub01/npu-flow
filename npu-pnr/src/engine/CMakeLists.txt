file(GLOB_RECURSE engine_headers "*.hpp")
file(GLOB_RECURSE engine_sources "*.cpp")

find_package(SQLite3 REQUIRED)

target_sources(npu_pnr
    PRIVATE ${engine_sources}
    PUBLIC ${engine_headers})

target_include_directories(npu_pnr PUBLIC ${PROJECT_SOURCE_DIR}/external_libs)
target_link_libraries(npu_pnr PRIVATE ${SQLite3_LIBRARIES})

file(GLOB_RECURSE utils_headers "*.hpp")

# TODO: avoid exposing the utils headers in the public interface
target_sources(npu_pnr PUBLIC ${utils_headers})

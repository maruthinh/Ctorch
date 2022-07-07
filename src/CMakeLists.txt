file(GLOB SOURCE_LIST "*.cpp")
add_library(models STATIC ${SOURCE_LIST} ${HEADER_LIST})
target_link_libraries(models ${TORCH_LIBRARIES})

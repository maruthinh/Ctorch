add_executable(app app.cpp)
target_link_libraries(app models)
set_property(TARGET app PROPERTY CXX_STANDARD 14)
set(CMAKE_EXECUTABLE_SUFFIX .exe)
project(flag-recognize)
find_package(OpenCV REQUIRED)
include_directories("/usr/include/eigen3/")
add_library(
   detection
   detection.cpp
 )
add_executable(flag-recognize flag-recognize.cpp)
target_link_libraries(flag-recognize detection ${OpenCV_LIBS})

cmake_minimum_required(VERSION 3.4.1)

project(ncnn_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_C_FLAGS -pthread)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -luuid -pthread")
add_compile_options(-pthread)


find_package(PkgConfig REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

include_directories(${CMAKE_CURRENT_BINARY_DIR})


find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(OpenCV REQUIRED)

set(ncnn_DIR "/media/test/test/test1/ncnn/build/install/lib/cmake/ncnn/")
find_package(ncnn REQUIRED)
if(NOT TARGET ncnn)
    message(WARNING "ncnn NOT FOUND!  Please set ncnn_DIR environment variable")
else()
    message("ncnn FOUND")
endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(ncnn_demo mainFV2.cpp jsonConfig.cpp hungarian.cpp kalmanTracker.cpp yolo-fastestv2.cpp)

target_include_directories(ncnn_demo PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}"
                          )

target_link_libraries(
    ncnn_demo
    ${CMAKE_THREAD_LIBS_INIT}
    ncnn
    ${OpenCV_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
)

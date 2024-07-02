cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(GameOfLife LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 17)

set(CUDNN_ROOT "C:/Program Files/NVIDIA/CUDNN/v9.2")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CUDNN_ROOT}/cudnn_samples/cmake")
set(cuDNN_INCLUDE_DIR "C:/Program Files/NVIDIA/CUDNN/v9.2/include/11.8")
set(cuDNN_LIBRARY_DIR "C:/Program Files/NVIDIA/CUDNN/v9.2/lib/11.8/x64")

find_package(Doxygen REQUIRED)
find_package(CUDNN REQUIRED)

find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

include_directories(${CUDNN_ROOT} ${cuDNN_INCLUDE_DIR})
link_directories(
    ${CUDNN_ROOT}
    ${cuDNN_LIBRARY_DIR}
)

add_subdirectory(glfw)
add_subdirectory(imgui)


if ( MSVC )
    SET( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ENTRY:mainCRTStartup" )
endif()

set(SOURCES
    objects.cpp
    objects.hpp
    windows.cpp
    windows.hpp
    main.cpp
    gpu.cu 
    gpu.cuh
    cuda_header.cuh
)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCES})

target_link_libraries(${CMAKE_PROJECT_NAME}
    PRIVATE
        glfw
        imgui
        ${OPENGL_LIBRARIES}
)


set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)


target_include_directories(${CMAKE_PROJECT_NAME}
    PRIVATE
        include
)

target_compile_options(${CMAKE_PROJECT_NAME}
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_75,code=sm_75>
)

target_link_options(${CMAKE_PROJECT_NAME}
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_75,code=sm_75>
)

target_link_libraries(${CMAKE_PROJECT_NAME}
    PRIVATE
        cuda
        cudnn64_9
)

set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 75)


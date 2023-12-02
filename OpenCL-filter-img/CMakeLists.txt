#
cmake_minimum_required (VERSION 3.8)

# create project
project ("OclFilterImg")
# Add source to this project's executable.
add_executable (OclFilterImg "OpenCL-filter-img.cpp" "BMP.h")

# OpencCL headers
target_include_directories(OclFilterImg PRIVATE "$ENV{CUDA_PATH}/include")

# OpenCL library
set(opencl_lib_folder "$ENV{CUDA_PATH}/lib")

# check 32 or 64 bits
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    # 64 bits
   set(opencl_lib_folder "${opencl_lib_folder}/x64")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
    # 32 bits
    set(opencl_lib_folder "${opencl_lib_folder}/Win32")
endif()

# link OpenCL library
target_link_libraries(OclFilterImg ${opencl_lib_folder}/OpenCL.lib)

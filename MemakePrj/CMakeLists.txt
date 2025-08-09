#
cmake_minimum_required (VERSION 3.8)

project ("MemakePrj")

# Add source to this project's executable.
add_executable (MemakePrj "main.cpp" "Memake/Memake.cpp" "Memake/Vector2d.cpp")

# SDL2 headers
target_include_directories(MemakePrj PRIVATE "SDL2-2.0.14/include")

# gpu vendor: amd, nvidia
set(gpu_vendor "amd")
# 
if("${gpu_vendor}" STREQUAL "amd")
    message("gpu_vendor is 'amd'")
    # set define
    target_compile_definitions(MemakePrj PUBLIC GPU_VENDOR_IS_AMD)
    # OpenCL library
    set(opencl_lib_folder "$ENV{OCL_ROOT}/lib")
    # OpenCL headers
    target_include_directories(MemakePrj PRIVATE "$ENV{OCL_ROOT}/include")
elseif("${gpu_vendor}" STREQUAL "nvidia")
    message("gpu_vendor is 'nvidia'")
    # set define    
    target_compile_definitions(MemakePrj PUBLIC GPU_VENDOR_IS_NVIDIA)
    # OpencCL headers
    target_include_directories(MemakePrj PRIVATE "$ENV{CUDA_PATH}/include")
    # OpenCL library
    set(opencl_lib_folder "$ENV{CUDA_PATH}/lib")
else()
    message("gpu_vendor is 'UNKNOWN'")
endif()

# add SDL_MAIN_HANDLED definition to avoid 
# "LNK2019 unresolved external symbol SDL_main referenced in function main_getcmdline" 
add_definitions( -DSDL_MAIN_HANDLED )

# SDL library folder
set(SDL2_lib_folder "${PROJECT_SOURCE_DIR}/SDL2-2.0.14/lib")

message(${CMAKE_BUILD_TYPE})

# check 32 or 64 bits
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
   # 64 bits
   set(SDL2_lib_folder "${SDL2_lib_folder}/x64")
   
   if("${gpu_vendor}" STREQUAL "amd")
       set(opencl_lib_folder "${opencl_lib_folder}")
   elseif("${gpu_vendor}" STREQUAL "nvidia")
       set(opencl_lib_folder "${opencl_lib_folder}/x64")
   else()
       message("gpu_vendor is 'UNKNOWN'")
   endif()
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
   # 32 bits
   set(SDL2_lib_folder "${SDL2_lib_folder}/x86")
   if("${gpu_vendor}" STREQUAL "amd")
       message("no 32-bit library")
   elseif("${gpu_vendor}" STREQUAL "nvidia")
       set(opencl_lib_folder "${opencl_lib_folder}/Win32")
   else()
       message("gpu_vendor is 'UNKNOWN'")
   endif()
endif()

# link SDL2 static lib
target_link_libraries(MemakePrj ${SDL2_lib_folder}/SDL2.lib)
target_link_libraries(MemakePrj ${SDL2_lib_folder}/SDL2main.lib)

# link OpenCL library
target_link_libraries(MemakePrj ${opencl_lib_folder}/OpenCL.lib)

# copy dynamic lib to folder with executable file
file(COPY ${SDL2_lib_folder}/SDL2.dll  DESTINATION ${PROJECT_BINARY_DIR})
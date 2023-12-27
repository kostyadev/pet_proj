#
cmake_minimum_required (VERSION 3.8)

project ("MemakePrj")

# Add source to this project's executable.
add_executable (MemakePrj "main.cpp" "Memake/Memake.cpp" "Memake/Vector2d.cpp")

# SDL2 headers
target_include_directories(MemakePrj PRIVATE "SDL2-2.0.14/include")

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
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
    # 32 bits
    set(SDL2_lib_folder "${SDL2_lib_folder}/x86")
endif()

# link SDL2 static lib
target_link_libraries(MemakePrj ${SDL2_lib_folder}/SDL2.lib)
target_link_libraries(MemakePrj ${SDL2_lib_folder}/SDL2main.lib)

# copy dynamic lib to folder with executable file
file(COPY ${SDL2_lib_folder}/SDL2.dll  DESTINATION ${PROJECT_BINARY_DIR})
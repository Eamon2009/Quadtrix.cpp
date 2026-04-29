cmake_minimum_required(VERSION 3.10)

project(QuadtrixCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TORCH_DIR "" CACHE PATH "Path to the LibTorch root directory")
option(QUADTRIX_ENABLE_TORCH "Build the LibTorch example target" ON)

if(MINGW)
    message(FATAL_ERROR
        "LibTorch on Windows is not reliably supported with MinGW GCC 6.3. "
        "Use MSVC for the torch target, or configure with -DQUADTRIX_ENABLE_TORCH=OFF.")
endif()

if(QUADTRIX_ENABLE_TORCH)
    if(TORCH_DIR)
        list(APPEND CMAKE_PREFIX_PATH "${TORCH_DIR}")
    endif()

    find_package(Torch REQUIRED)

    add_executable(quadtrix_torch_cpp
        src/torch_example.cpp
    )

    target_include_directories(quadtrix_torch_cpp PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(quadtrix_torch_cpp PRIVATE ${TORCH_LIBRARIES})
    set_property(TARGET quadtrix_torch_cpp PROPERTY CXX_STANDARD 17)
    set_property(TARGET quadtrix_torch_cpp PROPERTY CXX_STANDARD_REQUIRED ON)

    if(MSVC)
        target_compile_options(quadtrix_torch_cpp PRIVATE /W4 /EHsc)
    else()
        target_compile_options(quadtrix_torch_cpp PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(WIN32)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(TARGET quadtrix_torch_cpp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:quadtrix_torch_cpp>
        )
    endif()
endif()

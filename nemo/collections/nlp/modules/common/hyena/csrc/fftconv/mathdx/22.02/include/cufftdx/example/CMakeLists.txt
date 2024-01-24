cmake_minimum_required(VERSION 3.18.0)

# cuFFTDxExamples project
project(cuFFTDxExamples LANGUAGES CXX CUDA)

# Find CUDA Toolkit packaged, required for NVRTC sample
find_package(CUDAToolkit)

# Project options
option(USE_MATHDX_PACKAGE "Use mathDx package to find cuFFTDx" ON)
option(USE_CUFFTDX_PACKAGE "Use cuFFTDx package to find cuFFTDx" OFF)

if(DEFINED cufftdx_ROOT OR DEFINED ENV{cufftdx_ROOT})
    SET(USE_CUFFTDX_PACKAGE ON CACHE BOOL "Use cuFFTDx package to find cuFFTDx" FORCE)
    SET(USE_MATHDX_PACKAGE OFF CACHE BOOL "Use mathDx package to find cuFFTDx" FORCE)
endif()

if(DEFINED mathdx_ROOT OR DEFINED ENV{mathdx_ROOT})
    SET(USE_CUFFTDX_PACKAGE OFF CACHE BOOL "Use cuFFTDx package to find cuFFTDx" FORCE)
    SET(USE_MATHDX_PACKAGE ON CACHE BOOL "Use mathDx package to find cuFFTDx" FORCE)
endif()

if(NOT TARGET cufftdx)
    if(USE_MATHDX_PACKAGE)
        message(STATUS "Using mathDx package to find cuFFTDx")
        # Find mathDx and cuFFTDx (mathDx's component)
        # Default path: "/opt/nvidia/mathdx/22.2", path to mathDx can be passed cmake in mathdx_ROOT variable
        find_package(mathdx REQUIRED COMPONENTS cufftdx CONFIG
            PATHS
                "${PROJECT_SOURCE_DIR}/../.." # example/cufftdx
                "${PROJECT_SOURCE_DIR}/../../.." # include/cufftdx/example
                "/opt/nvidia/mathdx/22.2"
        )
    elseif(USE_CUFFTDX_PACKAGE)
        message(STATUS "Using cuFFTDx package to find cuFFTDx")
        # Find cuFFTDx
        # Default path: "/opt/nvidia/mathdx/22.2/include/cufftdx", path to cuFFTDx can be passed cmake in cufftdx_ROOT variable
        find_package(cufftdx REQUIRED CONFIG PATHS "/opt/nvidia/mathdx/22.2/include/cufftdx" "${PROJECT_SOURCE_DIR}/../../cufftdx")
    else()
        message(FATAL_ERROR "No cuFFTDx package found")
    endif()
endif()

if((NOT TARGET cufftdx) AND (NOT CUFFTDX_TEST_RELEASED_PACKAGE) AND (NOT MATHDX_TEST_RELEASED_PACKAGE))
    # Targeted CUDA Architectures, see https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES
    if(CUFFTDX_TARGET_ARCHS)
        set(CUFFTDX_TARGET_ARCHS 70;75;80 CACHE
            STRING "[LEGACY] List of targeted cuFFTDx Example CUDA architectures (compute capabilities), for example \"70;75\". Can't be older than 70."
        )
        list(SORT CUFFTDX_TARGET_ARCHS)
        # Remove unsupported architectures
        list(REMOVE_ITEM CUFFTDX_TARGET_ARCHS 30;32;35;37;50;52;53;60;61;62)

        # Translate legacy option CUFFTDX_TARGET_ARCHS into CUFFTDX_CUDA_ARCHITECTURES
        set(CUFFTDX_TARGET_ARCHS_TRANSLATED)
        foreach(ARCH ${CUFFTDX_TARGET_ARCHS})
            list(APPEND CUFFTDX_TARGET_ARCHS_TRANSLATED ${ARCH}-real)
        endforeach()
        set(CUFFTDX_CUDA_ARCHITECTURES ${CUFFTDX_TARGET_ARCHS_TRANSLATED} CACHE
            STRING "List of targeted cuFFTDx CUDA architectures, for example \"70-real;75-real;80\""
        )
    else()
        set(CUFFTDX_CUDA_ARCHITECTURES 70-real;75-real;80-real CACHE
            STRING "List of targeted cuFFTDX CUDA architectures, for example \"70-real;75-real;80\""
        )
        # Remove unsupported architectures
        list(REMOVE_ITEM CUFFTDX_CUDA_ARCHITECTURES 30;32;35;37;50;52;53;60;61;62)
        list(REMOVE_ITEM CUFFTDX_CUDA_ARCHITECTURES 30-real;32-real;35-real;37-real;50-real;52-real;53-real;60-real;61-real;62-real)
        list(REMOVE_ITEM CUFFTDX_CUDA_ARCHITECTURES 30-virtual;32-virtual;35-virtual;37-virtual;50-virtual;52-virtual;53-virtual;60-virtual;61-virtual;62-virtual)
    endif()
    message(STATUS "Targeted cuFFTDx Examples CUDA Architectures: ${CUFFTDX_CUDA_ARCHITECTURES}")

    # Global CXX/CUDA flags
    if(NOT MSVC)
        set(CUFFTDX_CUDA_CXX_FLAGS "${CUFFTDX_CUDA_CXX_FLAGS} -Wall -Wextra")
    else()
        add_definitions(-D_CRT_SECURE_NO_WARNINGS)
        add_definitions(-D_CRT_NONSTDC_NO_WARNINGS)
        add_definitions(-D_SCL_SECURE_NO_WARNINGS)
        add_definitions(-DNOMINMAX)
        set(CUFFTDX_CUDA_CXX_FLAGS "${CUFFT_CUDA_CXX_FLAGS} /W3") # Warning level
        set(CUFFTDX_CUDA_CXX_FLAGS "${CUFFT_CUDA_CXX_FLAGS} /WX") # All warnings are errors
    endif()

    # Global CXX flags/options
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CUFFTDX_CUDA_CXX_FLAGS}")

    # Global CUDA CXX flags/options
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"${CUFFTDX_CUDA_CXX_FLAGS}\"")

    # Clang
    if(BUILD_CUFFTDX)
        if(CMAKE_CUDA_HOST_COMPILER MATCHES ".*clang.*")
            # clang complains about unused function in CUDA system headers
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-unused-function")
        endif()
    endif()

    # CUDA Architectures
    set(CMAKE_CUDA_ARCHITECTURES OFF)

    # Enable testing (ctest)
    enable_testing()
endif()

# ###############################################################
# add_cufftdx_example
# ###############################################################
function(add_cufftdx_example GROUP_TARGET EXAMPLE_NAME EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    get_filename_component(EXAMPLE_TARGET ${EXAMPLE_MAIN_SOURCE} NAME_WE)
    set_source_files_properties(${EXAMPLE_SOURCES} PROPERTIES LANGUAGE CUDA)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})
    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
            $<IF:$<TARGET_EXISTS:mathdx::cufftdx>,mathdx::cufftdx,cufftdx::cufftdx>
    )
    if(NOT TARGET cufftdx)
        set_target_properties(${EXAMPLE_TARGET}
            PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/example/cufftdx"
        )
    endif()
    set_target_properties(${EXAMPLE_TARGET}
        PROPERTIES
            CUDA_ARCHITECTURES "${CUFFTDX_CUDA_ARCHITECTURES}"
    )
    target_compile_options(${EXAMPLE_TARGET}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
    )
    add_test(NAME ${EXAMPLE_NAME} COMMAND ${EXAMPLE_TARGET})
    set_tests_properties(${EXAMPLE_NAME}
        PROPERTIES
            LABELS "CUFFTDX_EXAMPLE"
    )
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
endfunction()

# ###############################################################
# add_cufft_and_cufftdx_example
# ###############################################################
function(add_cufft_and_cufftdx_example GROUP_TARGET EXAMPLE_NAME EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    get_filename_component(EXAMPLE_TARGET ${EXAMPLE_MAIN_SOURCE} NAME_WE)
    set_source_files_properties(${EXAMPLE_SOURCES} PROPERTIES LANGUAGE CUDA)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})
    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
            $<IF:$<TARGET_EXISTS:mathdx::cufftdx>,mathdx::cufftdx,cufftdx::cufftdx>
    )
    if(CUFFTDX_EXAMPLES_CUFFT_CALLBACK)
        if(TARGET cufft)
            target_link_libraries(${EXAMPLE_TARGET} PRIVATE cufft_static)
        else()
            target_link_libraries(${EXAMPLE_TARGET} PRIVATE CUDA::cufft_static)
        endif()
        set_target_properties(${EXAMPLE_TARGET}
            PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
        )
        target_compile_definitions(${EXAMPLE_TARGET} PRIVATE CUFFTDX_EXAMPLES_CUFFT_CALLBACK)
    else()
        if(TARGET cufft)
            target_link_libraries(${EXAMPLE_TARGET} PRIVATE cufft)
        else()
            target_link_libraries(${EXAMPLE_TARGET} PRIVATE CUDA::cufft)
        endif()
    endif()
    if(NOT TARGET cufftdx)
        set_target_properties(${EXAMPLE_TARGET}
            PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/example/cufftdx"
        )
    endif()
    set_target_properties(${EXAMPLE_TARGET}
        PROPERTIES
            CUDA_ARCHITECTURES "${CUFFTDX_CUDA_ARCHITECTURES}"
    )
    target_compile_options(${EXAMPLE_TARGET}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
    )
    add_test(NAME ${EXAMPLE_NAME} COMMAND ${EXAMPLE_TARGET})
    set_tests_properties(${EXAMPLE_NAME}
        PROPERTIES
            LABELS "CUFFTDX_EXAMPLE"
    )
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
endfunction()

# ###############################################################
# add_cufftdx_nvrtc_example
# ###############################################################
function(add_cufftdx_nvrtc_example GROUP_TARGET EXAMPLE_NAME EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    get_filename_component(EXAMPLE_TARGET ${EXAMPLE_MAIN_SOURCE} NAME_WE)
    set_source_files_properties(${EXAMPLE_SOURCES} PROPERTIES LANGUAGE CUDA)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})
    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
            $<IF:$<TARGET_EXISTS:mathdx::cufftdx>,mathdx::cufftdx,cufftdx::cufftdx>
            CUDA::nvrtc
    )
    if(NOT TARGET cufftdx)
        set_target_properties(${EXAMPLE_TARGET}
            PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/example/cufftdx"
        )
        target_compile_definitions(${EXAMPLE_TARGET}
            PRIVATE
                CUDA_INCLUDE_DIR="${CUDAToolkit_INCLUDE_DIRS}"
                CUFFTDX_INCLUDE_DIRS="${cufftdx_INCLUDE_DIRS}"
        )
    else()
        target_compile_definitions(${EXAMPLE_TARGET}
            PRIVATE
                CUDA_INCLUDE_DIR="${CUDAToolkit_INCLUDE_DIRS}"
                CUFFTDX_INCLUDE_DIRS="${CMAKE_SOURCE_DIR}/libcufftdx/include\\\;${CMAKE_BINARY_DIR}/libcufftdx/include"
        )
    endif()
    set_target_properties(${EXAMPLE_TARGET}
        PROPERTIES
            CUDA_ARCHITECTURES "${CUFFTDX_CUDA_ARCHITECTURES}"
    )
    add_test(NAME ${EXAMPLE_NAME} COMMAND ${EXAMPLE_TARGET})
    set_tests_properties(${EXAMPLE_NAME}
        PROPERTIES
            LABELS "CUFFTDX_EXAMPLE"
    )
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
endfunction()

# ###############################################################
# cuFFTDx Examples
# ###############################################################

add_custom_target(cufftdx_examples)

# CUFFTDX_EXAMPLES_CUFFT_CALLBACK
option(CUFFTDX_EXAMPLES_CUFFT_CALLBACK "Build cuFFTDx convolution_performance example with cuFFT callback" OFF)

add_cufftdx_example(cufftdx_examples "cuFFTDx.example.simple_fft_thread" simple_fft_thread.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.simple_fft_thread_fp16" simple_fft_thread_fp16.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.simple_fft_block" simple_fft_block.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.simple_fft_block_half2" simple_fft_block_half2.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.simple_fft_block_fp16" simple_fft_block_fp16.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.simple_fft_block_r2c" simple_fft_block_r2c.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.simple_fft_block_r2c_fp16" simple_fft_block_r2c_fp16.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.simple_fft_block_c2r" simple_fft_block_c2r.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.simple_fft_block_c2r_fp16" simple_fft_block_c2r_fp16.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.simple_fft_block_shared" simple_fft_block_shared.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.simple_fft_block_std_complex" simple_fft_block_std_complex.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.simple_fft_block_cub_io" simple_fft_block_cub_io.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.convolution" convolution.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.convolution_r2c_c2r" convolution_r2c_c2r.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.block_fft_performance" block_fft_performance.cu)
add_cufftdx_example(cufftdx_examples "cuFFTDx.example.block_fft_performance_many" block_fft_performance_many.cu)
add_cufft_and_cufftdx_example(cufftdx_examples "cuFFTDx.example.convolution_performance" convolution_performance.cu)
add_cufftdx_nvrtc_example(cufftdx_examples "cuFFTDx.example.nvrtc_fft_thread" nvrtc_fft_thread.cu)
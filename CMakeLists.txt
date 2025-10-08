cmake_minimum_required(VERSION 3.16)

# Set MPI compilers
set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx)

project(stencil3d LANGUAGES CXX) 

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Set compiler flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

# Define configurable variables as CMake options
set(VLEN 8 CACHE STRING "Vector length")
set(STENCIL_SIZE 25 CACHE STRING "Stencil size")
set(HWITER 1 CACHE STRING "Use HW iteration counter")
set(NODUPS 1 CACHE STRING "No re-shuffling of dups iterations")
set(SCHED 1 CACHE STRING "Default STATIC scheduling chunk size")
set(JUNROLL 1 CACHE STRING "Unrolling level in j-direction")

# Iterator selection options (only one should be enabled)
set(PAR2D 0 CACHE STRING "Use 2D parallel iterator")
set(HILBERT 0 CACHE STRING "Use 2D Hilbert curve iterator")
set(CHECKER 0 CACHE STRING "Use 3D colored checkerboard iterator")
set(SHUFFLE 0 CACHE STRING "Shuffle Hilbert index")

# Scheduling options
set(SCHED_STATIC 0 CACHE STRING "Static OMP scheduling without chunk size")

# Data type options
set(USE_DOUBLE 0 CACHE STRING "Use double precision instead of float")
set(ISOTROPIC 0 CACHE STRING "Use isotropic stencil coefficients")

# Subtype selection
set(SUBTYPE "vec" CACHE STRING "Kernel subtype: ref|simd|vec|vec_pipe")

# Source files
set(SOURCES
    main.cpp
    iterators.cpp
    gilbert2d.cpp
    aux.cpp
    grid.cpp
)

# Header files
set(HEADERS
    config.h
    iterators.h
    stencil_2.h
    stencil_7.h
    stencil_13.h
    stencil_25.h
    stencils.h
    aux.h
    grid.h
)

# Create executable
add_executable(stencil3d ${SOURCES} ${HEADERS})

# Define preprocessor macros
target_compile_definitions(stencil3d PRIVATE
    USE_NS_API
    TELEM_REGION
    VLEN=${VLEN}
    STENCIL_SIZE=${STENCIL_SIZE}
    JUNROLL=${JUNROLL}
    SCHED=${SCHED}
    SUBTYPE=${SUBTYPE}
)

# Conditional macro definitions
if(HWITER)
    target_compile_definitions(stencil3d PRIVATE HWITER=1)
endif()

if(NODUPS)
    target_compile_definitions(stencil3d PRIVATE NODUPS=1)
endif()

if(PAR2D)
    target_compile_definitions(stencil3d PRIVATE PAR2D=1)
endif()

if(HILBERT)
    target_compile_definitions(stencil3d PRIVATE HILBERT=1)
endif()

if(CHECKER)
    target_compile_definitions(stencil3d PRIVATE CHECKER=1)
endif()

if(SHUFFLE)
    target_compile_definitions(stencil3d PRIVATE SHUFFLE=1)
endif()

if(SCHED_STATIC)
    target_compile_definitions(stencil3d PRIVATE SCHED_STATIC=1)
endif()

if(USE_DOUBLE)
    target_compile_definitions(stencil3d PRIVATE USE_DOUBLE=1)
endif()

if(ISOTROPIC)
    target_compile_definitions(stencil3d PRIVATE ISOTROPIC=1)
endif()

# Set compiler options
target_compile_options(stencil3d PRIVATE 
    ${ARCH_FLAGS}
    ${OpenMP_CXX_FLAGS}
)

# Link libraries
target_link_libraries(stencil3d 
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    nsapi
)
# Minimum required version of CMake to build this project
cmake_minimum_required(VERSION 3.10)

# Project Information
project(IKD-SWOpt VERSION 1.0
        DESCRIPTION "Planning algorithms for autonomous robots using IKD-SWOpt"
        LANGUAGES CXX)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # Disable compiler-specific extensions

# Set Build Type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Find Required Packages
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# Add Executable
add_executable(${PROJECT_NAME}
    src/IKD-SWOpt.cpp
    src/ikd_Tree.cpp
)

# Include Directories
target_include_directories(${PROJECT_NAME} 
    PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src  # Include src directory to locate lbfgs.hpp
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
)

# Link Libraries
target_link_libraries(${PROJECT_NAME} 
    PRIVATE 
        ${OpenCV_LIBS}
        Eigen3::Eigen
        Threads::Threads
)

# Compiler Options
target_compile_options(${PROJECT_NAME} PRIVATE 
    -Wall
    -Wextra
    -Wpedantic
    -g  # Debug information; remove or conditionally set for release builds
)

# Installation Rules (Optional)
# Uncomment the following lines if you intend to install the executable and headers
# install(TARGETS ${PROJECT_NAME} DESTINATION bin)
# install(DIRECTORY include/ DESTINATION include)

# Project Summary
# This CMake configuration sets up the build environment for the IKD-SWOpt project,
# implementing planning algorithms for autonomous robots. It integrates essential
# libraries such as OpenCV and Eigen3, and configures source files and include
# directories to build the main executable.

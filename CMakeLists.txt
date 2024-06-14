cmake_minimum_required(VERSION 3.12)

project(pdes
        VERSION 0.0.1
        LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(CheckTypeSize)
include(CheckSymbolExists)

# Dependencies
find_package(MPI REQUIRED)

# libpdes
#file(GLOB_RECURSE LIBPDES_SOURCES CONFIGURE_DEPENDS framework/*.cc)
#add_library(libpdes SHARED ${LIBPDES_SOURCES})
#
#target_include_directories(libpdes
#        PRIVATE
#        $<INSTALL_INTERFACE:include/pdes>
#        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
#        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
#        ${CMAKE_CURRENT_SOURCE_DIR}
#        ${PROJECT_SOURCE_DIR}
#        PUBLIC
#        ${LUA_INCLUDE_DIR}
#        ${PETSC_INCLUDE_DIR}
#)
#target_link_libraries(libpdes PRIVATE MPI::MPI_CXX)

# pdes executable
add_executable(pdes "main.cc")
target_include_directories(pdes
        PRIVATE
        $<INSTALL_INTERFACE:include/pdes>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
)
target_link_libraries(pdes
        PRIVATE
        #        libpdes
        MPI::MPI_CXX
)



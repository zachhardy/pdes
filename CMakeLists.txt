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
find_package(GTest REQUIRED)
message(STATUS "GTest found at: ${GTest_DIR}")


# libpdes
file(GLOB_RECURSE LIBPDES_SOURCES CONFIGURE_DEPENDS framework/*.cc modules/*.cc)
add_library(libpdes SHARED ${LIBPDES_SOURCES})

target_include_directories(libpdes
        PRIVATE
        $<INSTALL_INTERFACE:include/pdes>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
        ${PROJECT_SOURCE_DIR}
        PUBLIC
        ${LUA_INCLUDE_DIR}
        ${PETSC_INCLUDE_DIR}
)
target_link_libraries(libpdes PRIVATE MPI::MPI_CXX)

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
        libpdes
        MPI::MPI_CXX
)

# GTest unit testing
enable_testing()
include(GoogleTest)

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS test/*.cc)

if (TEST_SOURCES)
    add_executable(pdes_test ${TEST_SOURCES})

    target_link_libraries(pdes_test
            PRIVATE
            libpdes
            GTest::gtest
            GTest::gtest_main
    )

    target_include_directories(pdes_test
            PRIVATE
            ${PROJECT_SOURCE_DIR}
    )

    gtest_discover_tests(pdes_test)
endif ()

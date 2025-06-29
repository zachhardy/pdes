file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
)

add_executable(pdes_test ${TEST_SOURCES})

target_link_libraries(pdes_test
        PRIVATE
        libpdes
        GTest::gtest
        GTest::gtest_main
)

target_include_directories(pdes_test PRIVATE ${PROJECT_SOURCE_DIR})

catch_discover_tests(pdes_test)

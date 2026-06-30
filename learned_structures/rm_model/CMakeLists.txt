cmake_minimum_required(VERSION 3.16)
project(rm_model LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT DEFINED LEARNEDHASH_BUILD_TOOLS)
  option(LEARNEDHASH_BUILD_TOOLS "Build command-line tools" ON)
endif()

find_package(Threads REQUIRED)

file(GLOB RM_MODEL_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/rm_model/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/rm_model/models/*.cpp
)

add_library(rm_model STATIC ${RM_MODEL_SOURCES})
target_include_directories(rm_model PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(rm_model PUBLIC Threads::Threads)
if (UNIX AND NOT APPLE)
  target_link_libraries(rm_model PUBLIC dl)
endif()
target_compile_options(rm_model PRIVATE -Wall -Wextra -Wpedantic)

if (LEARNEDHASH_BUILD_TOOLS)
  add_executable(rm_model_learner
    tools/learner/main.cpp
  )

  target_include_directories(rm_model_learner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  target_link_libraries(rm_model_learner PRIVATE rm_model)
  target_compile_options(rm_model_learner PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(rm_model_inferencer
    tools/inferencer/main.cpp
  )

  target_include_directories(rm_model_inferencer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  target_link_libraries(rm_model_inferencer PRIVATE rm_model)
  target_compile_options(rm_model_inferencer PRIVATE -Wall -Wextra -Wpedantic)

endif()

if ((DEFINED LEARNEDHASH_BUILD_TESTS AND LEARNEDHASH_BUILD_TESTS) OR BUILD_TESTING)
  enable_testing()
  add_executable(rm_model_tests
    tests/rm_model_tests.cpp
  )
  target_link_libraries(rm_model_tests PRIVATE rm_model)
  target_compile_options(rm_model_tests PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME rm_model_tests COMMAND rm_model_tests)
  set_tests_properties(rm_model_tests PROPERTIES LABELS "rm_model;runtime;smoke")

  if (TARGET rm_model_learner AND TARGET rm_model_inferencer)
    add_executable(rm_model_tool_tests
      tests/rm_model_tool_tests.cpp
    )
    target_link_libraries(rm_model_tool_tests PRIVATE rm_model)
    target_compile_definitions(rm_model_tool_tests PRIVATE
      RM_MODEL_LEARNER_PATH="$<TARGET_FILE:rm_model_learner>"
      RM_MODEL_INFERENCER_PATH="$<TARGET_FILE:rm_model_inferencer>"
    )
    add_dependencies(rm_model_tool_tests rm_model_learner rm_model_inferencer)
    target_compile_options(rm_model_tool_tests PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME rm_model_tool_tests COMMAND rm_model_tool_tests)
    set_tests_properties(rm_model_tool_tests PROPERTIES LABELS "rm_model;runtime;codegen;cli;smoke")
  endif()
endif()

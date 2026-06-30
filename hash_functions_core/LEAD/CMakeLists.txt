cmake_minimum_required(VERSION 3.16)

set(LEAD_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

if (NOT DEFINED LEARNEDHASH_BUILD_TOOLS)
  option(LEARNEDHASH_BUILD_TOOLS "Build command-line tools" ON)
endif()

add_library(lead_core
  src/mapping.cpp
  src/manifest.cpp
  src/rm_loader.cpp
  src/hasher.cpp
  src/trainer.cpp
)

target_include_directories(lead_core
  PUBLIC
    $<BUILD_INTERFACE:${LEAD_ROOT}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(lead_core PUBLIC rm_model)

target_compile_options(lead_core PRIVATE -Wall -Wextra -Wpedantic)

if (LEARNEDHASH_BUILD_TOOLS)
  add_executable(lead_hashgen
    tools/hashgen/main.cpp
  )

  target_include_directories(lead_hashgen PRIVATE
    ${LEAD_ROOT}/include
  )

  target_link_libraries(lead_hashgen PRIVATE lead_core rm_model)

  target_compile_options(lead_hashgen PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(lead_hasher
    tools/hasher/main.cpp
  )

  target_include_directories(lead_hasher PRIVATE
    ${LEAD_ROOT}/include
  )

  target_link_libraries(lead_hasher PRIVATE lead_core rm_model)

  target_compile_options(lead_hasher PRIVATE -Wall -Wextra -Wpedantic)
endif()

if (LEARNEDHASH_BUILD_TESTS)
  add_executable(lead_runtime_tests
    tests/lead_runtime_tests.cpp
  )

  target_include_directories(lead_runtime_tests PRIVATE
    ${LEAD_ROOT}/include
    ${LEAD_ROOT}/../../learned_structures/rm_model/tests
  )

  target_link_libraries(lead_runtime_tests PRIVATE lead_core rm_model)
  target_compile_options(lead_runtime_tests PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME lead_runtime_tests COMMAND lead_runtime_tests)
  set_tests_properties(lead_runtime_tests PROPERTIES LABELS "lead;runtime;codegen;smoke")

  if (TARGET lead_hashgen AND TARGET lead_hasher)
    add_executable(lead_tool_tests
      tests/lead_tool_tests.cpp
    )
    target_include_directories(lead_tool_tests PRIVATE
      ${LEAD_ROOT}/../../learned_structures/rm_model/tests
    )
    target_link_libraries(lead_tool_tests PRIVATE rm_model)
    target_compile_definitions(lead_tool_tests PRIVATE
      LEAD_HASHGEN_PATH="$<TARGET_FILE:lead_hashgen>"
      LEAD_HASHER_PATH="$<TARGET_FILE:lead_hasher>"
    )
    add_dependencies(lead_tool_tests lead_hashgen lead_hasher)
    target_compile_options(lead_tool_tests PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME lead_tool_tests COMMAND lead_tool_tests)
    set_tests_properties(lead_tool_tests PROPERTIES LABELS "lead;cli;smoke")
  endif()
endif()

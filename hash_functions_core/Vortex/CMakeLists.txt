cmake_minimum_required(VERSION 3.16)

set(VORTEX_V1_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
if (NOT DEFINED LEARNEDHASH_BUILD_TOOLS)
  option(LEARNEDHASH_BUILD_TOOLS "Build command-line tools" ON)
endif()
if (NOT DEFINED LEARNEDHASH_BUILD_TESTS)
  option(LEARNEDHASH_BUILD_TESTS "Build tests" ON)
endif()
if (NOT DEFINED LEARNEDHASH_BUILD_VORTEX_FAISS_COMPONENTS)
  option(LEARNEDHASH_BUILD_VORTEX_FAISS_COMPONENTS
    "Build FAISS-backed Vortex training, selector, and evaluation components" ON)
endif()
find_package(Threads REQUIRED)

add_library(vortex_v1_runtime
  ${VORTEX_V1_ROOT}/src/model.cpp
  ${VORTEX_V1_ROOT}/src/codegen.cpp
  ${VORTEX_V1_ROOT}/src/codegen_centroid_index.cpp
  ${VORTEX_V1_ROOT}/src/codegen_emit_header.cpp
  ${VORTEX_V1_ROOT}/src/codegen_emit_package.cpp
  ${VORTEX_V1_ROOT}/src/codegen_emit_source.cpp
  ${VORTEX_V1_ROOT}/src/codegen_loader.cpp
  ${VORTEX_V1_ROOT}/src/codegen_manifest.cpp
  ${VORTEX_V1_ROOT}/src/codegen_utils.cpp
)

target_include_directories(vortex_v1_runtime
  PUBLIC
    $<BUILD_INTERFACE:${VORTEX_V1_ROOT}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(vortex_v1_runtime
  PUBLIC
    rm_model
)

target_compile_options(vortex_v1_runtime PRIVATE -Wall -Wextra -Wpedantic)

if (LEARNEDHASH_BUILD_VORTEX_FAISS_COMPONENTS)
  add_library(vortex_v1_core
    ${VORTEX_V1_ROOT}/src/nsw_csr.cpp
    ${VORTEX_V1_ROOT}/src/model_selector.cpp
    ${VORTEX_V1_ROOT}/src/selector_candidates.cpp
    ${VORTEX_V1_ROOT}/src/selector_contexts.cpp
    ${VORTEX_V1_ROOT}/src/selector_eval_base_rank_cache.cpp
    ${VORTEX_V1_ROOT}/src/selector_eval_dataset.cpp
    ${VORTEX_V1_ROOT}/src/selector_eval_hash64.cpp
    ${VORTEX_V1_ROOT}/src/selector_eval_nearest_cache.cpp
    ${VORTEX_V1_ROOT}/src/selector_eval_utils.cpp
    ${VORTEX_V1_ROOT}/src/selector_evaluator.cpp
    ${VORTEX_V1_ROOT}/src/selector_options.cpp
    ${VORTEX_V1_ROOT}/src/selector_orchestration.cpp
    ${VORTEX_V1_ROOT}/src/selector_report.cpp
    ${VORTEX_V1_ROOT}/src/selector_report_candidate.cpp
    ${VORTEX_V1_ROOT}/src/selector_report_common.cpp
    ${VORTEX_V1_ROOT}/src/selector_report_diagnostics.cpp
    ${VORTEX_V1_ROOT}/src/selector_report_selection.cpp
    ${VORTEX_V1_ROOT}/src/selector_scheduler.cpp
    ${VORTEX_V1_ROOT}/src/selector_scoring.cpp
    ${VORTEX_V1_ROOT}/src/selector_search.cpp
    ${VORTEX_V1_ROOT}/src/selector_utils.cpp
    ${VORTEX_V1_ROOT}/src/training_base.cpp
    ${VORTEX_V1_ROOT}/src/training_cdf.cpp
    ${VORTEX_V1_ROOT}/src/training_common.cpp
    ${VORTEX_V1_ROOT}/src/training.cpp
    ${VORTEX_V1_ROOT}/src/training_nsw.cpp
    ${VORTEX_V1_ROOT}/src/training_order.cpp
    ${VORTEX_V1_ROOT}/src/training_ranges.cpp
    ${VORTEX_V1_ROOT}/src/vector_io.cpp
  )

  target_include_directories(vortex_v1_core
    PUBLIC
      $<BUILD_INTERFACE:${VORTEX_V1_ROOT}/include>
      $<INSTALL_INTERFACE:include>
  )

  target_link_libraries(vortex_v1_core
    PUBLIC
      vortex_v1_runtime
      faiss
      Threads::Threads
  )

  target_compile_options(vortex_v1_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if (LEARNEDHASH_BUILD_TOOLS AND TARGET vortex_v1_core)
  add_executable(vortex_v1_cli
    ${VORTEX_V1_ROOT}/tools/main.cpp
  )

  target_link_libraries(vortex_v1_cli PRIVATE vortex_v1_core)

  target_compile_options(vortex_v1_cli PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(vortex_eval
    ${VORTEX_V1_ROOT}/tools/vortex_eval.cpp
  )

  target_link_libraries(vortex_eval PRIVATE vortex_v1_core)

  target_compile_options(vortex_eval PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(hnsw_eval
    ${VORTEX_V1_ROOT}/tools/hnsw_eval.cpp
  )

  target_link_libraries(hnsw_eval PRIVATE vortex_v1_core)

  target_compile_options(hnsw_eval PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(vortex_model_selector
    ${VORTEX_V1_ROOT}/tools/vortex_model_selector.cpp
    ${VORTEX_V1_ROOT}/tools/vortex_model_selector_report.cpp
  )

  target_link_libraries(vortex_model_selector PRIVATE vortex_v1_core)

  target_compile_options(vortex_model_selector PRIVATE -Wall -Wextra -Wpedantic)
endif()

if (LEARNEDHASH_BUILD_TESTS)
  function(add_vortex_v1_runtime_test target source labels)
    add_executable(${target}
      ${VORTEX_V1_ROOT}/tests/${source}
    )
    target_link_libraries(${target} PRIVATE vortex_v1_runtime)
    target_include_directories(${target} PRIVATE ${VORTEX_V1_ROOT}/tests)
    target_compile_definitions(${target} PRIVATE
      VORTEX_V1_PROJECT_ROOT="${PROJECT_SOURCE_DIR}"
    )
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES LABELS "${labels}")
  endfunction()

  function(add_vortex_v1_test target source labels)
    add_executable(${target}
      ${VORTEX_V1_ROOT}/tests/${source}
    )
    target_link_libraries(${target} PRIVATE vortex_v1_core)
    target_include_directories(${target} PRIVATE ${VORTEX_V1_ROOT}/tests)
    target_compile_definitions(${target} PRIVATE
      VORTEX_V1_PROJECT_ROOT="${PROJECT_SOURCE_DIR}"
    )
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES LABELS "${labels}")
  endfunction()

  add_vortex_v1_runtime_test(vortex_v1_codegen_tests
    vortex_codegen_tests.cpp
    "vortex;codegen;smoke")
  if (TARGET vortex_v1_core)
    add_vortex_v1_test(vortex_v1_runtime_tests
      vortex_runtime_tests.cpp
      "vortex;runtime;smoke")
    add_vortex_v1_test(vortex_v1_training_tests
      vortex_training_tests.cpp
      "vortex;training;smoke")
    add_vortex_v1_test(vortex_v1_selector_policy_tests
      vortex_selector_policy_tests.cpp
      "vortex;selector;policy")
    add_vortex_v1_test(vortex_v1_selector_cache_tests
      vortex_selector_cache_tests.cpp
      "vortex;selector;cache")
    add_vortex_v1_test(vortex_v1_selector_tests
      vortex_selector_tests.cpp
      "vortex;selector;integration;smoke")
    add_vortex_v1_test(vortex_v1_selector_calibration_tests
      vortex_selector_calibration_tests.cpp
      "vortex;selector;integration;calibration")
    add_vortex_v1_test(vortex_v1_selector_skeleton_tests
      vortex_selector_skeleton_tests.cpp
      "vortex;selector;integration;skeleton")
    add_vortex_v1_test(vortex_v1_selector_recall_tests
      vortex_selector_recall_tests.cpp
      "vortex;selector;integration;recall")
    # Selector tests exercise FAISS/OpenMP-heavy training paths and have shown
    # nondeterministic crashes when run concurrently with other generated-load tests.
    set_tests_properties(
      vortex_v1_selector_policy_tests
      vortex_v1_selector_cache_tests
      vortex_v1_selector_tests
      vortex_v1_selector_calibration_tests
      vortex_v1_selector_skeleton_tests
      vortex_v1_selector_recall_tests
      PROPERTIES RUN_SERIAL TRUE)
    add_vortex_v1_test(vortex_v1_cli_tests
      vortex_cli_tests.cpp
      "vortex;cli;selector;smoke")

    if (TARGET vortex_v1_cli)
      add_dependencies(vortex_v1_cli_tests vortex_v1_cli)
      target_compile_definitions(vortex_v1_cli_tests PRIVATE
        VORTEX_V1_CLI_PATH="$<TARGET_FILE:vortex_v1_cli>"
      )
    endif()
    if (TARGET vortex_model_selector)
      add_dependencies(vortex_v1_cli_tests vortex_model_selector)
      target_compile_definitions(vortex_v1_cli_tests PRIVATE
        VORTEX_MODEL_SELECTOR_PATH="$<TARGET_FILE:vortex_model_selector>"
      )
    endif()
  endif()
endif()

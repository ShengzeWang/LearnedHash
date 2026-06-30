cmake_minimum_required(VERSION 3.16)
project(LearnedHash VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LEARNEDHASH_BUILD_TESTS "Build tests" ON)
option(LEARNEDHASH_BUILD_TOOLS "Build command-line tools" ON)
option(LEARNEDHASH_BUILD_VORTEX_FAISS_COMPONENTS
  "Build FAISS-backed Vortex training, selector, and evaluation components" ON)
set(LEARNEDHASH_FAISS_PROVIDER "bundled" CACHE STRING
  "FAISS provider: bundled, system, or auto")
set(LEARNEDHASH_FAISS_PROVIDER_CHOICES bundled system auto)
set_property(CACHE LEARNEDHASH_FAISS_PROVIDER PROPERTY STRINGS bundled system auto)
string(TOLOWER "${LEARNEDHASH_FAISS_PROVIDER}" LEARNEDHASH_FAISS_PROVIDER)
if (NOT "${LEARNEDHASH_FAISS_PROVIDER}" IN_LIST LEARNEDHASH_FAISS_PROVIDER_CHOICES)
  message(FATAL_ERROR
    "LEARNEDHASH_FAISS_PROVIDER must be one of: bundled, system, auto")
endif()
set(LEARNEDHASH_FAISS_PROVIDER_RESOLVED "" CACHE INTERNAL "Resolved FAISS provider" FORCE)

function(learnedhash_use_bundled_faiss)
  add_subdirectory(include/faiss)
  target_include_directories(faiss SYSTEM PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
  set(LEARNEDHASH_FAISS_PROVIDER_RESOLVED "bundled" CACHE INTERNAL
    "Resolved FAISS provider" FORCE)
endfunction()

function(learnedhash_use_system_faiss required)
  if (${required})
    find_package(faiss CONFIG REQUIRED)
  else()
    find_package(faiss CONFIG QUIET)
  endif()

  if (TARGET faiss)
    set(LEARNEDHASH_FAISS_PROVIDER_RESOLVED "system" CACHE INTERNAL
      "Resolved FAISS provider" FORCE)
  elseif (TARGET faiss::faiss)
    add_library(faiss INTERFACE IMPORTED GLOBAL)
    target_link_libraries(faiss INTERFACE faiss::faiss)
    set(LEARNEDHASH_FAISS_PROVIDER_RESOLVED "system" CACHE INTERNAL
      "Resolved FAISS provider" FORCE)
  elseif (${required})
    message(FATAL_ERROR
      "System FAISS package was found, but it did not expose a 'faiss' or "
      "'faiss::faiss' CMake target. Use -DLEARNEDHASH_FAISS_PROVIDER=bundled "
      "or provide a FAISS package config that exports one of those targets.")
  endif()
endfunction()

function(learnedhash_configure_bundled_faiss_targets)
  # Keep third-party FAISS warnings from obscuring warnings in project code.
  foreach(_faiss_target faiss faiss_avx2 faiss_avx512 faiss_avx512_spr faiss_sve)
    if (TARGET ${_faiss_target})
      if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(${_faiss_target} PRIVATE
          $<$<COMPILE_LANGUAGE:CXX>:-w>)
      elseif (MSVC)
        target_compile_options(${_faiss_target} PRIVATE
          $<$<COMPILE_LANGUAGE:CXX>:/w>)
      endif()

      if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${_faiss_target} PRIVATE
          $<$<COMPILE_LANGUAGE:CXX>:-Wno-c++20-extensions>
          $<$<COMPILE_LANGUAGE:CXX>:-Wno-inconsistent-missing-override>
          "$<$<COMPILE_LANGUAGE:CXX>:-Wno-#warnings>")
      endif()
    endif()
  endforeach()

  find_package(OpenMP)
  if (OpenMP_CXX_FOUND)
    target_link_libraries(faiss PUBLIC OpenMP::OpenMP_CXX)
  endif()
endfunction()

if (LEARNEDHASH_BUILD_TESTS)
  enable_testing()
  add_executable(learnedhash_common_tests
    tests/learnedhash_common_tests.cpp
  )
  target_include_directories(learnedhash_common_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/learned_structures/rm_model/include
    ${PROJECT_SOURCE_DIR}/learned_structures/rm_model/tests
  )
  target_compile_options(learnedhash_common_tests PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME learnedhash_common_tests COMMAND learnedhash_common_tests)
  set_tests_properties(learnedhash_common_tests PROPERTIES LABELS "common;runtime;smoke")
endif()

# Help FindOpenMP on AppleClang when libomp is installed via Homebrew.
if (APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if (NOT OpenMP_CXX_FLAGS)
    foreach(_omp_prefix "/opt/homebrew/opt/libomp" "/usr/local/opt/libomp")
      if (EXISTS "${_omp_prefix}/include/omp.h" AND EXISTS "${_omp_prefix}/lib/libomp.dylib")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${_omp_prefix}/include" CACHE STRING "OpenMP C flags" FORCE)
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${_omp_prefix}/include" CACHE STRING "OpenMP CXX flags" FORCE)
        set(OpenMP_C_LIB_NAMES "omp" CACHE STRING "OpenMP C libraries" FORCE)
        set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "OpenMP CXX libraries" FORCE)
        set(OpenMP_omp_LIBRARY "${_omp_prefix}/lib/libomp.dylib" CACHE FILEPATH "OpenMP runtime library" FORCE)
        break()
      endif()
    endforeach()
  endif()
endif()

add_subdirectory(learned_structures/rm_model)
if (LEARNEDHASH_BUILD_TESTS)
  target_link_libraries(learnedhash_common_tests PRIVATE rm_model)
endif()
add_subdirectory(hash_functions_core/LEAD)

if (LEARNEDHASH_BUILD_VORTEX_FAISS_COMPONENTS)
  if (LEARNEDHASH_FAISS_PROVIDER STREQUAL "bundled")
    learnedhash_use_bundled_faiss()
  elseif (LEARNEDHASH_FAISS_PROVIDER STREQUAL "system")
    learnedhash_use_system_faiss(TRUE)
  else()
    learnedhash_use_system_faiss(FALSE)
    if (NOT LEARNEDHASH_FAISS_PROVIDER_RESOLVED STREQUAL "system")
      learnedhash_use_bundled_faiss()
    endif()
  endif()

  if (LEARNEDHASH_FAISS_PROVIDER_RESOLVED STREQUAL "bundled")
    learnedhash_configure_bundled_faiss_targets()
  endif()
  message(STATUS "LearnedHash FAISS provider: ${LEARNEDHASH_FAISS_PROVIDER_RESOLVED}")
else()
  set(LEARNEDHASH_FAISS_PROVIDER_RESOLVED "disabled" CACHE INTERNAL
    "Resolved FAISS provider" FORCE)
  message(STATUS
    "LearnedHash FAISS provider: disabled (LEARNEDHASH_BUILD_VORTEX_FAISS_COMPONENTS=OFF)")
endif()

add_subdirectory(hash_functions_core/Vortex)

set(LEARNEDHASH_INSTALL_TARGETS
  rm_model
  lead_core
  vortex_v1_runtime
)
if (TARGET vortex_v1_core)
  list(APPEND LEARNEDHASH_INSTALL_TARGETS vortex_v1_core)
endif()

install(TARGETS ${LEARNEDHASH_INSTALL_TARGETS}
  EXPORT LearnedHashTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

set(LEARNEDHASH_TOOL_INSTALL_TARGETS)
foreach(_learnedhash_tool
    rm_model_learner
    rm_model_inferencer
    lead_hashgen
    lead_hasher
    vortex_v1_cli
    vortex_model_selector
    vortex_eval
    hnsw_eval)
  if (TARGET ${_learnedhash_tool})
    list(APPEND LEARNEDHASH_TOOL_INSTALL_TARGETS ${_learnedhash_tool})
  endif()
endforeach()

if (LEARNEDHASH_TOOL_INSTALL_TARGETS)
  install(TARGETS ${LEARNEDHASH_TOOL_INSTALL_TARGETS}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()

install(DIRECTORY
  learned_structures/rm_model/include/
  hash_functions_core/LEAD/include/
  hash_functions_core/Vortex/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

set(LEARNEDHASH_CONFIG_INSTALL_DIR
  ${CMAKE_INSTALL_LIBDIR}/cmake/LearnedHash)

if (TARGET vortex_v1_core)
  set(LEARNEDHASH_CONFIG_FAISS_DEPENDENCY
"find_dependency(faiss CONFIG)

if (NOT TARGET faiss AND TARGET faiss::faiss)
  add_library(faiss INTERFACE IMPORTED)
  target_link_libraries(faiss INTERFACE faiss::faiss)
endif()")
else()
  set(LEARNEDHASH_CONFIG_FAISS_DEPENDENCY "")
endif()

configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/LearnedHashConfig.cmake.in
  ${PROJECT_BINARY_DIR}/cmake/LearnedHashConfig.cmake
  INSTALL_DESTINATION ${LEARNEDHASH_CONFIG_INSTALL_DIR}
)

write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/cmake/LearnedHashConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

install(FILES
  ${PROJECT_BINARY_DIR}/cmake/LearnedHashConfig.cmake
  ${PROJECT_BINARY_DIR}/cmake/LearnedHashConfigVersion.cmake
  DESTINATION ${LEARNEDHASH_CONFIG_INSTALL_DIR}
)

install(EXPORT LearnedHashTargets
  NAMESPACE LearnedHash::
  DESTINATION ${LEARNEDHASH_CONFIG_INSTALL_DIR}
)

find_package(Python3 COMPONENTS Interpreter QUIET)
if (Python3_Interpreter_FOUND)
  add_custom_target(learnedhash_check_install_export
    COMMAND ${Python3_EXECUTABLE}
      ${PROJECT_SOURCE_DIR}/tools/check_install_export.py
      --build-dir ${PROJECT_BINARY_DIR}
      --config $<CONFIG>
    DEPENDS ${LEARNEDHASH_INSTALL_TARGETS} ${LEARNEDHASH_TOOL_INSTALL_TARGETS}
    USES_TERMINAL
    VERBATIM
    COMMENT "Checking installed LearnedHash CMake package with external consumers"
  )
endif()

################################################################################
# ompMC - An OpenMP parallel implementation for Monte Carlo particle transport
# simulations
#
# Cross platform build definition for the ompMC user codes.
#
#   omc_dosxyz   command line executable
#   omc_matrad   MATLAB MEX file, built only when a MATLAB installation is found
#
# Typical use:
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --config Release
#
# The MATLAB installation to build omc_matrad against can be selected with
# -DMatlab_ROOT_DIR=/path/to/MATLAB/R20XXy .
################################################################################

cmake_minimum_required(VERSION 3.20)

project(ompMC
    VERSION 1.0.0
    DESCRIPTION "An OpenMP parallel implementation for Monte Carlo particle transport simulations"
    HOMEPAGE_URL "https://github.com/e0404/ompMC"
    LANGUAGES C)

include(CheckIncludeFile)
include(CheckSymbolExists)
include(CheckLibraryExists)
include(CheckCCompilerFlag)

################################################################################
# Options
################################################################################

option(OMPMC_BUILD_DOSXYZ "Build the omc_dosxyz command line user code" ON)
option(OMPMC_WITH_OPENMP "Build with OpenMP multi threading support" ON)
option(OMPMC_NATIVE_TUNING
    "Tune the generated code for the building machine (-mtune=native)" OFF)

set(OMPMC_BUILD_MATRAD_MEX "AUTO" CACHE STRING
    "Build the omc_matrad MEX file: ON (fail if MATLAB is missing), OFF, or AUTO")
set_property(CACHE OMPMC_BUILD_MATRAD_MEX PROPERTY STRINGS AUTO ON OFF)

# Single configuration generators get a sensible default build type.
get_property(_ompmc_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _ompmc_multi_config AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY
        STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)          # strtok_r and friends live behind _GNU_SOURCE

# The core objects end up inside the MEX shared object, so they must be PIC.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Collect everything that is meant to be used in one place.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
foreach(_cfg IN LISTS CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER "${_cfg}" _cfg_upper)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_cfg_upper} "${CMAKE_BINARY_DIR}/bin")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${_cfg_upper} "${CMAKE_BINARY_DIR}/bin")
endforeach()

################################################################################
# Platform feature checks
################################################################################

# MSVC ships no <getopt.h>; omc_dosxyz then uses the bundled replacement in
# src/compat. Everywhere else the system header is used unchanged.
check_include_file("getopt.h" OMPMC_HAVE_GETOPT_H)
if(OMPMC_HAVE_GETOPT_H)
    check_symbol_exists(getopt_long "getopt.h" OMPMC_HAVE_GETOPT_LONG)
endif()

check_library_exists(m sqrt "" OMPMC_HAVE_LIBM)

################################################################################
# OpenMP
################################################################################

set(OMPMC_OPENMP_ENABLED FALSE)

if(OMPMC_WITH_OPENMP)
    # Apple's clang needs a separately installed libomp. Give CMake a hand by
    # pointing it at the Homebrew keg, which is where it usually lives.
    if(APPLE AND NOT DEFINED OpenMP_ROOT AND NOT DEFINED ENV{OpenMP_ROOT})
        find_program(OMPMC_BREW_EXECUTABLE brew)
        if(OMPMC_BREW_EXECUTABLE)
            execute_process(
                COMMAND "${OMPMC_BREW_EXECUTABLE}" --prefix libomp
                OUTPUT_VARIABLE _ompmc_libomp_prefix
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                RESULT_VARIABLE _ompmc_brew_result)
            if(_ompmc_brew_result EQUAL 0 AND IS_DIRECTORY "${_ompmc_libomp_prefix}")
                set(OpenMP_ROOT "${_ompmc_libomp_prefix}")
                message(STATUS "Using Homebrew libomp from ${OpenMP_ROOT}")
            endif()
        endif()
    endif()

    find_package(OpenMP COMPONENTS C)

    if(OpenMP_C_FOUND)
        set(OMPMC_OPENMP_ENABLED TRUE)
    else()
        message(WARNING
            "OpenMP was requested but no working OpenMP compiler was found. "
            "ompMC will be built for serial execution. On macOS install libomp "
            "(brew install libomp); on Linux install the OpenMP runtime of your "
            "compiler.")
    endif()
endif()

################################################################################
# Core library shared by all user codes
################################################################################

add_library(ompmc_core STATIC
    src/ompmc.c
    src/omc_utilities.c
    src/omc_random.c)

target_include_directories(ompmc_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")

if(MSVC)
    # The code base uses the classic C string and stdio functions throughout.
    target_compile_definitions(ompmc_core PUBLIC
        _CRT_SECURE_NO_WARNINGS
        _USE_MATH_DEFINES)
    target_compile_options(ompmc_core PRIVATE /W3)
else()
    target_compile_options(ompmc_core PRIVATE -Wall)
endif()

if(OMPMC_HAVE_LIBM)
    target_link_libraries(ompmc_core PUBLIC m)
endif()

if(OMPMC_OPENMP_ENABLED)
    target_link_libraries(ompmc_core PUBLIC OpenMP::OpenMP_C)
endif()

if(OMPMC_NATIVE_TUNING)
    check_c_compiler_flag("-mtune=native" OMPMC_HAVE_MTUNE_NATIVE)
    if(OMPMC_HAVE_MTUNE_NATIVE)
        target_compile_options(ompmc_core PUBLIC "-mtune=native")
    else()
        message(STATUS "-mtune=native is not supported by this compiler, ignoring")
    endif()
endif()

################################################################################
# omc_dosxyz - command line user code
################################################################################

if(OMPMC_BUILD_DOSXYZ)
    add_executable(omc_dosxyz ucodes/omc_dosxyz/omc_dosxyz.c)
    target_link_libraries(omc_dosxyz PRIVATE ompmc_core)

    if(NOT OMPMC_HAVE_GETOPT_LONG)
        message(STATUS "No system getopt_long(), using the bundled replacement")
        target_sources(omc_dosxyz PRIVATE src/compat/getopt.c)
        target_include_directories(omc_dosxyz PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src/compat")
    endif()
endif()

################################################################################
# omc_matrad - MATLAB MEX user code
################################################################################

set(OMPMC_MEX_ENABLED FALSE)

if(NOT OMPMC_BUILD_MATRAD_MEX STREQUAL "OFF")
    if(OMPMC_BUILD_MATRAD_MEX STREQUAL "AUTO")
        find_package(Matlab QUIET COMPONENTS MX_LIBRARY)
        if(NOT Matlab_FOUND)
            message(STATUS
                "No MATLAB installation found, skipping the omc_matrad MEX file. "
                "Set -DMatlab_ROOT_DIR=<matlabroot> to point the build at one.")
        endif()
    else()
        find_package(Matlab REQUIRED COMPONENTS MX_LIBRARY)
    endif()

    if(Matlab_FOUND)
        set(OMPMC_MEX_ENABLED TRUE)
    endif()
endif()

if(OMPMC_MEX_ENABLED)
    matlab_add_mex(
        NAME omc_matrad
        SRC ucodes/omc_matrad/omc_matrad.c
        OUTPUT_NAME omc_matrad
        LINK_TO ompmc_core
        MODULE)

    # matlab_add_mex() does not go through the usual output directory
    # variables for every generator, so pin the location explicitly.
    set_target_properties(omc_matrad PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    foreach(_cfg IN LISTS CMAKE_CONFIGURATION_TYPES)
        string(TOUPPER "${_cfg}" _cfg_upper)
        set_target_properties(omc_matrad PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_${_cfg_upper} "${CMAKE_BINARY_DIR}/bin"
            RUNTIME_OUTPUT_DIRECTORY_${_cfg_upper} "${CMAKE_BINARY_DIR}/bin")
    endforeach()

    if(MSVC)
        target_compile_definitions(omc_matrad PRIVATE
            _CRT_SECURE_NO_WARNINGS
            _USE_MATH_DEFINES)
    endif()
endif()

################################################################################
# Installation
################################################################################

include(GNUInstallDirs)

if(OMPMC_BUILD_DOSXYZ)
    install(TARGETS omc_dosxyz RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()

if(OMPMC_MEX_ENABLED)
    install(TARGETS omc_matrad
        LIBRARY DESTINATION "${CMAKE_INSTALL_BINDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()

################################################################################
# Configuration summary
################################################################################

message(STATUS "")
message(STATUS "ompMC ${PROJECT_VERSION} configuration summary")
message(STATUS "  System           : ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  C compiler       : ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
if(_ompmc_multi_config)
    message(STATUS "  Build type       : ${CMAKE_CONFIGURATION_TYPES} (multi config)")
else()
    message(STATUS "  Build type       : ${CMAKE_BUILD_TYPE}")
endif()
message(STATUS "  OpenMP           : ${OMPMC_OPENMP_ENABLED}")
message(STATUS "  omc_dosxyz       : ${OMPMC_BUILD_DOSXYZ}")
message(STATUS "  omc_matrad (mex) : ${OMPMC_MEX_ENABLED}")
if(OMPMC_MEX_ENABLED)
    message(STATUS "    MATLAB root    : ${Matlab_ROOT_DIR}")
    message(STATUS "    MATLAB version : ${Matlab_VERSION_STRING}")
    message(STATUS "    MEX extension  : ${Matlab_MEX_EXTENSION}")
endif()
message(STATUS "")

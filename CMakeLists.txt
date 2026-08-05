################################################################################
# ompMC - An OpenMP parallel implementation for Monte Carlo particle transport
# simulations
#
# Cross platform build definition for the ompMC user codes.
#
#   omc_dosxyz   command line executable
#   omc_matrad   MATLAB MEX file, built only when a MATLAB installation is found
#   omc_python   Python extension module, built only with -DOMPMC_BUILD_PYTHON=ON
#                which is what pip sets when it builds the wheel
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
    VERSION 0.2.0
    DESCRIPTION "An OpenMP parallel implementation for Monte Carlo particle transport simulations"
    HOMEPAGE_URL "https://github.com/e0404/ompMC"
    LANGUAGES C)

include(CheckIncludeFile)
include(CheckSymbolExists)
include(CheckLibraryExists)
include(CheckCCompilerFlag)
include(CheckIPOSupported)

################################################################################
# Options
################################################################################

option(OMPMC_BUILD_DOSXYZ "Build the omc_dosxyz command line user code" ON)
option(OMPMC_WITH_OPENMP "Build with OpenMP multi threading support" ON)
option(OMPMC_NATIVE_TUNING
    "Tune the generated code for the building machine (-mtune=native)" OFF)
option(OMPMC_LTO
    "Enable link time optimization on optimized builds" ON)
option(OMPMC_BUILD_TESTS "Build the unit tests and register them with CTest" ON)
option(OMPMC_WITH_OPENLIBM
    "Fetch openlibm and resolve the libm calls from it instead of the \
toolchain's math library. Worthwhile with MinGW GCC, whose bundled software \
log/exp are several times slower; pointless with MSVC or glibc." OFF)

set(OMPMC_BUILD_MATRAD_MEX "AUTO" CACHE STRING
    "Build the omc_matrad MEX file: ON (fail if MATLAB is missing), OFF, or AUTO")
set_property(CACHE OMPMC_BUILD_MATRAD_MEX PROPERTY STRINGS AUTO ON OFF)

set(OMPMC_BUILD_MATRAD_OCT "AUTO" CACHE STRING
    "Build the omc_matrad MEX file for GNU Octave: ON (fail if Octave is \
missing), OFF, or AUTO")
set_property(CACHE OMPMC_BUILD_MATRAD_OCT PROPERTY STRINGS AUTO ON OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

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
# Version header
################################################################################

# omc_version.h is generated rather than committed so that the project()
# VERSION above stays the single source of truth for the OMPMC_VERSION_*
# macros the user codes print.
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omc_version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/omc_version.h"
    @ONLY)

################################################################################
# Core library shared by all user codes
################################################################################

add_library(ompmc_core STATIC
    src/ompmc.c
    src/omc_utilities.c
    src/omc_random.c
    src/omc_score.c
    src/omc_host.c
    src/omc_geom.c
    src/omc_spectrum.c
    src/omc_source_beamlet.c
    src/omc_engine_dij.c
    src/omc_engine_cube.c
    src/omc_engine_forward.c)

target_include_directories(ompmc_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_BINARY_DIR}/generated")

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

# OpenMP is attached in two steps. The compile options and the omp.h include
# path go on the core library, so every user code compiles its pragmas the same
# way. The runtime *library* is linked per user code instead, because the MEX
# file must not always bring its own -- see the omc_matrad section below.
set(OMPMC_OPENMP_RUNTIME "")

if(OMPMC_OPENMP_ENABLED)
    get_target_property(_ompmc_omp_options OpenMP::OpenMP_C INTERFACE_COMPILE_OPTIONS)
    if(_ompmc_omp_options)
        target_compile_options(ompmc_core PUBLIC ${_ompmc_omp_options})
    endif()

    get_target_property(_ompmc_omp_includes OpenMP::OpenMP_C INTERFACE_INCLUDE_DIRECTORIES)
    if(_ompmc_omp_includes)
        target_include_directories(ompmc_core SYSTEM PUBLIC ${_ompmc_omp_includes})
    endif()

    set(OMPMC_OPENMP_RUNTIME OpenMP::OpenMP_C)
endif()

if(OMPMC_NATIVE_TUNING)
    check_c_compiler_flag("-mtune=native" OMPMC_HAVE_MTUNE_NATIVE)
    if(OMPMC_HAVE_MTUNE_NATIVE)
        target_compile_options(ompmc_core PUBLIC "-mtune=native")
    else()
        message(STATUS "-mtune=native is not supported by this compiler, ignoring")
    endif()
endif()

# The transport code calls sqrt() constantly (uphi21/uphi32, mscat,
# spinRejection, msdist). By default GCC and Clang must assume the caller may
# inspect errno afterwards, so they cannot reduce those calls to a bare sqrt
# instruction and emit a domain check plus a libm call path around each one.
# ompMC never reads errno, so allow the plain instruction. This is not part of
# -ffast-math: no reassociation, no loss of IEEE semantics for the values
# themselves, and results are unchanged.
check_c_compiler_flag("-fno-math-errno" OMPMC_HAVE_NO_MATH_ERRNO)
if(OMPMC_HAVE_NO_MATH_ERRNO)
    target_compile_options(ompmc_core PUBLIC "-fno-math-errno")
endif()

# ausgab(), howfar() and hownear() are defined by the user code but called from
# the transport loop in ompmc.c, so without link time optimization every
# invocation is an opaque cross translation unit call -- hownear() once per
# electron step, howfar() at least that often. LTO lets them inline.
set(OMPMC_LTO_ENABLED FALSE)

if(OMPMC_LTO)
    check_ipo_supported(RESULT OMPMC_HAVE_IPO OUTPUT _ompmc_ipo_error LANGUAGES C)
    if(OMPMC_HAVE_IPO)
        set(OMPMC_LTO_ENABLED TRUE)

        # Plain binutils ar/ranlib cannot read GCC's LTO objects without being
        # handed the plugin, and warn "plugin needed to handle lto object" when
        # they meet one. The gcc-ar/gcc-ranlib wrappers pass it themselves, so
        # use those whenever the toolchain ships them.
        if(CMAKE_C_COMPILER_AR AND CMAKE_C_COMPILER_RANLIB)
            set(CMAKE_AR "${CMAKE_C_COMPILER_AR}")
            set(CMAKE_RANLIB "${CMAKE_C_COMPILER_RANLIB}")
        endif()
    else()
        message(STATUS
            "Link time optimization is not available with this toolchain, "
            "ignoring: ${_ompmc_ipo_error}")
    endif()
endif()

# Only on optimized configurations; LTO on a Debug build costs build time
# without buying anything, and makes the result harder to step through.
function(ompmc_enable_lto target)
    if(OMPMC_LTO_ENABLED)
        set_property(TARGET ${target} PROPERTY
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
        set_property(TARGET ${target} PROPERTY
            INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
        set_property(TARGET ${target} PROPERTY
            INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL TRUE)
    endif()
endfunction()

ompmc_enable_lto(ompmc_core)

################################################################################
# openlibm
################################################################################

# MinGW GCC does not use the fast UCRT math routines: log(), exp() and friends
# come from its own bundled software implementations, measured about 5x slower
# here -- and the transport samples -log(rng) for every photon flight segment.
# Linking openlibm in front of the toolchain's math library recovers most of
# that. The archive is linked PUBLIC on the core so that every user code
# resolves the symbols the same way, before the implicit toolchain libraries.
if(OMPMC_WITH_OPENLIBM)
    # openlibm's own CMake build asks for this
    if(CMAKE_VERSION VERSION_LESS 3.25)
        message(FATAL_ERROR
            "OMPMC_WITH_OPENLIBM needs CMake 3.25 or newer to configure "
            "openlibm; found ${CMAKE_VERSION}. Update CMake or disable the "
            "option.")
    endif()

    include(FetchContent)
    FetchContent_Declare(openlibm
        GIT_REPOSITORY https://github.com/JuliaMath/openlibm.git
        GIT_TAG        v0.8.7
        GIT_SHALLOW    TRUE)

    # Build openlibm as a static archive: the binaries and the MEX file must
    # not depend on shipping an extra DLL, and a static archive is what lets
    # the linker satisfy log/exp ahead of the toolchain's own math library.
    set(_ompmc_saved_shared ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)
    FetchContent_MakeAvailable(openlibm)
    set(BUILD_SHARED_LIBS ${_ompmc_saved_shared})

    target_link_libraries(ompmc_core PUBLIC openlibm)
    message(STATUS "Using openlibm for the math library calls")
endif()

################################################################################
# Coverage instrumentation
################################################################################

option(OMPMC_COVERAGE
    "Instrument ompmc_core and every executable linked against it with \
gcov/llvm-cov line coverage (GCC or Clang only). Meant for a dedicated CI job \
building Debug, not for anything the timing depends on." OFF)

if(OMPMC_COVERAGE)
    if(NOT (CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang"))
        message(FATAL_ERROR
            "OMPMC_COVERAGE needs GCC or Clang, found ${CMAKE_C_COMPILER_ID}")
    endif()

    # PUBLIC on the core so that omc_dosxyz and the test executables, which
    # all link ompmc_core, get the same instrumentation for their own
    # translation units too, and so that the final link pulls libgcov in.
    # -O0 overrides whatever optimization the chosen build type would
    # otherwise add: inlining and dead code elimination move and merge lines,
    # which skews line coverage away from what the source actually looks
    # like.
    target_compile_options(ompmc_core PUBLIC --coverage -O0)
    target_link_options(ompmc_core PUBLIC --coverage)
endif()

################################################################################
# omc_dosxyz - command line user code
################################################################################

if(OMPMC_BUILD_DOSXYZ)
    add_executable(omc_dosxyz ucodes/omc_dosxyz/omc_dosxyz.c)
    target_link_libraries(omc_dosxyz PRIVATE ompmc_core ${OMPMC_OPENMP_RUNTIME})
    ompmc_enable_lto(omc_dosxyz)

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

    if(WIN32 AND NOT MSVC)
        # See the comment in the .def file; without it the MEX file can lose
        # its mexFunction export as soon as openlibm joins the link.
        target_sources(omc_matrad PRIVATE ucodes/omc_matrad/omc_matrad.def)
    endif()

    set(OMPMC_MEX_OPENMP_SOURCE "none")

    if(OMPMC_OPENMP_ENABLED)
        if(APPLE)
            # How the MEX file gets its OpenMP runtime on macOS depends on
            # whether MATLAB brings one into the process, which differs between
            # the Apple Silicon and the Intel builds of MATLAB. Ask the
            # installation rather than guessing from the architecture.
            string(REGEX REPLACE "^mex" "" _ompmc_matlab_arch "${Matlab_MEX_EXTENSION}")
            file(GLOB _ompmc_matlab_openmp
                "${Matlab_ROOT_DIR}/bin/${_ompmc_matlab_arch}/libomp.dylib"
                "${Matlab_ROOT_DIR}/bin/${_ompmc_matlab_arch}/libiomp5.dylib")

            if(_ompmc_matlab_openmp)
                # Apple Silicon MATLAB ships bin/maca64/libomp.dylib, the very
                # LLVM runtime clang targets. A MEX file that linked a second
                # copy -- Homebrew's libomp.dylib, say -- would put two copies of
                # the same runtime in the process; they share symbol names, so
                # worker threads started by one end up in the other's code
                # operating on foreign thread state. The observed symptoms are
                # "OMP: Error #179 pthread_mutex_init failed" followed by a
                # segmentation fault in __kmp_suspend_64.
                #
                # So link no runtime at all here. The OpenMP symbols stay
                # undefined and dyld binds them, when MATLAB loads the MEX file,
                # to the copy MATLAB has already brought in. That also keeps the
                # MEX file free of an absolute path into a particular MATLAB or
                # Homebrew installation.
                target_link_options(omc_matrad PRIVATE "SHELL:-undefined dynamic_lookup")
                set(OMPMC_MEX_OPENMP_SOURCE "loaded by MATLAB, bound at load time")
            else()
                # Intel MATLAB ships no OpenMP runtime a MEX file can bind to:
                # bin/maci64 holds neither libomp.dylib nor libiomp5.dylib, only
                # the libmwompwrapper shim, which defines none of the __kmpc_*
                # entry points. Binding at load time therefore fails outright
                # with an unresolved __kmpc_dispatch_deinit.
                #
                # Give the MEX file a private runtime instead, linked from the
                # static libomp.a. Nothing can collide with it: the MEX file
                # exports only mexFunction (matlab_add_mex passes an
                # -exported_symbols_list), so the runtime stays invisible to the
                # rest of the process, and its internal calls are bound at link
                # time rather than going through the flat namespace.
                set(_ompmc_libomp_hints "")
                if(OpenMP_omp_LIBRARY)
                    get_filename_component(_ompmc_libomp_dir "${OpenMP_omp_LIBRARY}" DIRECTORY)
                    list(APPEND _ompmc_libomp_hints "${_ompmc_libomp_dir}")
                endif()
                if(OpenMP_ROOT)
                    list(APPEND _ompmc_libomp_hints "${OpenMP_ROOT}/lib")
                endif()

                find_library(OMPMC_LIBOMP_STATIC
                    NAMES libomp.a
                    HINTS ${_ompmc_libomp_hints}
                    DOC "Static LLVM OpenMP runtime to link into the MEX file")

                if(NOT OMPMC_LIBOMP_STATIC)
                    message(FATAL_ERROR
                        "This MATLAB (${Matlab_ROOT_DIR}) ships no OpenMP runtime "
                        "for the MEX file to use, so a static libomp.a has to be "
                        "linked into it, and none was found. Install one with "
                        "'brew install libomp', point the build at it with "
                        "-DOMPMC_LIBOMP_STATIC=<path to libomp.a>, or build "
                        "without the MEX file (-DOMPMC_BUILD_MATRAD_MEX=OFF) or "
                        "without OpenMP (-DOMPMC_WITH_OPENMP=OFF).")
                endif()

                # libomp is implemented in C++, so it needs the C++ runtime even
                # though every ompMC translation unit is C.
                target_link_libraries(omc_matrad "${OMPMC_LIBOMP_STATIC}" c++)
                set(OMPMC_MEX_OPENMP_SOURCE "static ${OMPMC_LIBOMP_STATIC}")
            endif()
        elseif(WIN32 AND CMAKE_C_COMPILER_ID STREQUAL "Clang")
            # MATLAB on Windows ships the Intel OpenMP runtime, libiomp5md.dll,
            # with an import library right next to it -- and LLVM's libomp is a
            # fork of that very runtime, so the __kmpc_* calls clang emits bind
            # to it directly. Link the MEX file against MATLAB's copy instead
            # of bringing libomp.dll into the process: two KMP-family runtimes
            # in one process trip Intel's duplicate-runtime check ("OMP: Error
            # #15") or, worse, interleave their thread pools.
            #
            # This stays clang-only. GCC emits GOMP_* calls and the Intel
            # runtime on Windows exports no GOMP compatibility layer, so the
            # GCC MEX file keeps its own libgomp; the two coexist because they
            # share no symbols.
            # Unlike on macOS the extension (mexw64) does not name the bin
            # subdirectory (win64), and win64 is the only Windows platform
            # MATLAB still supports.
            set(_ompmc_iomp5 "${Matlab_ROOT_DIR}/bin/win64/libiomp5md.lib")

            if(EXISTS "${_ompmc_iomp5}")
                target_link_libraries(omc_matrad "${_ompmc_iomp5}")
                # Newer clang emits one runtime call MATLAB's older Intel
                # runtime does not export; see the shim for why a no-op is the
                # faithful translation.
                target_sources(omc_matrad PRIVATE
                    ucodes/omc_matrad/omc_kmp_compat.c)
                set(OMPMC_MEX_OPENMP_SOURCE "MATLAB's libiomp5md, bound at load time")
            else()
                target_link_libraries(omc_matrad ${OMPMC_OPENMP_RUNTIME})
                set(OMPMC_MEX_OPENMP_SOURCE "linked shared runtime")
            endif()
        else()
            target_link_libraries(omc_matrad ${OMPMC_OPENMP_RUNTIME})
            set(OMPMC_MEX_OPENMP_SOURCE "linked shared runtime")
        endif()
    endif()

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

    ompmc_enable_lto(omc_matrad)
endif()

################################################################################
# omc_matrad - GNU Octave MEX user code
################################################################################

# The same omc_matrad.c, built a second time against Octave's MEX API. Octave
# implements the API MATLAB's mex.h declares, so the source needs no changes;
# only the headers, the library and the file extension differ.
set(OMPMC_OCT_ENABLED FALSE)

if(NOT OMPMC_BUILD_MATRAD_OCT STREQUAL "OFF")
    # Octave's libraries on Windows are MinGW import libraries, which MSVC
    # cannot link. Rather than fail the MSVC build, say so and move on.
    if(MSVC)
        if(OMPMC_BUILD_MATRAD_OCT STREQUAL "ON")
            message(FATAL_ERROR
                "The Octave MEX file cannot be built with MSVC: Octave ships "
                "MinGW import libraries. Build it with MinGW GCC or clang, or "
                "set -DOMPMC_BUILD_MATRAD_OCT=OFF.")
        endif()
        message(STATUS
            "Skipping the Octave MEX file: MSVC cannot link Octave's MinGW "
            "import libraries.")
    elseif(OMPMC_BUILD_MATRAD_OCT STREQUAL "AUTO")
        find_package(Octave QUIET)
        if(NOT Octave_FOUND)
            message(STATUS
                "No Octave installation found, skipping the Octave MEX file. "
                "Set -DOctave_ROOT=<octave prefix> to point the build at one.")
        endif()
    else()
        find_package(Octave REQUIRED)
    endif()

    if(Octave_FOUND)
        set(OMPMC_OCT_ENABLED TRUE)
    endif()
endif()

if(OMPMC_OCT_ENABLED)
    # A plain MODULE library rather than matlab_add_mex()'s equivalent: Octave
    # loads the file by name, so only the extension has to be right.
    add_library(omc_matrad_oct MODULE ucodes/omc_matrad/omc_matrad.c)

    # Octave 10 refuses to load a .mex that does not declare which liboctmex
    # ABI it was built against; releases without liboctmex predate the check
    # and need no stub. FindOctave leaves the version empty in that case.
    if(Octave_MEX_SOVERSION)
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/ucodes/omc_matrad/omc_mex_soversion.c.in"
            "${CMAKE_CURRENT_BINARY_DIR}/generated/omc_mex_soversion.c"
            @ONLY)
        target_sources(omc_matrad_oct PRIVATE
            "${CMAKE_CURRENT_BINARY_DIR}/generated/omc_mex_soversion.c")
    endif()

    target_link_libraries(omc_matrad_oct PRIVATE
        ompmc_core Octave::mex ${OMPMC_OPENMP_RUNTIME})

    # On ELF and Mach-O the MEX API symbols are left undefined and bound to
    # the Octave process that loads the file, which is what mkoctfile does
    # too. Linux allows that by default for a shared object; macOS has to be
    # told explicitly.
    if(APPLE)
        target_link_options(omc_matrad_oct PRIVATE
            "SHELL:-undefined dynamic_lookup")
    endif()

    # mkoctfile compiles MEX sources with -DMEX_DEBUG, which is what makes
    # Octave's mex.h declare the MATLAB compatible entry points rather than
    # its own internal ones.
    target_compile_definitions(omc_matrad_oct PRIVATE MEX_DEBUG)

    # Octave reports MEX errors by throwing a C++ exception from inside
    # liboctmex, so mexErrMsgIdAndTxt() never returns and the stack is
    # unwound through this file's frames. Without -fexceptions GCC and Clang
    # emit C translation units with no unwind tables, and that unwind hits
    # std::terminate instead, taking the whole Octave session down. mkoctfile
    # passes the same flag for the same reason.
    check_c_compiler_flag("-fexceptions" OMPMC_HAVE_FEXCEPTIONS)
    if(OMPMC_HAVE_FEXCEPTIONS)
        target_compile_options(omc_matrad_oct PRIVATE "-fexceptions")
    endif()

    # Octave dlopen()s the file and looks up mexFunction by name. The MinGW
    # linker exports nothing from a module by default once anything else is
    # exported explicitly, so ask for the same blanket export mkoctfile uses.
    if(WIN32)
        target_link_options(omc_matrad_oct PRIVATE "-Wl,--export-all-symbols")
    endif()

    # Octave's .mex is a plain shared object with a fixed extension and no
    # "lib" prefix, on every platform.
    set_target_properties(omc_matrad_oct PROPERTIES
        OUTPUT_NAME omc_matrad
        PREFIX ""
        SUFFIX ".mex"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    foreach(_cfg IN LISTS CMAKE_CONFIGURATION_TYPES)
        string(TOUPPER "${_cfg}" _cfg_upper)
        set_target_properties(omc_matrad_oct PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_${_cfg_upper} "${CMAKE_BINARY_DIR}/bin"
            RUNTIME_OUTPUT_DIRECTORY_${_cfg_upper} "${CMAKE_BINARY_DIR}/bin")
    endforeach()

    ompmc_enable_lto(omc_matrad_oct)
endif()

################################################################################
# omc_python - Python extension user code, built as the module _ompmc
################################################################################

# Built through scikit-build-core (pip install .), which sets this option and
# points CMake at the interpreter to build for. Building it from a plain CMake
# invocation works too, as long as nanobind is importable.
option(OMPMC_BUILD_PYTHON "Build the _ompmc Python extension module" OFF)

if(OMPMC_BUILD_PYTHON)
    # The binding is the only C++ in the project, so the language is enabled
    # here rather than in project(): a plain C build must not need a C++
    # compiler just because this option exists.
    enable_language(CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # SKBUILD_SABI_COMPONENT is set to Development.SABIModule by
    # scikit-build-core when wheel.py-api asks for a limited API build, and to
    # the empty string otherwise -- including on the Pythons that are older
    # than the requested ABI, where it silently falls back to a version
    # specific wheel. nanobind's STABLE_ABI below is a no-op without that
    # component, so the two settings have to travel together.
    find_package(Python 3.9 REQUIRED
        COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})

    # nanobind ships its CMake config inside the installed package
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_VARIABLE _ompmc_nanobind_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _ompmc_nanobind_result)

    if(NOT _ompmc_nanobind_result EQUAL 0)
        message(FATAL_ERROR
            "OMPMC_BUILD_PYTHON needs nanobind importable by "
            "${Python_EXECUTABLE}. Install it with 'pip install nanobind'.")
    endif()

    list(APPEND CMAKE_PREFIX_PATH "${_ompmc_nanobind_dir}")
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(_ompmc STABLE_ABI NB_STATIC ucodes/omc_python/omc_python.cpp)
    target_link_libraries(_ompmc PRIVATE ompmc_core ${OMPMC_OPENMP_RUNTIME})
    ompmc_enable_lto(_ompmc)

    if(MSVC)
        target_compile_definitions(_ompmc PRIVATE
            _CRT_SECURE_NO_WARNINGS
            _USE_MATH_DEFINES)
    endif()

    install(TARGETS _ompmc LIBRARY DESTINATION ompmc)

    # The data the calculation reads is found relative to the package
    install(DIRECTORY data pegs4 spectra DESTINATION ompmc)
endif()

################################################################################
# Tests
################################################################################

if(OMPMC_BUILD_TESTS)
    enable_testing()

    add_executable(test_ompmc tests/test_ompmc.c)
    target_link_libraries(test_ompmc PRIVATE ompmc_core ${OMPMC_OPENMP_RUNTIME})
    ompmc_enable_lto(test_ompmc)

    if(MSVC)
        target_compile_definitions(test_ompmc PRIVATE
            _CRT_SECURE_NO_WARNINGS
            _USE_MATH_DEFINES)
    endif()

    add_test(NAME unit COMMAND test_ompmc)

    # Needs the PEGS and cross section data, so it runs from the repository
    # root like the smoke test below.
    add_executable(test_media_data tests/test_media_data.c)
    target_link_libraries(test_media_data
        PRIVATE ompmc_core ${OMPMC_OPENMP_RUNTIME})
    ompmc_enable_lto(test_media_data)

    if(MSVC)
        target_compile_definitions(test_media_data PRIVATE
            _CRT_SECURE_NO_WARNINGS
            _USE_MATH_DEFINES)
    endif()

    add_test(NAME media COMMAND test_media_data
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

    # The smoke test drives the whole transport chain, so it has to run from
    # the repository root where the data, pegs4 and phantoms folders live.
    if(OMPMC_BUILD_DOSXYZ)
        add_test(NAME smoke
            COMMAND omc_dosxyz -i ucodes/omc_dosxyz/smoke_test -o smoke_test
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()

    # Drives the Octave .mex through the same test_omc_matrad_mex.m the MATLAB
    # build uses: loading the file, a real dose calculation, the progress
    # callback and releasing the file afterwards. Octave exits nonzero when
    # the script raises, which is what CTest keys off. The MATLAB MEX file has
    # no equivalent test here because starting MATLAB needs the licensing that
    # the CI's matlab-actions steps set up.
    if(OMPMC_OCT_ENABLED AND Octave_EXECUTABLE)
        add_test(NAME octave_mex
            COMMAND "${Octave_EXECUTABLE}" --no-gui --quiet --eval
                "addpath('${CMAKE_BINARY_DIR}/bin'); addpath('${CMAKE_CURRENT_SOURCE_DIR}/ucodes/omc_matrad'); test_omc_matrad_mex"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
endif()

################################################################################
# Documentation
################################################################################

# Convenience target for people who drive everything through CMake; not part
# of ALL and not needed for Read the Docs, which runs sphinx-build directly
# (see .readthedocs.yaml) so that building the docs never needs this
# configure step or a C++ toolchain. See docs/README.md for the equivalent
# plain sphinx-build command.
find_program(SPHINX_BUILD_EXECUTABLE sphinx-build)
if(SPHINX_BUILD_EXECUTABLE)
    add_custom_target(docs
        COMMAND "${SPHINX_BUILD_EXECUTABLE}" -b html
                "${CMAKE_CURRENT_SOURCE_DIR}/docs"
                "${CMAKE_CURRENT_BINARY_DIR}/docs/html"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Building Sphinx documentation")
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
message(STATUS "  LTO              : ${OMPMC_LTO_ENABLED}")
message(STATUS "  openlibm         : ${OMPMC_WITH_OPENLIBM}")
message(STATUS "  Tests            : ${OMPMC_BUILD_TESTS}")
message(STATUS "  Coverage         : ${OMPMC_COVERAGE}")
message(STATUS "  omc_dosxyz       : ${OMPMC_BUILD_DOSXYZ}")
message(STATUS "  omc_matrad (mex) : ${OMPMC_MEX_ENABLED}")
if(OMPMC_MEX_ENABLED AND OMPMC_OPENMP_ENABLED)
    message(STATUS "    OpenMP runtime : ${OMPMC_MEX_OPENMP_SOURCE}")
endif()
if(OMPMC_MEX_ENABLED)
    message(STATUS "    MATLAB root    : ${Matlab_ROOT_DIR}")
    message(STATUS "    MATLAB version : ${Matlab_VERSION_STRING}")
    message(STATUS "    MEX extension  : ${Matlab_MEX_EXTENSION}")
endif()
message(STATUS "  omc_matrad (oct) : ${OMPMC_OCT_ENABLED}")
if(OMPMC_OCT_ENABLED)
    message(STATUS "    Octave version : ${Octave_VERSION}")
    if(Octave_MEX_SOVERSION)
        message(STATUS "    MEX soversion  : ${Octave_MEX_SOVERSION}")
    else()
        message(STATUS "    MEX soversion  : not checked by this Octave")
    endif()
endif()
message(STATUS "")

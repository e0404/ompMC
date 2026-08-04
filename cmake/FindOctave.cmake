################################################################################
# FindOctave - locate a GNU Octave installation to build .mex files against
#
# Octave ships no CMake package, but it does ship octave-config, which reports
# the installation layout. Everything here is derived from it rather than
# guessed from paths.
#
# Result variables:
#
#   Octave_FOUND               TRUE when a usable installation was found
#   Octave_VERSION             e.g. 10.3.0
#   Octave_INCLUDE_DIRS        directories holding mex.h and octave-config.h
#   Octave_MEX_LIBRARY         liboctmex, when this Octave has one (see below)
#   Octave_MEX_SOVERSION       ABI number Octave checks when loading a .mex,
#                              empty when this Octave does not check one
#   Octave_EXECUTABLE          octave-cli, used to run the tests
#   Octave_CONFIG_EXECUTABLE   octave-config
#   Octave_MKOCTFILE_EXECUTABLE  mkoctfile, not used to build but a good check
#
# Imported target:
#
#   Octave::mex                include directories, and the Octave libraries
#                              on the platforms that have to link them
#
# Set Octave_ROOT to point the search at a particular installation.
#
# Two things differ between Octave releases and platforms, and both are
# detected rather than assumed:
#
#   * Octave 10 moved the MEX entry points out of liboctinterp into their own
#     liboctmex. Octave 9 and older have no such library, and mkoctfile links
#     -loctinterp -loctave there instead of -loctmex.
#   * A .mex is loaded into a process that already provides those symbols, so
#     on ELF and Mach-O nothing needs to be linked at all -- mkoctfile links
#     no Octave library there. Windows PE cannot leave symbols undefined, so
#     there the libraries genuinely have to be on the link line.
################################################################################

# The Windows installer lays the tree out as <root>/mingw64/bin, which no
# default search path covers, so offer the usual install locations as a last
# resort. PATHS rather than HINTS, so that an Octave the user actually put on
# PATH still wins over one that merely happens to be installed.
file(GLOB _octave_windows_paths
    "C:/Program Files/GNU Octave/Octave-*/mingw64/bin"
    "C:/Octave/Octave-*/mingw64/bin")

# Newest first, and NATURAL so that Octave-9 sorts below Octave-10 rather than
# above it the way a plain string comparison would have it.
if(_octave_windows_paths)
    list(SORT _octave_windows_paths COMPARE NATURAL ORDER DESCENDING)
endif()

find_program(Octave_CONFIG_EXECUTABLE
    NAMES octave-config
    PATHS ${_octave_windows_paths}
    DOC "Octave's octave-config, which reports the installation layout")

# Queries one of octave-config's properties. Returns an empty string when the
# property is unknown, which is how the caller detects an unusable install.
function(_octave_config_property var property)
    execute_process(
        COMMAND "${Octave_CONFIG_EXECUTABLE}" -p "${property}"
        OUTPUT_VARIABLE _value
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _result)

    if(NOT _result EQUAL 0)
        set(_value "")
    endif()

    # octave-config reports native Windows paths with backslashes, which CMake
    # would otherwise treat as escape sequences.
    file(TO_CMAKE_PATH "${_value}" _value)
    set(${var} "${_value}" PARENT_SCOPE)
endfunction()

if(Octave_CONFIG_EXECUTABLE)
    _octave_config_property(Octave_VERSION VERSION)
    _octave_config_property(_octave_include_dir OCTINCLUDEDIR)
    _octave_config_property(_octave_lib_dir OCTLIBDIR)

    # OCTINCLUDEDIR is <prefix>/include/octave-<ver>/octave. mex.h includes
    # octave-config.h from the same directory, but other Octave headers are
    # reached as <octave/...>, so the parent has to be on the include path too.
    get_filename_component(_octave_include_parent "${_octave_include_dir}"
        DIRECTORY)

    if(_octave_include_dir AND EXISTS "${_octave_include_dir}/mex.h")
        set(Octave_INCLUDE_DIRS
            "${_octave_include_dir}" "${_octave_include_parent}")
    else()
        set(Octave_INCLUDE_DIRS "")
    endif()

    # Optional: absent on Octave 9 and older, and not needed on ELF/Mach-O
    # even when present.
    find_library(Octave_MEX_LIBRARY
        NAMES octmex
        HINTS "${_octave_lib_dir}"
        DOC "Octave's liboctmex, linked only where PE requires it")

    # Where liboctmex does not exist the same entry points are still in
    # liboctinterp. mkoctfile links liboctave alongside it there, so do the
    # same rather than rely on liboctinterp pulling it in.
    find_library(Octave_INTERP_LIBRARY
        NAMES octinterp
        HINTS "${_octave_lib_dir}"
        DOC "Octave's liboctinterp, the pre-Octave 10 home of the MEX API")

    find_library(Octave_CORE_LIBRARY
        NAMES octave
        HINTS "${_octave_lib_dir}"
        DOC "Octave's liboctave, linked next to liboctinterp before Octave 10")

    # What actually gets linked, on the platforms that need a library at all.
    # This mirrors what mkoctfile --mex puts on its link line: -loctmex from
    # Octave 10 on, -loctinterp -loctave before that.
    if(Octave_MEX_LIBRARY)
        set(Octave_LINK_LIBRARIES "${Octave_MEX_LIBRARY}")
    elseif(Octave_INTERP_LIBRARY AND Octave_CORE_LIBRARY)
        set(Octave_LINK_LIBRARIES
            "${Octave_INTERP_LIBRARY}" "${Octave_CORE_LIBRARY}")
    else()
        set(Octave_LINK_LIBRARIES "")
    endif()

    # Both of these have to come from the installation octave-config was found
    # in, not from whichever one turns up first all over again -- with several
    # Octaves installed side by side that would happily pair the headers of one
    # with the interpreter of another.
    get_filename_component(_octave_bin_dir "${Octave_CONFIG_EXECUTABLE}"
        DIRECTORY)

    find_program(Octave_EXECUTABLE
        NAMES octave-cli octave
        HINTS "${_octave_bin_dir}"
        NO_DEFAULT_PATH
        DOC "Octave interpreter used to run the .mex tests")

    find_program(Octave_MKOCTFILE_EXECUTABLE
        NAMES mkoctfile
        HINTS "${_octave_bin_dir}"
        NO_DEFAULT_PATH
        DOC "Octave's mkoctfile")
endif()

################################################################################
# MEX ABI version
################################################################################

# Octave 10 refuses to load a .mex file that does not define
# __octave_mex_soversion__ ("No SOVERSION found in .mex file function"), and
# mkoctfile supplies it by generating a one-line stub. Building the .mex from
# CMake instead means finding the number ourselves, and octave-config does not
# report it. Releases without liboctmex predate the check and want no stub at
# all, so the two are looked up together.
#
# The libtool archive next to liboctmex names the runtime library it belongs
# to (dlname='../../../bin/liboctmex-1.dll'), which is exactly the number
# wanted. Debian strips .la files, so fall back to the runtime library's own
# file name, which carries the same number on every platform:
#
#   liboctmex-1.dll   liboctmex.so.1   liboctmex.1.dylib
if(Octave_MEX_LIBRARY AND NOT DEFINED Octave_MEX_SOVERSION)
    get_filename_component(_octave_mex_lib_dir "${Octave_MEX_LIBRARY}" DIRECTORY)

    set(_octave_soversion "")

    if(EXISTS "${_octave_mex_lib_dir}/liboctmex.la")
        file(STRINGS "${_octave_mex_lib_dir}/liboctmex.la" _octave_la_dlname
            REGEX "^dlname=")
        if(_octave_la_dlname AND
           _octave_la_dlname MATCHES "liboctmex[-.]([0-9]+)")
            set(_octave_soversion "${CMAKE_MATCH_1}")
        endif()
    endif()

    if(NOT _octave_soversion)
        # The versioned runtime library sits next to the import library on
        # Unix, but in <prefix>/bin on Windows, where the .dll is separate
        # from the .dll.a.
        get_filename_component(_octave_prefix_bin "${Octave_CONFIG_EXECUTABLE}"
            DIRECTORY)
        file(GLOB _octave_mex_runtimes
            "${_octave_mex_lib_dir}/liboctmex.so.*"
            "${_octave_mex_lib_dir}/liboctmex.*.dylib"
            "${_octave_mex_lib_dir}/liboctmex-*.dll"
            "${_octave_prefix_bin}/liboctmex-*.dll")

        foreach(_candidate IN LISTS _octave_mex_runtimes)
            get_filename_component(_candidate_name "${_candidate}" NAME)
            if(_candidate_name MATCHES "liboctmex[-.]([0-9]+)")
                set(_octave_soversion "${CMAKE_MATCH_1}")
                break()
            endif()
        endforeach()
    endif()

    set(Octave_MEX_SOVERSION "${_octave_soversion}" CACHE STRING
        "ABI version Octave checks when loading a .mex file")
    mark_as_advanced(Octave_MEX_SOVERSION)
endif()

if(NOT DEFINED Octave_MEX_SOVERSION)
    set(Octave_MEX_SOVERSION "")
endif()

################################################################################

set(_octave_required_vars Octave_CONFIG_EXECUTABLE Octave_INCLUDE_DIRS)

# Only where the linker insists on resolving everything up front. Elsewhere no
# Octave library is linked at all, so having none to find is the normal case.
if(WIN32)
    list(APPEND _octave_required_vars Octave_LINK_LIBRARIES)
endif()

# An Octave new enough to ship liboctmex is also new enough to check the ABI
# number, so failing to work it out means the .mex would build and then be
# refused at load time.
if(Octave_MEX_LIBRARY)
    list(APPEND _octave_required_vars Octave_MEX_SOVERSION)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Octave
    REQUIRED_VARS ${_octave_required_vars}
    VERSION_VAR Octave_VERSION)

if(Octave_FOUND AND NOT TARGET Octave::mex)
    # INTERFACE rather than UNKNOWN IMPORTED: there is nothing to point
    # IMPORTED_LOCATION at when no library gets linked.
    add_library(Octave::mex INTERFACE IMPORTED)
    set_target_properties(Octave::mex PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${Octave_INCLUDE_DIRS}")

    if(WIN32 AND Octave_LINK_LIBRARIES)
        set_target_properties(Octave::mex PROPERTIES
            INTERFACE_LINK_LIBRARIES "${Octave_LINK_LIBRARIES}")
    endif()
endif()

mark_as_advanced(
    Octave_CONFIG_EXECUTABLE
    Octave_MKOCTFILE_EXECUTABLE
    Octave_EXECUTABLE
    Octave_MEX_LIBRARY
    Octave_INTERP_LIBRARY)

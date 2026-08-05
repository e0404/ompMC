/******************************************************************************
 ompMC - An OpenMP parallel implementation for Monte Carlo particle transport
 simulations

 Copyright (C) 2018 Edgardo Doerner (edoerner@fis.puc.cl)


 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
*****************************************************************************/

/*!
 @file
 omc_host - How shared ompMC code talks back to whoever is embedding it.

 Code that is meant to be used from more than one host -- a command line
 program, a MEX file, a Python extension -- cannot call printf() or
 mexErrMsgIdAndTxt() directly. It calls omcLog() and omcFail() instead, and the
 host installs the sinks that give those meaning.

 @warning THREADING: both sinks are called on the master thread only, never
 from inside an OpenMP parallel region. Hosts may therefore call back into a
 runtime that has no business being entered from a worker thread -- MATLAB's
 mexPrintf, or a Python callable under the GIL. Diagnostics that transport
 code emits from inside a parallel region keep going straight to stdout
 instead, which is why ompmc.c still uses printf() there.
*****************************************************************************/

#ifndef OMC_HOST_H
#define OMC_HOST_H

/*! @cond OMC_INTERNAL
 omcFail() never comes back, and saying so lets callers end a function with
 it the way they used to end one with exit(), without the compiler asking for
 a return value it will never need. The format attribute keeps the printf
 style arguments checked, which the direct printf() calls got for free. */
#if defined(__GNUC__) || defined(__clang__)
    #define OMC_NORETURN __attribute__((noreturn))
    #define OMC_PRINTF_LIKE(fmtArg, firstArg) \
        __attribute__((format(printf, fmtArg, firstArg)))
#elif defined(_MSC_VER)
    #define OMC_NORETURN __declspec(noreturn)
    #define OMC_PRINTF_LIKE(fmtArg, firstArg)
#else
    #define OMC_NORETURN
    #define OMC_PRINTF_LIKE(fmtArg, firstArg)
#endif
/*! @endcond */

/*! Severity of a message passed to omcLog(). The sink decides what to do
 with each level; nothing is filtered on the way there, so that a host can be
 as chatty or as quiet as it likes without the shared code knowing. */
enum OmcLogLevel {
    OMC_LOG_WARNING = 0,
    OMC_LOG_INFO,               /**< progress and summaries */
    OMC_LOG_DETAIL,             /**< details a curious user might want */
    OMC_LOG_DEBUG                /**< dumps only useful when something is wrong */
};

/*! The sinks a host installs with omcSetHost(). */
struct OmcHost {
    /*! Receives an already formatted message, without a trailing newline.

     @param level One of enum OmcLogLevel.
     @param message The formatted message.
     @param user The pointer from struct OmcHost::user, untouched. */
    void (*log)(int level, const char *message, void *user);

    /*! Reports a fatal condition.

     @param id Dotted identifier for hosts that can carry one, e.g.
     `"ompMC:geometry:badMaterialIndex"`. Hosts that cannot may ignore it.
     @param message The formatted message.
     @param user The pointer from struct OmcHost::user, untouched.

     @warning MUST NOT RETURN: the shared code calls this where it has no way
     to carry on, and simply continues into undefined state if the call comes
     back. Hosts end it by exiting the process (command line), throwing out
     of the call (MATLAB, Octave) or jumping back to the entry point with
     longjmp() (Python). omcFail() calls abort() if a sink returns anyway, so
     the mistake is loud rather than silent. */
    void (*fail)(const char *id, const char *message, void *user);

    void *user;                 ///< passed back to both sinks untouched
};

/*! Install the sinks.

 @param host The sinks to install; the pointer is not retained, the struct
 is copied. Passing `NULL` restores the built-in default, which prints to
 stdout/stderr and exits the process on a failure -- what a plain command
 line program wants, and a safe fallback for a host that forgets to install
 its own. */
void omcSetHost(const struct OmcHost *host);

/*! Format and hand a message to the log sink.

 @param level One of enum OmcLogLevel.
 @param fmt printf style format string, followed by its arguments. */
void omcLog(int level, const char *fmt, ...) OMC_PRINTF_LIKE(2, 3);

/*! Format and hand a fatal message to the fail sink. Does not return.

 @param id Dotted identifier, see struct OmcHost::fail.
 @param fmt printf style format string, followed by its arguments. */
OMC_NORETURN void omcFail(const char *id, const char *fmt, ...)
    OMC_PRINTF_LIKE(2, 3);

#endif

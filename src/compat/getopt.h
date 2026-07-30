#ifndef OMPMC_COMPAT_GETOPT_H
#define OMPMC_COMPAT_GETOPT_H
/******************************************************************************
 ompMC - An OpenMP parallel implementation for Monte Carlo particle transport
 simulations

 Copyright (C) 2024 ompMC developers


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

/******************************************************************************
 Drop-in replacement for the POSIX <getopt.h> header for toolchains that do not
 provide one -- most notably MSVC. The build system only puts the directory
 holding this file on the include path when the platform lacks getopt_long(),
 so on POSIX systems the ompMC user codes keep using the system header.

 The implementation covers the subset of the GNU interface that the ompMC user
 codes rely on: short options with required and optional arguments, long
 options in both "--name value" and "--name=value" form, unambiguous
 abbreviation of long names, and the "flag"/"val" mechanism of struct option.
 Argument permutation is not performed; option parsing stops at the first
 non-option argument, as it does under POSIXLY_CORRECT.
*****************************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

/* Argument of the option just returned, or NULL if it takes none. */
extern char *optarg;

/* Index of the next element of argv to be scanned. */
extern int optind;

/* Set to zero to silence the built-in error messages. */
extern int opterr;

/* The option character that caused the last '?' or ':' return. */
extern int optopt;

/* Values for the has_arg field of struct option. */
#define no_argument       0
#define required_argument 1
#define optional_argument 2

struct option {
    const char *name;   /* long option name, without the leading "--" */
    int has_arg;        /* no_argument, required_argument or optional_argument */
    int *flag;          /* if not NULL, *flag is set to val and 0 is returned */
    int val;            /* value to return, or to store through flag */
};

int getopt(int argc, char *const argv[], const char *optstring);

int getopt_long(int argc, char *const argv[], const char *optstring,
                const struct option *longopts, int *longindex);

#ifdef __cplusplus
}
#endif

#endif /* OMPMC_COMPAT_GETOPT_H */

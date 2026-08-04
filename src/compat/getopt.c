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
 Minimal getopt()/getopt_long() implementation for toolchains without a POSIX
 <getopt.h>. See getopt.h for the scope of the emulation.
*****************************************************************************/

#include "getopt.h"

#include <stdio.h>
#include <string.h>

char *optarg = NULL;
int optind = 1;
int opterr = 1;
int optopt = '?';

/* Offset of the next character to look at inside argv[optind]. A value of zero
 means that argv[optind] has not been touched yet. It is non-zero only while a
 cluster of short options such as "-abc" is being taken apart. */
static int subidx = 0;

/* Basename of argv[0], used as the prefix of the diagnostic messages. */
static const char *progname(char *const argv[]) {

    const char *base = (argv != NULL && argv[0] != NULL) ? argv[0] : "";
    const char *s;

    for (s = base; *s != '\0'; s++) {
        if (*s == '/' || *s == '\\') {
            base = s + 1;
        }
    }

    return base;
}

/* Handle argv[optind], which is known to start with "--" followed by at least
 one more character. */
static int parse_long(int argc, char *const argv[], const char *optstring,
                      const struct option *longopts, int *longindex) {

    const char *name = argv[optind] + 2;            /* skip the leading "--" */
    const char *eq = strchr(name, '=');
    size_t len = (eq != NULL) ? (size_t)(eq - name) : strlen(name);
    int quiet = (optstring[0] == ':');
    int match = -1;
    int nmatch = 0;
    int i;

    for (i = 0; longopts[i].name != NULL; i++) {
        if (strncmp(longopts[i].name, name, len) != 0) {
            continue;
        }
        if (strlen(longopts[i].name) == len) {      /* exact match wins */
            match = i;
            nmatch = 1;
            break;
        }
        match = i;                                  /* unique abbreviation? */
        nmatch++;
    }

    optind++;

    if (nmatch != 1) {
        optopt = 0;
        if (opterr && !quiet) {
            fprintf(stderr, "%s: %s option '--%.*s'\n", progname(argv),
                    (nmatch == 0) ? "unrecognized" : "ambiguous",
                    (int)len, name);
        }
        return '?';
    }

    if (longindex != NULL) {
        *longindex = match;
    }

    if (longopts[match].has_arg == no_argument) {
        if (eq != NULL) {
            optopt = longopts[match].val;
            if (opterr && !quiet) {
                fprintf(stderr, "%s: option '--%s' doesn't allow an argument\n",
                        progname(argv), longopts[match].name);
            }
            return '?';
        }
    }
    else if (eq != NULL) {
        optarg = (char *)eq + 1;
    }
    else if (longopts[match].has_arg == required_argument) {
        if (optind >= argc) {
            optopt = longopts[match].val;
            if (opterr && !quiet) {
                fprintf(stderr, "%s: option '--%s' requires an argument\n",
                        progname(argv), longopts[match].name);
            }
            return quiet ? ':' : '?';
        }
        optarg = argv[optind++];
    }

    if (longopts[match].flag != NULL) {
        *longopts[match].flag = longopts[match].val;
        return 0;
    }

    return longopts[match].val;
}

int getopt_long(int argc, char *const argv[], const char *optstring,
                const struct option *longopts, int *longindex) {

    const char *arg;
    const char *spec;
    int quiet;
    int c;

    optarg = NULL;

    if (optstring == NULL) {
        optstring = "";
    }
    /* The GNU "+" and "-" ordering prefixes are accepted but ignored: this
     implementation never permutes argv. */
    while (*optstring == '+' || *optstring == '-') {
        optstring++;
    }
    quiet = (*optstring == ':');

    if (optind < 1) {
        optind = 1;
    }
    if (argv == NULL || optind >= argc) {
        subidx = 0;
        return -1;
    }

    arg = argv[optind];

    if (subidx == 0) {
        if (arg == NULL || arg[0] != '-' || arg[1] == '\0') {
            return -1;                              /* non-option, or "-" */
        }
        if (arg[1] == '-') {
            if (arg[2] == '\0') {                   /* "--" ends the options */
                optind++;
                return -1;
            }
            if (longopts != NULL) {
                return parse_long(argc, argv, optstring, longopts, longindex);
            }
        }
        subidx = 1;
    }

    c = (unsigned char)arg[subidx++];
    if (arg[subidx] == '\0') {          /* last character of this argv element */
        optind++;
        subidx = 0;
    }

    spec = (c == ':') ? NULL : strchr(optstring, c);
    if (spec == NULL) {
        optopt = c;
        if (opterr && !quiet) {
            fprintf(stderr, "%s: invalid option -- '%c'\n", progname(argv), c);
        }
        return '?';
    }

    if (spec[1] != ':') {
        return c;                                   /* takes no argument */
    }

    if (subidx != 0) {              /* argument attached, as in "-ivalue" */
        optarg = (char *)arg + subidx;
        optind++;
        subidx = 0;
        return c;
    }

    if (spec[2] == ':') {           /* optional argument, and none attached */
        return c;
    }

    if (optind >= argc) {
        optopt = c;
        if (opterr && !quiet) {
            fprintf(stderr, "%s: option requires an argument -- '%c'\n",
                    progname(argv), c);
        }
        return quiet ? ':' : '?';
    }

    optarg = argv[optind++];
    return c;
}

int getopt(int argc, char *const argv[], const char *optstring) {

    return getopt_long(argc, argv, optstring, NULL, NULL);
}

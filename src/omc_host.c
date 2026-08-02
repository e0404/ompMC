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

#include "omc_host.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

/* Long enough for every message the code base produces; the few that embed a
 file path are the only ones that get anywhere near it. Truncation is
 preferable to a heap allocation on a path that is often an error path. */
#define OMC_MESSAGE_SIZE 1024

/******************************************************************************/
/* Default sinks: a plain command line program. */

static void defaultLog(int level, const char *message, void *user) {

    (void)level;
    (void)user;

    printf("%s\n", message);

    return;
}

static void defaultFail(const char *id, const char *message, void *user) {

    (void)id;
    (void)user;

    fflush(stdout);
    fprintf(stderr, "%s\n", message);

    exit(EXIT_FAILURE);
}

static struct OmcHost host = { defaultLog, defaultFail, NULL };

/******************************************************************************/

void omcSetHost(const struct OmcHost *newHost) {

    if (newHost == NULL) {
        host.log = defaultLog;
        host.fail = defaultFail;
        host.user = NULL;
        return;
    }

    /* A host that installs only one of the two sinks keeps the default for
     the other, rather than a NULL that would be called later */
    host.log = newHost->log ? newHost->log : defaultLog;
    host.fail = newHost->fail ? newHost->fail : defaultFail;
    host.user = newHost->user;

    return;
}

void omcLog(int level, const char *fmt, ...) {

    char message[OMC_MESSAGE_SIZE];
    va_list args;

    va_start(args, fmt);
    vsnprintf(message, sizeof(message), fmt, args);
    va_end(args);

    host.log(level, message, host.user);

    return;
}

void omcFail(const char *id, const char *fmt, ...) {

    char message[OMC_MESSAGE_SIZE];
    va_list args;

    va_start(args, fmt);
    vsnprintf(message, sizeof(message), fmt, args);
    va_end(args);

    host.fail(id, message, host.user);

    /* The sink broke its contract. Carrying on would run the very code the
     caller decided it could not run, so stop here where the cause is still
     obvious. */
    fflush(stdout);
    fprintf(stderr,
        "ompMC: the host's failure handler returned, which it must not do. "
        "The message was: %s\n", message);
    abort();
}

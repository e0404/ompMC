/******************************************************************************
 ompMC - An OpenMP parallel implementation for Monte Carlo particle transport
 simulations
 
 Copyright (C) 2020 Edgardo Doerner (edoerner@fis.puc.cl)


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

#include "omc_utilities.h"

#include "omc_host.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Redefine printf() function due to conflicts with mex and OpenMP */
#ifdef _OPENMP
    #include <omp.h>

    #undef printf
    #define printf(...) fprintf(stderr,__VA_ARGS__)
#endif

/* The one definition of the table omc_utilities.h declares. It used to be
 written twice in this file: a tentative definition of incomplete type here,
 and the real one at the bottom. Legal C, since the type is completed before
 the end of the translation unit, but there is no reason to make a reader work
 that out. */
struct inputItems input_items[INPUT_PAIRS];     // key,value pairs
int input_idx = 0;                              // number of key,value pairs

/* Thread-local geometry memo declared in omc_utilities.h. Zero initialized,
 which marks both halves as empty. */
#if defined(_MSC_VER)
    __declspec(thread) struct OmcGeomCache omc_geom_cache;
#else
    struct OmcGeomCache omc_geom_cache;
#endif

/******************************************************************************/
/* Timing utilities. If OpenMP is enabled it calculates the wall time through 
 omp_get_wtime() function. Otherwise, it calculates CPU time through the clock() 
 function, available in time.h library. */

double omc_get_time() {

    double time_s;    // time in seconds.

#ifdef _OPENMP
    time_s = omp_get_wtime();
#else
    time_s = (double)clock()/CLOCKS_PER_SEC;
#endif

    return time_s;
}
/******************************************************************************/

/******************************************************************************/
/* A simple C/C++ class to parse input files and return requested
 key value -- https://github.com/bmaynard/iniReader */

#include <string.h>
#include <ctype.h>

/* Trim leading and trailing whitespace in place. Internal whitespace stays,
 so multi-word keys like "global ecut" keep their exact spelling. */
static void trimSpaces(char *str) {

    char *start = str;
    while (isspace((unsigned char)*start)) {
        start++;
    }

    size_t len = strlen(start);
    while (len > 0 && isspace((unsigned char)start[len - 1])) {
        len--;
    }

    memmove(str, start, len);
    str[len] = '\0';

    return;
}

/* Parse a configuration file */
void parseInputFile(char *input_file) {

    char buf[BUFFER_SIZE];      // support lines up to 120 characters

    /* Make space for the new string */
    const char *extension = INPUT_EXT;
    char *file_name = malloc(strlen(input_file) + strlen(extension) + 1);
    strcpy(file_name, input_file);
    strcat(file_name, extension); /* add the extension */

    FILE *fp;
    if ((fp = fopen(file_name, "r")) == NULL) {
        printf("Unable to open file: %s\n", file_name);
        exit(EXIT_FAILURE);
    }

    /* Set to the key of the pair that did not fit, if the table fills up */
    const char *overflow_key = NULL;

    while (fgets(buf, BUFFER_SIZE , fp) != NULL) {
        /* Jumps lines labeled with #, together with only white
         spaced or empty ones. */
        if (strstr(buf, "#") || lineBlack(buf)) {
            continue;
        }

        /* Lines without a '=' cannot form a key, value pair; skip them
         instead of handing strcpy a NULL */
        char *key = strtok(buf, "=\r\n");
        char *value = (key != NULL) ? strtok(NULL, "\r\n") : NULL;
        if (key == NULL || value == NULL) {
            printf("Skipping malformed input line without 'key = value' "
                   "form.\n");
            continue;
        }

        /* The table is a fixed size array, and nothing stopped a long enough
         deck -- or a second call without omcClearInputValues() in between --
         from walking off the end of it. Stop at the edge and report it rather
         than storing the pair; a deck whose settings were silently dropped
         would calculate with defaults nobody asked for. */
        if (input_idx >= INPUT_PAIRS) {
            overflow_key = key;
            break;
        }

        /* Store trimmed of surrounding whitespace, so that keys can be
         compared exactly rather than by substring */
        strcpy(input_items[input_idx].key, key);
        strcpy(input_items[input_idx].value, value);
        trimSpaces(input_items[input_idx].key);
        trimSpaces(input_items[input_idx].value);
        input_idx++;
    }

    /* No decrement here. This used to leave input_idx at the index of the last
     pair while every other way of filling the table left a count, and the two
     differ by one exactly when the file holds a single pair -- which is the
     case whose lookups then failed. */
    fclose(fp);

    if (overflow_key != NULL) {
        /* omcFail() does not return, and a host that carries on afterwards --
         a MEX file throwing out, the Python module jumping back -- stays
         resident, so hand the message a copy and let the buffers go first. */
        char key_copy[BUFFER_SIZE];
        char name_copy[PATH_SIZE];
        snprintf(key_copy, sizeof(key_copy), "%s", overflow_key);
        snprintf(name_copy, sizeof(name_copy), "%s", file_name);
        free(file_name);

        omcFail("ompMC:input:tooManyItems",
            "Cannot store input item '%s' from %s: the table holds at most "
            "%d pairs.", key_copy, name_copy, INPUT_PAIRS);
    }

    if(verbose_flag) {
        for (int i = 0; i < input_idx; i++) {
            printf("key = %s, value = %s\n", input_items[i].key,
                   input_items[i].value);
        }
    }

    /* Cleaning */
    free(file_name);
    
    return;
}

/* Copy the value of the selected input item to the char pointer */
int getInputValue(char *dest, char *key) {

    /* No "nothing got parsed" guard on input_idx here. It used to return early
     when input_idx was 0, which a one pair table was indistinguishable from
     back when this counted to the last index instead of counting pairs. Now
     that input_idx is a count, an empty table is 0 and the loop simply does
     not run. */
    for (int i = 0; i < input_idx; i++) {
        /* Keys are stored trimmed, so exact comparison is safe. The substring
         match used before let a short key like "ecut" answer for
         "global ecut", depending only on storage order. */
        if (strcmp(input_items[i].key, key) == 0) {
            strcpy(dest, input_items[i].value);
            return 1;
        }
    }

    return 0;
}

void omcSetInputValue(const char *key, const char *value) {

    /* Replace the value if this key is already known, so that a host can
     override one setting of a deck it just parsed without the table growing a
     second entry that only storage order would decide between. */
    for (int i = 0; i < input_idx; i++) {
        if (strcmp(input_items[i].key, key) == 0) {
            strncpy(input_items[i].value, value, BUFFER_SIZE - 1);
            input_items[i].value[BUFFER_SIZE - 1] = '\0';
            return;
        }
    }

    if (input_idx >= INPUT_PAIRS) {
        omcFail("ompMC:input:tooManyItems",
            "Cannot store input item '%s': the table holds at most %d pairs.",
            key, INPUT_PAIRS);
    }

    /* Append at the count and then raise it, so the first pair set on a
     cleared table lands in slot 0. Pre-incrementing instead, as this used to,
     left slot 0 permanently empty and the table one pair short of the
     INPUT_PAIRS it advertises. */
    strncpy(input_items[input_idx].key, key, BUFFER_SIZE - 1);
    input_items[input_idx].key[BUFFER_SIZE - 1] = '\0';
    strncpy(input_items[input_idx].value, value, BUFFER_SIZE - 1);
    input_items[input_idx].value[BUFFER_SIZE - 1] = '\0';
    input_idx++;

    return;
}

void omcClearInputValues(void) {

    for (int i = 0; i < INPUT_PAIRS; i++) {
        input_items[i].key[0] = '\0';
        input_items[i].value[0] = '\0';
    }
    input_idx = 0;

    return;
}

/* Returns nonzero if line is a string containing only whitespace or is empty */
int lineBlack(char *line) {
    char * ch;
    int is_blank = 1;
    
    /* Iterate through each character. */
    for (ch = line; *ch != '\0'; ++ch) {
        if (!isspace(*ch)) {
            /* Found a non-whitespace character. */
            is_blank = 0;
            break;
        }
    }
    
    return is_blank;
}

/* Remove white spaces from string str_untrimmed and saves the results in
 str_trimmed. Useful for string input values, such as file names */
 void removeSpaces(char* str_trimmed,
                  const char* str_untrimmed) {
    
    while (*str_untrimmed != '\0') {
        if(!isspace(*str_untrimmed)) {
            *str_trimmed = *str_untrimmed;
            str_trimmed++;
        }
        str_untrimmed++;
    }
    
    *str_trimmed = '\0';
    return;
}

/******************************************************************************/
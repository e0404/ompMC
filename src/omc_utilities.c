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

struct inputItems input_items[];     // key,value pairs
int input_idx;                       // number of key,value pair

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

        /* Store trimmed of surrounding whitespace, so that keys can be
         compared exactly rather than by substring */
        strcpy(input_items[input_idx].key, key);
        strcpy(input_items[input_idx].value, value);
        trimSpaces(input_items[input_idx].key);
        trimSpaces(input_items[input_idx].value);
        input_idx++;
    }

    input_idx--;
    fclose(fp);
    
    if(verbose_flag) {
        /* input_idx is the index of the last pair, not a count, so the last
         one has to be included here too */
        for (int i = 0; i <= input_idx; i++) {
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
     when input_idx was 0, but parseInputFile() leaves input_idx at the index
     of the LAST pair, so a file holding exactly one pair also ends at 0 and
     every lookup against it failed. An empty table needs no guard: it either
     leaves input_idx at -1, so the loop below does not run, or holds empty
     keys, which no real key compares equal to. */
    for (int i = 0; i <= input_idx; i++) {
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

    /* Replace the value if this key is already known. getInputValue() walks
     up to and including input_idx, so a duplicate would be found only by
     storage order. */
    for (int i = 0; i <= input_idx && i < INPUT_PAIRS; i++) {
        if (strcmp(input_items[i].key, key) == 0) {
            strncpy(input_items[i].value, value, BUFFER_SIZE - 1);
            input_items[i].value[BUFFER_SIZE - 1] = '\0';
            return;
        }
    }

    if (input_idx >= INPUT_PAIRS - 1) {
        omcFail("ompMC:input:tooManyItems",
            "Cannot store input item '%s': the table holds at most %d pairs.",
            key, INPUT_PAIRS);
    }

    /* input_idx is the index of the last stored pair rather than a count --
     that is what parseInputFile() leaves behind and what getInputValue()
     scans up to -- so appending pre-increments. On a table that was never
     filled this skips slot 0, which costs one of INPUT_PAIRS entries and is
     otherwise harmless: its key stays the empty string and matches nothing.
     Following the same convention is what lets the two ways of filling the
     table be mixed. */
    input_idx++;
    strncpy(input_items[input_idx].key, key, BUFFER_SIZE - 1);
    input_items[input_idx].key[BUFFER_SIZE - 1] = '\0';
    strncpy(input_items[input_idx].value, value, BUFFER_SIZE - 1);
    input_items[input_idx].value[BUFFER_SIZE - 1] = '\0';

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

struct inputItems input_items[INPUT_PAIRS];     // key,value pairs
int input_idx = 0;                              // number of key,value pair

/******************************************************************************/
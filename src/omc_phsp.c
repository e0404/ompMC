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

#include "omc_phsp.h"

#include "omc_host.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Long enough for the deepest install path anyone has hit us with, and for
 the longest line the format's own reference implementation allows. */
#define PHSP_PATH_SIZE 1024
#define PHSP_LINE_SIZE 512

static const char HEADER_EXT[] = ".IAEAheader";
static const char PHSP_EXT[] = ".IAEAphsp";

/*******************************************************************************
* Paths
*
* A dataset is two files sharing a base name. Callers may hand over the base
* name or either of the two file names, so both extensions come off first.
*******************************************************************************/

static int endsWithNoCase(const char *text, const char *suffix) {

    size_t ntext = strlen(text);
    size_t nsuffix = strlen(suffix);

    if (ntext < nsuffix) {
        return 0;
    }

    const char *tail = text + (ntext - nsuffix);
    for (size_t i = 0; i < nsuffix; i++) {
        if (tolower((unsigned char)tail[i]) != tolower((unsigned char)suffix[i])) {
            return 0;
        }
    }

    return 1;
}

/* Fill @p dest with the path of one of the dataset's files. */
static void datasetPath(char *dest, size_t cap, const char *path,
                        const char *extension) {

    size_t n = strlen(path);

    if (endsWithNoCase(path, HEADER_EXT)) {
        n -= strlen(HEADER_EXT);
    }
    else if (endsWithNoCase(path, PHSP_EXT)) {
        n -= strlen(PHSP_EXT);
    }

    if (n + strlen(extension) + 1 > cap) {
        omcFail("ompMC:phsp:pathTooLong",
            "Path to phase space file is longer than the %d characters this "
            "reader can hold: %s", (int)cap - 1, path);
    }

    memcpy(dest, path, n);
    memcpy(dest + n, extension, strlen(extension) + 1);
}

/*******************************************************************************
* Reading the header
*
* The header is a set of blocks rather than a sequence of them: a block opens
* with "$KEYWORD:" and runs to the next line starting with a $, and nothing
* says which order the writer put them in. So each keyword is looked up from
* the top of the file, the way the format's reference implementation does it.
* Values may carry C and C++ comments, which the writer uses to label what it
* has written, and blank lines are ignored.
*******************************************************************************/

struct HeaderReader {
    FILE *fp;
    const char *path;           /* for the failure messages */
    int inComment;              /* a /_* block comment is still open */
};

/* Strip comments from a line and trim it. Returns the length of what is
 left, which is zero for a line that was only a comment. */
static size_t stripLine(struct HeaderReader *reader, char *line) {

    size_t at = 0;
    size_t out = 0;

    while (line[at] != '\0') {
        if (reader->inComment) {
            if (line[at] == '*' && line[at + 1] == '/') {
                reader->inComment = 0;
                at += 2;
            }
            else {
                at++;
            }
            continue;
        }

        if (line[at] == '/' && line[at + 1] == '/') {
            break;              /* the rest of the line is a comment */
        }
        if (line[at] == '/' && line[at + 1] == '*') {
            reader->inComment = 1;
            at += 2;
            continue;
        }

        line[out++] = line[at++];
    }

    line[out] = '\0';

    /* Trim, including the carriage return a header written on Windows and
     read on anything else still carries. */
    size_t begin = 0;
    while (line[begin] != '\0' && isspace((unsigned char)line[begin])) {
        begin++;
    }

    size_t end = strlen(line + begin);
    while (end > 0 && isspace((unsigned char)line[begin + end - 1])) {
        end--;
    }

    memmove(line, line + begin, end);
    line[end] = '\0';

    return end;
}

/* Read the next line that has something on it.

 @return 1 if @p line was filled, 0 at the end of the file. */
static int readLine(struct HeaderReader *reader, char *line, size_t cap) {

    while (fgets(line, (int)cap, reader->fp) != NULL) {
        if (stripLine(reader, line) > 0) {
            return 1;
        }
    }

    return 0;
}

/* Position the reader just after the line that opens a block.

 @return 1 if the block is there, 0 if the file has no such keyword. */
static int findBlock(struct HeaderReader *reader, const char *keyword) {

    char line[PHSP_LINE_SIZE];

    rewind(reader->fp);
    reader->inComment = 0;

    while (readLine(reader, line, sizeof(line))) {
        if (line[0] != '$') {
            continue;
        }

        char *end = strchr(line + 1, ':');
        if (end == NULL) {
            continue;
        }

        *end = '\0';
        if (strcmp(line + 1, keyword) == 0) {
            return 1;
        }
    }

    return 0;
}

/* Read the next value of the block the reader is inside.

 @return 1 if @p line was filled, 0 at the end of the block or file. */
static int nextValue(struct HeaderReader *reader, char *line, size_t cap) {

    if (!readLine(reader, line, cap)) {
        return 0;
    }

    return line[0] != '$';
}

static void requireBlock(struct HeaderReader *reader, const char *keyword) {

    if (!findBlock(reader, keyword)) {
        fclose(reader->fp);
        omcFail("ompMC:phsp:missingKeyword",
            "Phase space header %s has no $%s: block, which the format "
            "requires.", reader->path, keyword);
    }
}

/* Read the one value of a block that holds a single number. */
static void readValue(struct HeaderReader *reader, const char *keyword,
                      char *line, size_t cap) {

    requireBlock(reader, keyword);

    if (!nextValue(reader, line, cap)) {
        fclose(reader->fp);
        omcFail("ompMC:phsp:parseFailed",
            "Block $%s: of phase space header %s is empty.", keyword,
            reader->path);
    }
}

static long parseLong(struct HeaderReader *reader, const char *keyword,
                      const char *text) {

    char *end;
    long value = strtol(text, &end, 10);

    if (end == text || *end != '\0') {
        fclose(reader->fp);
        omcFail("ompMC:phsp:parseFailed",
            "Block $%s: of phase space header %s holds \"%s\", which is not a "
            "whole number.", keyword, reader->path, text);
    }

    return value;
}

static unsigned long long parseULongLong(struct HeaderReader *reader,
                                         const char *keyword,
                                         const char *text) {

    char *end;
    unsigned long long value;

    if (text[0] == '-') {
        fclose(reader->fp);
        omcFail("ompMC:phsp:parseFailed",
            "Block $%s: of phase space header %s holds \"%s\", which is "
            "negative.", keyword, reader->path, text);
    }

    value = strtoull(text, &end, 10);

    if (end == text || *end != '\0') {
        fclose(reader->fp);
        omcFail("ompMC:phsp:parseFailed",
            "Block $%s: of phase space header %s holds \"%s\", which is not a "
            "whole number.", keyword, reader->path, text);
    }

    return value;
}

static double parseDouble(struct HeaderReader *reader, const char *keyword,
                          const char *text) {

    char *end;
    double value = strtod(text, &end);

    if (end == text || *end != '\0') {
        fclose(reader->fp);
        omcFail("ompMC:phsp:parseFailed",
            "Block $%s: of phase space header %s holds \"%s\", which is not a "
            "number.", keyword, reader->path, text);
    }

    return value;
}

static long readLongBlock(struct HeaderReader *reader, const char *keyword) {

    char line[PHSP_LINE_SIZE];

    readValue(reader, keyword, line, sizeof(line));

    return parseLong(reader, keyword, line);
}

static unsigned long long readULongLongBlock(struct HeaderReader *reader,
                                             const char *keyword) {

    char line[PHSP_LINE_SIZE];

    readValue(reader, keyword, line, sizeof(line));

    return parseULongLong(reader, keyword, line);
}

/* The optional per particle type counts. Absent means the header does not
 say, which is not the same as zero, but there is nothing else to report. */
static unsigned long long readOptionalCount(struct HeaderReader *reader,
                                            const char *keyword) {

    char line[PHSP_LINE_SIZE];

    if (!findBlock(reader, keyword)) {
        return 0;
    }
    if (!nextValue(reader, line, sizeof(line))) {
        return 0;
    }

    return parseULongLong(reader, keyword, line);
}

/* $RECORD_CONTENTS: nine flags saying what a record holds, followed by what
 the extra floats and longs it holds are for. */
static void readRecordContents(struct HeaderReader *reader,
                               struct OmcPhspHeader *header) {

    char line[PHSP_LINE_SIZE];
    long values[OMC_PHSP_NVARIABLES + 2];

    requireBlock(reader, "RECORD_CONTENTS");

    for (int i = 0; i < OMC_PHSP_NVARIABLES + 2; i++) {
        if (!nextValue(reader, line, sizeof(line))) {
            fclose(reader->fp);
            omcFail("ompMC:phsp:parseFailed",
                "Block $RECORD_CONTENTS: of phase space header %s holds %d "
                "values, expected the %d the format defines.", reader->path,
                i, OMC_PHSP_NVARIABLES + 2);
        }
        values[i] = parseLong(reader, "RECORD_CONTENTS", line);
    }

    for (int i = 0; i < OMC_PHSP_NVARIABLES; i++) {
        header->stored[i] = values[i] != 0;
    }

    header->nExtraFloat = (int)values[OMC_PHSP_NVARIABLES];
    header->nExtraLong = (int)values[OMC_PHSP_NVARIABLES + 1];

    if (header->nExtraFloat < 0 || header->nExtraFloat > OMC_PHSP_MAX_EXTRA) {
        fclose(reader->fp);
        omcFail("ompMC:phsp:tooManyExtras",
            "Phase space header %s announces %d extra floats per record, "
            "outside the 0 to %d this reader holds. Increase "
            "OMC_PHSP_MAX_EXTRA!", reader->path, header->nExtraFloat,
            OMC_PHSP_MAX_EXTRA);
    }
    if (header->nExtraLong < 0 || header->nExtraLong > OMC_PHSP_MAX_EXTRA) {
        fclose(reader->fp);
        omcFail("ompMC:phsp:tooManyExtras",
            "Phase space header %s announces %d extra longs per record, "
            "outside the 0 to %d this reader holds. Increase "
            "OMC_PHSP_MAX_EXTRA!", reader->path, header->nExtraLong,
            OMC_PHSP_MAX_EXTRA);
    }

    /* What each extra is for follows in the same block, floats first. */
    for (int i = 0; i < header->nExtraFloat; i++) {
        if (!nextValue(reader, line, sizeof(line))) {
            fclose(reader->fp);
            omcFail("ompMC:phsp:parseFailed",
                "Block $RECORD_CONTENTS: of phase space header %s announces "
                "%d extra floats but says what only %d of them are for.",
                reader->path, header->nExtraFloat, i);
        }
        header->extraFloatType[i] =
            (int)parseLong(reader, "RECORD_CONTENTS", line);
    }

    for (int i = 0; i < header->nExtraLong; i++) {
        if (!nextValue(reader, line, sizeof(line))) {
            fclose(reader->fp);
            omcFail("ompMC:phsp:parseFailed",
                "Block $RECORD_CONTENTS: of phase space header %s announces "
                "%d extra longs but says what only %d of them are for.",
                reader->path, header->nExtraLong, i);
        }
        header->extraLongType[i] =
            (int)parseLong(reader, "RECORD_CONTENTS", line);
    }
}

/* $RECORD_CONSTANT: a value for each quantity $RECORD_CONTENTS said is NOT
 stored, in the order the format lists them. Which line means what therefore
 depends on the block before this one, and a file where every quantity is
 stored needs no such block at all. */
static void readRecordConstants(struct HeaderReader *reader,
                                struct OmcPhspHeader *header) {

    char line[PHSP_LINE_SIZE];
    int wanted = 0;

    for (int i = 0; i < OMC_PHSP_NVARIABLES; i++) {
        header->constant[i] = 0.0f;
        if (!header->stored[i]) {
            wanted++;
        }
    }

    if (wanted == 0) {
        return;
    }

    requireBlock(reader, "RECORD_CONSTANT");

    for (int i = 0; i < OMC_PHSP_NVARIABLES; i++) {
        if (header->stored[i]) {
            continue;
        }

        if (!nextValue(reader, line, sizeof(line))) {
            fclose(reader->fp);
            omcFail("ompMC:phsp:parseFailed",
                "Block $RECORD_CONSTANT: of phase space header %s is missing "
                "the value of a quantity $RECORD_CONTENTS: says is not stored "
                "per particle.", reader->path);
        }

        header->constant[i] =
            (float)parseDouble(reader, "RECORD_CONSTANT", line);
    }
}

/* What a record of this layout measures.

 @warning w is not in here, and that is not an oversight: a record never
 holds w, only its sign, which rides along in the sign of the particle type
 byte. Counting it would put every particle but the first at the wrong
 offset. */
static int recordLengthOf(const struct OmcPhspHeader *header) {

    int length = 1 + 4;         /* particle type and energy, always there */

    for (int i = 0; i < OMC_PHSP_NVARIABLES; i++) {
        if (i != OMC_PHSP_W && header->stored[i]) {
            length += 4;
        }
    }

    length += 4*header->nExtraFloat;
    length += 4*header->nExtraLong;

    return length;
}

void omcPhspHeaderFromFile(struct OmcPhspHeader *header, const char *path) {

    char headerPath[PHSP_PATH_SIZE];
    struct HeaderReader reader;

    datasetPath(headerPath, sizeof(headerPath), path, HEADER_EXT);

    reader.fp = fopen(headerPath, "r");
    reader.path = headerPath;
    reader.inComment = 0;

    if (reader.fp == NULL) {
        omcFail("ompMC:phsp:openFailed",
            "Unable to open phase space header file: %s", headerPath);
    }

    omcLog(OMC_LOG_DEBUG, "Path to phase space header file : %s", headerPath);

    memset(header, 0, sizeof(*header));

    /* What kind of file this is and how it is laid out decide whether the
     rest is worth reading, and are checked before it: a file describing an
     event generator does not carry the same blocks at all, and complaining
     that it has no $PARTICLES: block would be answering the wrong
     question. */
    header->fileType = (int)readLongBlock(&reader, "FILE_TYPE");
    if (header->fileType != 0) {
        fclose(reader.fp);
        omcFail("ompMC:phsp:unsupportedFileType",
            "Phase space header %s is of file type %d. Only 0, a phase space "
            "file, can be read; 1 describes an event generator.", headerPath,
            header->fileType);
    }

    header->byteOrder = (int)readLongBlock(&reader, "BYTE_ORDER");
    if (header->byteOrder != 1234) {
        fclose(reader.fp);
        omcFail("ompMC:phsp:unsupportedByteOrder",
            "Phase space header %s declares byte order %d. Only 1234, little "
            "endian, can be read.", headerPath, header->byteOrder);
    }

    header->checksum = readULongLongBlock(&reader, "CHECKSUM");

    readRecordContents(&reader, header);
    readRecordConstants(&reader, header);

    header->recordLength = (int)readLongBlock(&reader, "RECORD_LENGTH");
    header->origHistories = readULongLongBlock(&reader, "ORIG_HISTORIES");
    header->particles = readULongLongBlock(&reader, "PARTICLES");

    header->typeCount[OMC_PHSP_PHOTON - 1] =
        readOptionalCount(&reader, "PHOTONS");
    header->typeCount[OMC_PHSP_ELECTRON - 1] =
        readOptionalCount(&reader, "ELECTRONS");
    header->typeCount[OMC_PHSP_POSITRON - 1] =
        readOptionalCount(&reader, "POSITRONS");
    header->typeCount[OMC_PHSP_NEUTRON - 1] =
        readOptionalCount(&reader, "NEUTRONS");
    header->typeCount[OMC_PHSP_PROTON - 1] =
        readOptionalCount(&reader, "PROTONS");

    fclose(reader.fp);

    int expected = recordLengthOf(header);
    if (header->recordLength != expected) {
        omcFail("ompMC:phsp:recordLengthMismatch",
            "Phase space header %s declares a record length of %d bytes, but "
            "the quantities it says a record holds add up to %d.", headerPath,
            header->recordLength, expected);
    }

    /* The checksum is the size the binary file should have, so there is
     nothing to check against it here -- omcPhspFromFile() does that. It is
     worth saying that it is NOT the record length times the particle count:
     published datasets exist whose particle count is a little over what the
     file holds, and taking the two to agree would turn them away. */

    omcLog(OMC_LOG_DETAIL, "Phase space file holds %llu particles of %d bytes "
           "from %llu histories.", (unsigned long long)header->particles,
           header->recordLength, (unsigned long long)header->origHistories);

    return;
}

/*******************************************************************************
* Reading the particles
*
* The records are stored little endian, and are pulled apart a byte at a time
* rather than cast at, so that a big endian machine reads the same file the
* same way and an odd record length cannot land a float on an address it may
* not be read from.
*******************************************************************************/

static float leFloat(const unsigned char *at) {

    uint32_t bits = (uint32_t)at[0] |
                    ((uint32_t)at[1] << 8) |
                    ((uint32_t)at[2] << 16) |
                    ((uint32_t)at[3] << 24);
    float value;

    memcpy(&value, &bits, sizeof(value));

    return value;
}

static int32_t leInt32(const unsigned char *at) {

    uint32_t bits = (uint32_t)at[0] |
                    ((uint32_t)at[1] << 8) |
                    ((uint32_t)at[2] << 16) |
                    ((uint32_t)at[3] << 24);
    int32_t value;

    memcpy(&value, &bits, sizeof(value));

    return value;
}

/* Take one record apart. Everything that could go wrong with the bytes was
 settled while loading, so this only has to read them. */
static void decodeRecord(const struct OmcPhspHeader *header,
                         const unsigned char *at,
                         struct OmcPhspRecord *record) {

    /* The type byte is signed, and its sign is where the sign of w lives.
     Reading it through signed char rather than the plain char that may be
     unsigned is the difference between -1 and 255. */
    int type = (int)(signed char)at[0];
    double wsign = 1.0;

    at += 1;

    if (type < 0) {
        wsign = -1.0;
        type = -type;
    }
    record->type = type;

    /* And a negative energy is how the format says a new history starts
     here, so the energy proper is its magnitude. */
    double energy = leFloat(at);
    at += 4;

    record->newHistory = energy < 0.0;
    record->energy = fabs(energy);

    double value[OMC_PHSP_NVARIABLES];
    for (int i = 0; i < OMC_PHSP_NVARIABLES; i++) {
        if (i == OMC_PHSP_W) {
            continue;           /* never in a record, see below */
        }
        if (header->stored[i]) {
            value[i] = leFloat(at);
            at += 4;
        }
        else {
            value[i] = header->constant[i];
        }
    }

    record->x = value[OMC_PHSP_X];
    record->y = value[OMC_PHSP_Y];
    record->z = value[OMC_PHSP_Z];
    record->u = value[OMC_PHSP_U];
    record->v = value[OMC_PHSP_V];
    record->weight = value[OMC_PHSP_WEIGHT];

    /* w is the one quantity a record never carries: a stored w is recovered
     from u and v, and only a constant one is taken as it stands. Rounding
     can leave u and v describing more than a unit vector, in which case they
     are scaled back onto the unit circle rather than a negative number being
     handed to sqrt(). */
    if (header->stored[OMC_PHSP_W]) {
        double aux = record->u*record->u + record->v*record->v;

        if (aux <= 1.0) {
            record->w = wsign*sqrt(1.0 - aux);
        }
        else {
            aux = sqrt(aux);
            record->u /= aux;
            record->v /= aux;
            record->w = 0.0;
        }
    }
    else {
        record->w = header->constant[OMC_PHSP_W];
    }

    for (int i = 0; i < header->nExtraFloat; i++) {
        record->extraFloat[i] = leFloat(at);
        at += 4;
    }
    for (int i = 0; i < header->nExtraLong; i++) {
        record->extraLong[i] = leInt32(at);
        at += 4;
    }
}

void omcPhspFromFile(struct OmcPhsp *phsp, const char *path) {

    char phspPath[PHSP_PATH_SIZE];
    FILE *fp;

    omcPhspHeaderFromFile(&phsp->header, path);

    phsp->nRecords = phsp->header.particles;
    phsp->newHistories = 0;
    phsp->raw = NULL;
    phsp->cursor = 0;

    datasetPath(phspPath, sizeof(phspPath), path, PHSP_EXT);

    /* Binary, and on Windows that matters: a 0x1A byte in the middle of a
     record ends a text mode read. */
    fp = fopen(phspPath, "rb");
    if (fp == NULL) {
        omcFail("ompMC:phsp:openFailed",
            "Unable to open phase space file: %s", phspPath);
    }

    omcLog(OMC_LOG_DEBUG, "Path to phase space file : %s", phspPath);

    unsigned long long needed =
        phsp->nRecords*(unsigned long long)phsp->header.recordLength;

    if (needed > (unsigned long long)SIZE_MAX) {
        fclose(fp);
        omcFail("ompMC:phsp:outOfMemory",
            "Phase space file %s holds %llu bytes, more than this build can "
            "address at once.", phspPath, needed);
    }

    if (needed > 0) {
        phsp->raw = malloc((size_t)needed);
        if (phsp->raw == NULL) {
            fclose(fp);
            omcFail("ompMC:phsp:outOfMemory",
                "Could not hold the %llu bytes of phase space file %s.",
                needed, phspPath);
        }
    }

    /* The whole file in one read, which also settles how long it actually is
     without asking for its size: a file shorter than the header promised
     comes up short here, and a longer one leaves something behind. */
    unsigned long long nread = needed > 0 ?
        (unsigned long long)fread(phsp->raw, 1, (size_t)needed, fp) : 0;
    int hasMore = fgetc(fp) != EOF;

    fclose(fp);

    /* How many particles there are is a question only the file can answer.
     The header's count is worth reporting a disagreement over, but not worth
     believing: the datasets IAEA publishes include at least one whose
     $PARTICLES: is a particle more than the file holds, and reading the
     particle that is not there would be the real fault. */
    unsigned long long claimed = phsp->nRecords;

    phsp->nRecords = nread/(unsigned long long)phsp->header.recordLength;

    if (phsp->nRecords == 0 && claimed > 0) {
        omcPhspFree(phsp);
        omcFail("ompMC:phsp:fileSizeMismatch",
            "Phase space file %s holds %llu bytes, not even one record of the "
            "%d bytes its header describes.", phspPath, nread,
            phsp->header.recordLength);
    }

    if (nread % (unsigned long long)phsp->header.recordLength != 0) {
        omcLog(OMC_LOG_WARNING, "Phase space file %s ends in the middle of a "
               "record, %llu bytes into one of %d. The part of it that is "
               "there is ignored.", phspPath,
               nread % (unsigned long long)phsp->header.recordLength,
               phsp->header.recordLength);
    }

    if (phsp->nRecords != claimed) {
        omcLog(OMC_LOG_WARNING, "Phase space file %s holds %llu particles, "
               "and its header announces %llu. Going with what the file "
               "holds.", phspPath, (unsigned long long)phsp->nRecords,
               claimed);
    }

    if (hasMore) {
        omcLog(OMC_LOG_WARNING, "Phase space file %s is longer than the %llu "
               "particles its header announces. The rest is ignored.",
               phspPath, claimed);
    }
    else if (phsp->header.checksum != nread) {
        /* The format defines the checksum as the size of the binary file. */
        omcLog(OMC_LOG_WARNING, "Phase space file %s is %llu bytes, and its "
               "header puts the checksum at %llu. One of the two was written "
               "by something that had the other wrong.", phspPath, nread,
               (unsigned long long)phsp->header.checksum);
    }

    /* Look at every record now rather than when it is read, so that a file
     with something wrong in the middle is turned away at the door and
     omcPhspGet() is left with nothing to check. Counting the histories costs
     nothing on the way past. */
    phsp->newHistories = 0;

    for (unsigned long long i = 0; i < phsp->nRecords; i++) {
        const unsigned char *at =
            phsp->raw + i*(unsigned long long)phsp->header.recordLength;
        int type = (int)(signed char)at[0];

        if (type < 0) {
            type = -type;
        }
        if (type < OMC_PHSP_PHOTON || type > OMC_PHSP_PROTON) {
            unsigned long long which = i;
            omcPhspFree(phsp);
            omcFail("ompMC:phsp:badParticleType",
                "Particle %llu of phase space file %s is of type %d, and the "
                "format defines %d to %d.", which, phspPath, type,
                OMC_PHSP_PHOTON, OMC_PHSP_PROTON);
        }

        /* A new history is a negative energy, and the sign bit of that
         little endian float is the top bit of the byte before the next
         quantity. */
        if (at[4] & 0x80) {
            phsp->newHistories++;
        }
    }

    omcLog(OMC_LOG_INFO, "Read %llu particles from phase space file %s.",
           (unsigned long long)phsp->nRecords, phspPath);

    if (phsp->newHistories == 0 && phsp->nRecords > 0) {
        omcLog(OMC_LOG_WARNING, "No particle in phase space file %s marks the "
               "start of a history, so the particles one history left behind "
               "cannot be told from another's. Anything drawing from it has "
               "to treat every particle as its own history, and will "
               "understate its uncertainty by however much they are "
               "correlated.", phspPath);
    }
    else {
        omcLog(OMC_LOG_DETAIL, "Phase space file marks %llu new histories, "
               "%.2f particles each.", (unsigned long long)phsp->newHistories,
               (double)phsp->nRecords/(double)phsp->newHistories);
    }

    return;
}

unsigned long long omcPhspCount(const struct OmcPhsp *phsp) {

    return phsp->raw != NULL ? phsp->nRecords : 0;
}

void omcPhspGet(const struct OmcPhsp *phsp, unsigned long long index,
                struct OmcPhspRecord *record) {

    if (phsp->raw == NULL || index >= phsp->nRecords) {
        omcFail("ompMC:phsp:indexOutOfRange",
            "Asked for particle %llu of a phase space file holding %llu.",
            (unsigned long long)index, omcPhspCount(phsp));
    }

    decodeRecord(&phsp->header,
                 phsp->raw + index*(unsigned long long)phsp->header.recordLength,
                 record);

    return;
}

int omcPhspNext(struct OmcPhsp *phsp, struct OmcPhspRecord *record) {

    if (phsp->cursor >= omcPhspCount(phsp)) {
        return 0;
    }

    omcPhspGet(phsp, phsp->cursor, record);
    phsp->cursor++;

    return 1;
}

void omcPhspRewind(struct OmcPhsp *phsp) {

    phsp->cursor = 0;

    return;
}

void omcPhspFree(struct OmcPhsp *phsp) {

    free(phsp->raw);

    phsp->raw = NULL;
    phsp->nRecords = 0;
    phsp->newHistories = 0;
    phsp->cursor = 0;

    return;
}

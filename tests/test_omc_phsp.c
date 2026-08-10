/******************************************************************************
 ompMC - An OpenMP parallel implementation for Monte Carlo particle transport
 simulations

 Unit tests for the IAEA phase space reader, omc_phsp.

 The datasets are synthesized here rather than committed, the way the input
 deck tests in test_ompmc.c write the decks they parse: a header is built from
 a struct describing what it should say, so that a test can change one field of
 an otherwise valid dataset and pin down what the reader does about it, and the
 binary records are packed by hand so the tests do not depend on the reader
 being able to write what it reads.

 tests/data/small.IAEAheader and .IAEAphsp, a small committed dataset, are read
 by the last test as a check on the fixture builders themselves.
*****************************************************************************/

#include "omc_host.h"
#include "omc_phsp.h"

#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ompmc.c is compiled into the library this links against and refers to this,
 so it has to exist even though no test here reaches it. */
int verbose_flag = 0;

/*******************************************************************************
* Minimal assertion harness, the same one test_ompmc.c uses
*******************************************************************************/
static int tests_run = 0;
static int tests_failed = 0;
static const char *current_test = "";

#define CHECK(cond)                                                           \
    do {                                                                      \
        if (!(cond)) {                                                        \
            printf("  FAIL %s:%d in %s: %s\n",                                \
                   __FILE__, __LINE__, current_test, #cond);                  \
            tests_failed++;                                                   \
        }                                                                     \
    } while (0)

#define CHECK_CLOSE(got, want, tol)                                           \
    do {                                                                      \
        double _g = (got), _w = (want);                                       \
        if (!(fabs(_g - _w) <= (tol))) {                                      \
            printf("  FAIL %s:%d in %s: %s == %.17g, expected %.17g "         \
                   "(tol %g)\n", __FILE__, __LINE__, current_test,            \
                   #got, _g, _w, (double)(tol));                              \
            tests_failed++;                                                   \
        }                                                                     \
    } while (0)

#define RUN(fn)                                                               \
    do {                                                                      \
        current_test = #fn;                                                   \
        tests_run++;                                                          \
        int _before = tests_failed;                                           \
        fn();                                                                 \
        printf("%-52s %s\n", #fn,                                             \
               tests_failed == _before ? "ok" : "FAILED");                    \
    } while (0)

/*******************************************************************************
* A host whose sinks the tests can see
*
* Every failure the reader reports goes through omcFail(), which does not
* return, so a test that wants to reach one has to give it somewhere to go.
* This is the catcher test_ompmc.c installs, and EXPECT_FAIL below wraps the
* setjmp() dance around a call that is supposed to fail.
*******************************************************************************/

static void silentLog(int level, const char *message, void *user) {
    (void)level; (void)message; (void)user;
}

static jmp_buf fail_jmp;
static char fail_id[128];
static int fail_seen;

static void catchingFail(const char *id, const char *message, void *user) {

    (void)message; (void)user;
    snprintf(fail_id, sizeof(fail_id), "%s", id != NULL ? id : "");
    fail_seen = 1;
    longjmp(fail_jmp, 1);
}

static void installFailCatcher(void) {

    fail_seen = 0;
    fail_id[0] = '\0';

    struct OmcHost catcher = {silentLog, catchingFail, NULL};
    omcSetHost(&catcher);
}

/* Run a call that is supposed to fail, and check which failure it was. */
#define EXPECT_FAIL(id, call)                                                 \
    do {                                                                      \
        fail_seen = 0;                                                        \
        fail_id[0] = '\0';                                                    \
        if (setjmp(fail_jmp) == 0) {                                          \
            call;                                                             \
        }                                                                     \
        if (!fail_seen) {                                                     \
            printf("  FAIL %s:%d in %s: %s did not fail, expected %s\n",      \
                   __FILE__, __LINE__, current_test, #call, (id));            \
            tests_failed++;                                                   \
        }                                                                     \
        else if (strcmp(fail_id, (id)) != 0) {                                \
            printf("  FAIL %s:%d in %s: %s failed with %s, expected %s\n",    \
                   __FILE__, __LINE__, current_test, #call, fail_id, (id));   \
            tests_failed++;                                                   \
        }                                                                     \
    } while (0)

/* And its opposite, for a call that is supposed to come back. */
#define EXPECT_OK(call)                                                       \
    do {                                                                      \
        fail_seen = 0;                                                        \
        if (setjmp(fail_jmp) == 0) {                                          \
            call;                                                             \
        }                                                                     \
        if (fail_seen) {                                                      \
            printf("  FAIL %s:%d in %s: %s failed with %s\n",                 \
                   __FILE__, __LINE__, current_test, #call, fail_id);         \
            tests_failed++;                                                   \
        }                                                                     \
    } while (0)

/*******************************************************************************
* Writing datasets to read back
*******************************************************************************/

/* What a header should say. Tests start from validSpec() and change the one
 field they are about, so that everything else stays consistent. */
struct HeaderSpec {
    int fileType;
    int byteOrder;
    int recordLength;           /* 0 : compute it from the flags below */
    long long checksum;         /* -1 : compute it as recordLength*particles */
    unsigned long long particles;
    unsigned long long origHistories;

    int stored[OMC_PHSP_NVARIABLES];
    float constant[OMC_PHSP_NVARIABLES];

    int nExtraFloat;
    int nExtraLong;
    int extraFloatType[OMC_PHSP_MAX_EXTRA];
    int extraLongType[OMC_PHSP_MAX_EXTRA];

    /* Written only when nonzero, the way a real header only lists the types
     it holds. */
    unsigned long long photons, electrons;

    int omitBlock;              /* index of a mandatory block to leave out */
};

/* Everything stored, no extras: a 29 byte record. */
static struct HeaderSpec validSpec(unsigned long long particles) {

    struct HeaderSpec spec;
    memset(&spec, 0, sizeof(spec));

    spec.fileType = 0;
    spec.byteOrder = 1234;
    spec.recordLength = 0;
    spec.checksum = -1;
    spec.particles = particles;
    spec.origHistories = 10;
    spec.omitBlock = -1;

    for (int i = 0; i < OMC_PHSP_NVARIABLES; i++) {
        spec.stored[i] = 1;
    }

    return spec;
}

/* The record length the format defines: a type byte, an energy, the stored
 quantities and the extras. w is never in a record, only its sign is, so a
 stored w costs nothing. */
static int specRecordLength(const struct HeaderSpec *spec) {

    int length = 1 + 4;

    for (int i = 0; i < OMC_PHSP_NVARIABLES; i++) {
        if (i != OMC_PHSP_W && spec->stored[i]) {
            length += 4;
        }
    }

    length += 4*spec->nExtraFloat;
    length += 4*spec->nExtraLong;

    return length;
}

/* Blocks a header cannot do without, in the order writeHeader() writes them,
 so that a test can ask for one to be left out by index. */
enum {
    BLOCK_FILE_TYPE = 0,
    BLOCK_CHECKSUM,
    BLOCK_RECORD_CONTENTS,
    BLOCK_RECORD_CONSTANT,
    BLOCK_RECORD_LENGTH,
    BLOCK_BYTE_ORDER,
    BLOCK_ORIG_HISTORIES,
    BLOCK_PARTICLES
};

static void append(char *dest, size_t cap, const char *fmt, ...) {

    va_list args;
    size_t used = strlen(dest);

    va_start(args, fmt);
    vsnprintf(dest + used, cap - used, fmt, args);
    va_end(args);
}

/* Build the text of a .IAEAheader. Blocks come out in a fixed order here, and
 one test shuffles them afterwards to show the reader does not care. */
static void buildHeader(char *out, size_t cap, const struct HeaderSpec *spec) {

    int recordLength = spec->recordLength > 0 ?
        spec->recordLength : specRecordLength(spec);
    unsigned long long checksum = spec->checksum >= 0 ?
        (unsigned long long)spec->checksum :
        (unsigned long long)recordLength*spec->particles;

    out[0] = '\0';

    append(out, cap, "$IAEA_INDEX:\n1\n\n");
    append(out, cap, "$TITLE:\nsynthetic dataset for the unit tests\n\n");

    if (spec->omitBlock != BLOCK_FILE_TYPE) {
        append(out, cap, "$FILE_TYPE:\n%d\n\n", spec->fileType);
    }
    if (spec->omitBlock != BLOCK_CHECKSUM) {
        append(out, cap, "$CHECKSUM:\n%llu\n\n", checksum);
    }

    if (spec->omitBlock != BLOCK_RECORD_CONTENTS) {
        append(out, cap, "$RECORD_CONTENTS:\n");
        append(out, cap, "    %d     // X is stored ?\n", spec->stored[OMC_PHSP_X]);
        append(out, cap, "    %d     // Y is stored ?\n", spec->stored[OMC_PHSP_Y]);
        append(out, cap, "    %d     // Z is stored ?\n", spec->stored[OMC_PHSP_Z]);
        append(out, cap, "    %d     // U is stored ?\n", spec->stored[OMC_PHSP_U]);
        append(out, cap, "    %d     // V is stored ?\n", spec->stored[OMC_PHSP_V]);
        append(out, cap, "    %d     // W is stored ?\n", spec->stored[OMC_PHSP_W]);
        append(out, cap, "    %d     // Weight is stored ?\n",
               spec->stored[OMC_PHSP_WEIGHT]);
        append(out, cap, "    %d     // Extra floats stored ?\n", spec->nExtraFloat);
        append(out, cap, "    %d     // Extra longs stored ?\n", spec->nExtraLong);

        for (int i = 0; i < spec->nExtraFloat && i < OMC_PHSP_MAX_EXTRA; i++) {
            append(out, cap, "    %d     // Extra float %d\n",
                   spec->extraFloatType[i], i + 1);
        }
        for (int i = 0; i < spec->nExtraLong && i < OMC_PHSP_MAX_EXTRA; i++) {
            append(out, cap, "    %d     // Extra long %d\n",
                   spec->extraLongType[i], i + 1);
        }
        append(out, cap, "\n");
    }

    if (spec->omitBlock != BLOCK_RECORD_CONSTANT) {
        static const char *names[OMC_PHSP_NVARIABLES] =
            {"X", "Y", "Z", "U", "V", "W", "Weight"};

        append(out, cap, "$RECORD_CONSTANT:\n");
        for (int i = 0; i < OMC_PHSP_NVARIABLES; i++) {
            if (!spec->stored[i]) {
                append(out, cap, "   %8.5f     // Constant %s\n",
                       (double)spec->constant[i], names[i]);
            }
        }
        append(out, cap, "\n");
    }

    if (spec->omitBlock != BLOCK_RECORD_LENGTH) {
        append(out, cap, "$RECORD_LENGTH:\n%d\n\n", recordLength);
    }
    if (spec->omitBlock != BLOCK_BYTE_ORDER) {
        append(out, cap, "$BYTE_ORDER:\n%d\n\n", spec->byteOrder);
    }
    if (spec->omitBlock != BLOCK_ORIG_HISTORIES) {
        append(out, cap, "$ORIG_HISTORIES:\n%llu\n\n", spec->origHistories);
    }
    if (spec->omitBlock != BLOCK_PARTICLES) {
        append(out, cap, "$PARTICLES:\n%llu\n\n", spec->particles);
    }

    if (spec->photons > 0) {
        append(out, cap, "$PHOTONS:\n%llu\n\n", spec->photons);
    }
    if (spec->electrons > 0) {
        append(out, cap, "$ELECTRONS:\n%llu\n\n", spec->electrons);
    }

    append(out, cap, "$COORDINATE_SYSTEM_DESCRIPTION:\n"
                     "right handed, origin at the target\n\n");
    append(out, cap, "$TRANSPORT_PARAMETERS:\nnone worth mentioning\n\n");
}

static void datasetPath(char *dest, size_t cap, const char *stem,
                        const char *extension) {

    snprintf(dest, cap, "%s%s", stem, extension);
}

static int writeText(const char *path, const char *contents) {

    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        printf("  FAIL cannot write %s\n", path);
        tests_failed++;
        return 0;
    }
    fputs(contents, fp);
    fclose(fp);

    return 1;
}

static int writeBytes(const char *path, const unsigned char *bytes, size_t n) {

    FILE *fp = fopen(path, "wb");
    if (fp == NULL) {
        printf("  FAIL cannot write %s\n", path);
        tests_failed++;
        return 0;
    }
    if (n > 0) {
        fwrite(bytes, 1, n, fp);
    }
    fclose(fp);

    return 1;
}

static int writeHeaderFile(const char *stem, const struct HeaderSpec *spec) {

    char path[256];
    char text[8192];

    buildHeader(text, sizeof(text), spec);
    datasetPath(path, sizeof(path), stem, ".IAEAheader");

    return writeText(path, text);
}

static int writePhspFile(const char *stem, const unsigned char *bytes,
                         size_t n) {

    char path[256];

    datasetPath(path, sizeof(path), stem, ".IAEAphsp");

    return writeBytes(path, bytes, n);
}

static void removeDataset(const char *stem) {

    char path[256];

    datasetPath(path, sizeof(path), stem, ".IAEAheader");
    remove(path);
    datasetPath(path, sizeof(path), stem, ".IAEAphsp");
    remove(path);
}

/*******************************************************************************
* Packing binary records
*
* Little endian, byte by byte, so that the tests write the same bytes whatever
* the machine running them does natively. The reader has to do the same thing
* in reverse, and the two agreeing on a big endian machine is the point.
*******************************************************************************/

static size_t putByte(unsigned char *dst, int value) {

    dst[0] = (unsigned char)(value & 0xFF);
    return 1;
}

static size_t putLeFloat(unsigned char *dst, float value) {

    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));

    dst[0] = (unsigned char)(bits & 0xFF);
    dst[1] = (unsigned char)((bits >> 8) & 0xFF);
    dst[2] = (unsigned char)((bits >> 16) & 0xFF);
    dst[3] = (unsigned char)((bits >> 24) & 0xFF);

    return 4;
}

static size_t putLeInt32(unsigned char *dst, int32_t value) {

    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));

    dst[0] = (unsigned char)(bits & 0xFF);
    dst[1] = (unsigned char)((bits >> 8) & 0xFF);
    dst[2] = (unsigned char)((bits >> 16) & 0xFF);
    dst[3] = (unsigned char)((bits >> 24) & 0xFF);

    return 4;
}

/* One particle, as a record of the layout a header spec describes. The type
 is signed: a negative one is how the format says w points backwards, and a
 negative energy is how it says a new history starts here. */
struct RecordSpec {
    int type;                   /* signed, as it goes into the file */
    float energy;               /* signed, as it goes into the file */
    float value[OMC_PHSP_NVARIABLES];
    float extraFloat[OMC_PHSP_MAX_EXTRA];
    int32_t extraLong[OMC_PHSP_MAX_EXTRA];
};

static struct RecordSpec validRecord(void) {

    struct RecordSpec record;
    memset(&record, 0, sizeof(record));

    record.type = OMC_PHSP_PHOTON;
    record.energy = 2.5f;
    record.value[OMC_PHSP_X] = 1.0f;
    record.value[OMC_PHSP_Y] = -2.0f;
    record.value[OMC_PHSP_Z] = 50.0f;
    record.value[OMC_PHSP_U] = 0.1f;
    record.value[OMC_PHSP_V] = 0.2f;
    record.value[OMC_PHSP_WEIGHT] = 0.75f;

    return record;
}

static size_t putRecord(unsigned char *dst, const struct HeaderSpec *spec,
                        const struct RecordSpec *record) {

    size_t at = 0;

    at += putByte(dst + at, record->type);
    at += putLeFloat(dst + at, record->energy);

    for (int i = 0; i < OMC_PHSP_NVARIABLES; i++) {
        if (i != OMC_PHSP_W && spec->stored[i]) {
            at += putLeFloat(dst + at, record->value[i]);
        }
    }

    for (int i = 0; i < spec->nExtraFloat; i++) {
        at += putLeFloat(dst + at, record->extraFloat[i]);
    }
    for (int i = 0; i < spec->nExtraLong; i++) {
        at += putLeInt32(dst + at, record->extraLong[i]);
    }

    return at;
}

static size_t packRecords(unsigned char *dst, const struct HeaderSpec *spec,
                          const struct RecordSpec *records, size_t nrecords) {

    size_t at = 0;

    for (size_t i = 0; i < nrecords; i++) {
        at += putRecord(dst + at, spec, &records[i]);
    }

    return at;
}

/* Write a whole dataset: the header, and the records packed to match it. */
static int writeDataset(const char *stem, const struct HeaderSpec *spec,
                        const struct RecordSpec *records, size_t nrecords) {

    unsigned char bytes[4096];
    size_t at;

    if (!writeHeaderFile(stem, spec)) {
        return 0;
    }

    at = packRecords(bytes, spec, records, nrecords);

    return writePhspFile(stem, bytes, at);
}

/*******************************************************************************
* The header
*******************************************************************************/

static void test_header_open_failure(void) {

    struct OmcPhspHeader header;

    EXPECT_FAIL("ompMC:phsp:openFailed",
                omcPhspHeaderFromFile(&header, "no_such_dataset_at_all"));
}

static void test_header_reads_scalar_keywords(void) {

    const char *stem = "phsp_scalars";
    struct HeaderSpec spec = validSpec(3);
    struct OmcPhspHeader header;

    if (!writeHeaderFile(stem, &spec)) {
        return;
    }

    EXPECT_OK(omcPhspHeaderFromFile(&header, stem));

    CHECK(header.fileType == 0);
    CHECK(header.byteOrder == 1234);
    CHECK(header.recordLength == 29);
    CHECK(header.particles == 3);
    CHECK(header.origHistories == 10);
    CHECK(header.checksum == 87);

    removeDataset(stem);
}

/* A header is a set of blocks, not a sequence of them: the reader looks each
 keyword up rather than expecting them in the order the IAEA writer happens to
 emit. This shuffles the blocks and expects the same answers. */
static void test_header_blocks_in_any_order(void) {

    const char *stem = "phsp_shuffled";
    struct OmcPhspHeader header;
    const char *text =
        "$PARTICLES:\n7\n\n"
        "$BYTE_ORDER:\n1234\n\n"
        "$RECORD_CONSTANT:\n\n"
        "$ORIG_HISTORIES:\n99\n\n"
        "$RECORD_LENGTH:\n29\n\n"
        "$RECORD_CONTENTS:\n1\n1\n1\n1\n1\n1\n1\n0\n0\n\n"
        "$CHECKSUM:\n203\n\n"
        "$FILE_TYPE:\n0\n\n";
    char path[256];

    datasetPath(path, sizeof(path), stem, ".IAEAheader");
    if (!writeText(path, text)) {
        return;
    }

    EXPECT_OK(omcPhspHeaderFromFile(&header, stem));

    CHECK(header.particles == 7);
    CHECK(header.origHistories == 99);
    CHECK(header.recordLength == 29);
    CHECK(header.stored[OMC_PHSP_X] == 1);

    removeDataset(stem);
}

/* The IAEA writer puts a C++ comment after every value it writes, and the
 reference reader also allows C style comments and blank lines. */
static void test_header_strips_comments_and_blanks(void) {

    const char *stem = "phsp_comments";
    struct OmcPhspHeader header;
    const char *text =
        "$FILE_TYPE:\n"
        "\n"
        "   0    // a phase space file, not an event generator\n\n"
        "$CHECKSUM:\n87 // 29 bytes times 3 particles\n\n"
        "$RECORD_CONTENTS:\n"
        "/* everything is stored */\n"
        "1\n1\n1\n1\n1\n1\n1\n"
        "   0   // no extra floats\n"
        "   0   // no extra longs\n\n"
        "$RECORD_CONSTANT:\n\n"
        "$RECORD_LENGTH:\n  29  \n\n"
        "$BYTE_ORDER:\n1234\n\n"
        "$ORIG_HISTORIES:\n10\n\n"
        "$PARTICLES:\n3\n\n";
    char path[256];

    datasetPath(path, sizeof(path), stem, ".IAEAheader");
    if (!writeText(path, text)) {
        return;
    }

    EXPECT_OK(omcPhspHeaderFromFile(&header, stem));

    CHECK(header.fileType == 0);
    CHECK(header.checksum == 87);
    CHECK(header.recordLength == 29);
    CHECK(header.nExtraFloat == 0);
    CHECK(header.nExtraLong == 0);

    removeDataset(stem);
}

static void test_header_record_contents_flags(void) {

    const char *stem = "phsp_flags";
    struct HeaderSpec spec = validSpec(2);
    struct OmcPhspHeader header;

    spec.stored[OMC_PHSP_Z] = 0;
    spec.stored[OMC_PHSP_WEIGHT] = 0;
    spec.constant[OMC_PHSP_Z] = 15.0f;
    spec.constant[OMC_PHSP_WEIGHT] = 1.0f;

    if (!writeHeaderFile(stem, &spec)) {
        return;
    }

    EXPECT_OK(omcPhspHeaderFromFile(&header, stem));

    CHECK(header.stored[OMC_PHSP_X] == 1);
    CHECK(header.stored[OMC_PHSP_Y] == 1);
    CHECK(header.stored[OMC_PHSP_Z] == 0);
    CHECK(header.stored[OMC_PHSP_U] == 1);
    CHECK(header.stored[OMC_PHSP_V] == 1);
    CHECK(header.stored[OMC_PHSP_W] == 1);
    CHECK(header.stored[OMC_PHSP_WEIGHT] == 0);
    CHECK(header.nExtraFloat == 0);
    CHECK(header.nExtraLong == 0);

    /* Two fewer stored quantities than the 29 byte record of validSpec(). */
    CHECK(header.recordLength == 21);

    removeDataset(stem);
}

/* $RECORD_CONSTANT holds a value only for the quantities $RECORD_CONTENTS
 said are not stored, so which line means what depends on the block before
 it. */
static void test_header_record_constants(void) {

    const char *stem = "phsp_constants";
    struct HeaderSpec spec = validSpec(2);
    struct OmcPhspHeader header;

    spec.stored[OMC_PHSP_Z] = 0;
    spec.stored[OMC_PHSP_WEIGHT] = 0;
    spec.constant[OMC_PHSP_Z] = 15.0f;
    spec.constant[OMC_PHSP_WEIGHT] = 1.0f;

    if (!writeHeaderFile(stem, &spec)) {
        return;
    }

    EXPECT_OK(omcPhspHeaderFromFile(&header, stem));

    CHECK_CLOSE(header.constant[OMC_PHSP_Z], 15.0, 1e-6);
    CHECK_CLOSE(header.constant[OMC_PHSP_WEIGHT], 1.0, 1e-6);

    removeDataset(stem);
}

static void test_header_extra_type_codes(void) {

    const char *stem = "phsp_extras";
    struct HeaderSpec spec = validSpec(1);
    struct OmcPhspHeader header;

    spec.nExtraFloat = 1;
    spec.nExtraLong = 1;
    spec.extraFloatType[0] = OMC_PHSP_FLOAT_ZLAST;
    spec.extraLongType[0] = OMC_PHSP_LONG_LATCH;

    if (!writeHeaderFile(stem, &spec)) {
        return;
    }

    EXPECT_OK(omcPhspHeaderFromFile(&header, stem));

    CHECK(header.nExtraFloat == 1);
    CHECK(header.nExtraLong == 1);
    CHECK(header.extraFloatType[0] == OMC_PHSP_FLOAT_ZLAST);
    CHECK(header.extraLongType[0] == OMC_PHSP_LONG_LATCH);

    /* The 29 byte record of validSpec(), plus a float and a long. */
    CHECK(header.recordLength == 37);

    removeDataset(stem);
}

/* The trap in the format: w is flagged as stored like everything else, but a
 record never holds it -- only the sign, in the sign of the particle type. A
 record length computed by counting the flags would be four bytes too long,
 and every particle after the first would be read from the wrong offset. */
static void test_header_w_flag_adds_no_bytes(void) {

    const char *stem = "phsp_wflag";
    struct HeaderSpec stored = validSpec(1);
    struct HeaderSpec constant = validSpec(1);
    struct OmcPhspHeader header;

    if (!writeHeaderFile(stem, &stored)) {
        return;
    }
    EXPECT_OK(omcPhspHeaderFromFile(&header, stem));
    CHECK(header.recordLength == 29);

    /* Not storing w does not make the record shorter either. */
    constant.stored[OMC_PHSP_W] = 0;
    constant.constant[OMC_PHSP_W] = 1.0f;

    if (!writeHeaderFile(stem, &constant)) {
        return;
    }
    EXPECT_OK(omcPhspHeaderFromFile(&header, stem));
    CHECK(header.recordLength == 29);

    removeDataset(stem);
}

static void test_header_record_length_mismatch_fails(void) {

    const char *stem = "phsp_badlength";
    struct HeaderSpec spec = validSpec(3);
    struct OmcPhspHeader header;

    /* What the flags add up to is 29, and the checksum follows the claim so
     that the record length is the only thing wrong. */
    spec.recordLength = 33;

    if (!writeHeaderFile(stem, &spec)) {
        return;
    }

    EXPECT_FAIL("ompMC:phsp:recordLengthMismatch",
                omcPhspHeaderFromFile(&header, stem));

    removeDataset(stem);
}

static void test_header_missing_keyword_fails(void) {

    const char *stem = "phsp_missing";
    struct OmcPhspHeader header;
    const int mandatory[] = {BLOCK_FILE_TYPE, BLOCK_CHECKSUM,
                             BLOCK_RECORD_CONTENTS, BLOCK_RECORD_LENGTH,
                             BLOCK_BYTE_ORDER, BLOCK_ORIG_HISTORIES,
                             BLOCK_PARTICLES};

    for (size_t i = 0; i < sizeof(mandatory)/sizeof(mandatory[0]); i++) {
        struct HeaderSpec spec = validSpec(3);
        spec.omitBlock = mandatory[i];

        if (!writeHeaderFile(stem, &spec)) {
            return;
        }

        EXPECT_FAIL("ompMC:phsp:missingKeyword",
                    omcPhspHeaderFromFile(&header, stem));
    }

    removeDataset(stem);
}

static void test_header_rejects_big_endian_and_event_files(void) {

    const char *stem = "phsp_unsupported";
    struct OmcPhspHeader header;
    struct HeaderSpec bigEndian = validSpec(3);
    struct HeaderSpec generator = validSpec(3);

    bigEndian.byteOrder = 4321;
    if (!writeHeaderFile(stem, &bigEndian)) {
        return;
    }
    EXPECT_FAIL("ompMC:phsp:unsupportedByteOrder",
                omcPhspHeaderFromFile(&header, stem));

    generator.fileType = 1;
    if (!writeHeaderFile(stem, &generator)) {
        return;
    }
    EXPECT_FAIL("ompMC:phsp:unsupportedFileType",
                omcPhspHeaderFromFile(&header, stem));

    removeDataset(stem);
}

/* A header describing an event generator does not carry the blocks a phase
 space file does -- it names an input file instead of counting particles. So
 what it is has to be settled before the rest of it is asked for, or the
 complaint is that a file of the wrong kind is missing a block that file has
 no business holding. */
static void test_event_generator_header_rejected_before_anything_else(void) {

    const char *stem = "phsp_generator";
    struct OmcPhspHeader header;
    const char *text =
        "$FILE_TYPE:\n1\n\n"
        "$INPUT_FILE_FOR_EVENT_GENERATOR:\nsome_generator.inp\n\n"
        "$IAEA_INDEX:\n7\n\n";
    char path[256];

    datasetPath(path, sizeof(path), stem, ".IAEAheader");
    if (!writeText(path, text)) {
        return;
    }

    EXPECT_FAIL("ompMC:phsp:unsupportedFileType",
                omcPhspHeaderFromFile(&header, stem));

    removeDataset(stem);
}

/* The checksum is the size the binary file should have, and the header
 reader never opens that file, so it has nothing to hold the checksum
 against and keeps it as it stands. In particular it is NOT the record
 length times the particle count: the phase space IAEA publishes as index
 700 has a checksum matching its file exactly and a particle count one
 particle above it, and a reader taking those two to agree turns a real
 dataset away. */
static void test_header_keeps_a_checksum_it_cannot_check(void) {

    const char *stem = "phsp_sum";
    struct HeaderSpec spec = validSpec(3);
    struct OmcPhspHeader header;

    spec.checksum = 58;         /* 3 records of 29 bytes would be 87 */

    if (!writeHeaderFile(stem, &spec)) {
        return;
    }

    EXPECT_OK(omcPhspHeaderFromFile(&header, stem));

    CHECK(header.checksum == 58);
    CHECK(header.particles == 3);

    removeDataset(stem);
}

static void test_header_too_many_extras_fails(void) {

    const char *stem = "phsp_manyextras";
    struct HeaderSpec spec = validSpec(1);
    struct OmcPhspHeader header;

    /* buildHeader() writes the count but only OMC_PHSP_MAX_EXTRA type codes,
     which is fine: the reader gives up on the count. */
    spec.nExtraFloat = OMC_PHSP_MAX_EXTRA + 1;

    if (!writeHeaderFile(stem, &spec)) {
        return;
    }

    EXPECT_FAIL("ompMC:phsp:tooManyExtras",
                omcPhspHeaderFromFile(&header, stem));

    removeDataset(stem);
}

/* The IAEA convention is a base name shared by two files, but a caller who
 has one of the two file names in hand should not have to trim it. */
static void test_header_path_with_extension_accepted(void) {

    const char *stem = "phsp_extension";
    struct HeaderSpec spec = validSpec(3);
    struct OmcPhspHeader header;
    char path[256];

    if (!writeHeaderFile(stem, &spec)) {
        return;
    }

    datasetPath(path, sizeof(path), stem, ".IAEAheader");
    EXPECT_OK(omcPhspHeaderFromFile(&header, path));
    CHECK(header.particles == 3);

    datasetPath(path, sizeof(path), stem, ".IAEAphsp");
    EXPECT_OK(omcPhspHeaderFromFile(&header, path));
    CHECK(header.particles == 3);

    removeDataset(stem);
}

static void test_header_per_type_counts(void) {

    const char *stem = "phsp_types";
    struct HeaderSpec spec = validSpec(3);
    struct OmcPhspHeader header;

    spec.photons = 2;
    spec.electrons = 1;

    if (!writeHeaderFile(stem, &spec)) {
        return;
    }

    EXPECT_OK(omcPhspHeaderFromFile(&header, stem));

    CHECK(header.typeCount[OMC_PHSP_PHOTON - 1] == 2);
    CHECK(header.typeCount[OMC_PHSP_ELECTRON - 1] == 1);
    CHECK(header.typeCount[OMC_PHSP_POSITRON - 1] == 0);
    CHECK(header.typeCount[OMC_PHSP_NEUTRON - 1] == 0);
    CHECK(header.typeCount[OMC_PHSP_PROTON - 1] == 0);

    removeDataset(stem);
}

/*******************************************************************************
* The binary file
*******************************************************************************/

static void test_load_single_record_all_stored(void) {

    const char *stem = "phsp_single";
    struct HeaderSpec spec = validSpec(1);
    struct RecordSpec record = validRecord();
    struct OmcPhsp phsp;
    struct OmcPhspRecord got;

    if (!writeDataset(stem, &spec, &record, 1)) {
        return;
    }

    EXPECT_OK(omcPhspFromFile(&phsp, stem));

    CHECK(omcPhspCount(&phsp) == 1);

    EXPECT_OK(omcPhspGet(&phsp, 0, &got));

    CHECK(got.type == OMC_PHSP_PHOTON);
    CHECK(got.newHistory == 0);
    CHECK_CLOSE(got.energy, 2.5, 1e-6);
    CHECK_CLOSE(got.x, 1.0, 1e-6);
    CHECK_CLOSE(got.y, -2.0, 1e-6);
    CHECK_CLOSE(got.z, 50.0, 1e-6);
    CHECK_CLOSE(got.u, 0.1, 1e-6);
    CHECK_CLOSE(got.v, 0.2, 1e-6);
    CHECK_CLOSE(got.w, sqrt(1.0 - 0.01 - 0.04), 1e-6);
    CHECK_CLOSE(got.weight, 0.75, 1e-6);

    omcPhspFree(&phsp);
    removeDataset(stem);
}

/* A quantity the header calls constant is not in the record at all; every
 particle gets the value the header gave. */
static void test_constants_substituted(void) {

    const char *stem = "phsp_subst";
    struct HeaderSpec spec = validSpec(1);
    struct RecordSpec record = validRecord();
    struct OmcPhsp phsp;
    struct OmcPhspRecord got;

    spec.stored[OMC_PHSP_Z] = 0;
    spec.stored[OMC_PHSP_WEIGHT] = 0;
    spec.constant[OMC_PHSP_Z] = 15.0f;
    spec.constant[OMC_PHSP_WEIGHT] = 1.0f;

    /* What the record still carries for them is never written, so these
     values are here only to show they are not what comes back. */
    record.value[OMC_PHSP_Z] = -1.0f;
    record.value[OMC_PHSP_WEIGHT] = -1.0f;

    if (!writeDataset(stem, &spec, &record, 1)) {
        return;
    }

    EXPECT_OK(omcPhspFromFile(&phsp, stem));
    EXPECT_OK(omcPhspGet(&phsp, 0, &got));

    CHECK_CLOSE(got.z, 15.0, 1e-6);
    CHECK_CLOSE(got.weight, 1.0, 1e-6);

    /* The stored quantities around them still land in the right place. */
    CHECK_CLOSE(got.x, 1.0, 1e-6);
    CHECK_CLOSE(got.y, -2.0, 1e-6);
    CHECK_CLOSE(got.u, 0.1, 1e-6);
    CHECK_CLOSE(got.v, 0.2, 1e-6);

    omcPhspFree(&phsp);
    removeDataset(stem);
}

/* A constant w is taken as it stands. Only a w the header calls stored is
 recovered from u and v, which is the whole difference the flag makes: w is
 never in a record either way. */
static void test_w_constant_not_reconstructed(void) {

    const char *stem = "phsp_wconst";
    struct HeaderSpec spec = validSpec(1);
    struct RecordSpec record = validRecord();
    struct OmcPhsp phsp;
    struct OmcPhspRecord got;

    spec.stored[OMC_PHSP_W] = 0;
    spec.constant[OMC_PHSP_W] = 1.0f;

    /* Reconstruction would give 0.8 here. */
    record.value[OMC_PHSP_U] = 0.6f;
    record.value[OMC_PHSP_V] = 0.0f;

    if (!writeDataset(stem, &spec, &record, 1)) {
        return;
    }

    EXPECT_OK(omcPhspFromFile(&phsp, stem));
    EXPECT_OK(omcPhspGet(&phsp, 0, &got));

    CHECK_CLOSE(got.w, 1.0, 1e-6);

    omcPhspFree(&phsp);
    removeDataset(stem);
}

/* A new history is announced by writing the energy negative. */
static void test_negative_energy_marks_new_history(void) {

    const char *stem = "phsp_newhist";
    struct HeaderSpec spec = validSpec(2);
    struct RecordSpec records[2];
    struct OmcPhsp phsp;
    struct OmcPhspRecord got;

    records[0] = validRecord();
    records[1] = validRecord();
    records[0].energy = -1.25f;
    records[1].energy = 1.25f;

    if (!writeDataset(stem, &spec, records, 2)) {
        return;
    }

    EXPECT_OK(omcPhspFromFile(&phsp, stem));

    EXPECT_OK(omcPhspGet(&phsp, 0, &got));
    CHECK(got.newHistory == 1);
    CHECK_CLOSE(got.energy, 1.25, 1e-6);

    EXPECT_OK(omcPhspGet(&phsp, 1, &got));
    CHECK(got.newHistory == 0);
    CHECK_CLOSE(got.energy, 1.25, 1e-6);

    omcPhspFree(&phsp);
    removeDataset(stem);
}

/* The histories are counted while loading, and a file marking none of them
 is a real thing rather than an empty one: the Varian TrueBeam 6MV phase
 space IAEA publishes marks not one of its 52 million particles, which leaves
 anything drawing from it unable to tell the particles of one history from
 another's. Worth knowing before sampling, so it is counted and said. */
static void test_new_histories_are_counted(void) {

    const char *stem = "phsp_histories";
    struct HeaderSpec spec = validSpec(4);
    struct RecordSpec records[4];
    struct OmcPhsp phsp;

    for (int i = 0; i < 4; i++) {
        records[i] = validRecord();
    }
    records[0].energy = -1.0f;
    records[1].energy = 2.0f;
    records[2].energy = -3.0f;
    records[3].energy = 4.0f;

    if (!writeDataset(stem, &spec, records, 4)) {
        return;
    }

    EXPECT_OK(omcPhspFromFile(&phsp, stem));
    CHECK(phsp.newHistories == 2);
    omcPhspFree(&phsp);

    /* A file that marks none of them says zero, and says it about a file
     that is not empty. */
    for (int i = 0; i < 4; i++) {
        records[i].energy = 1.0f + (float)i;
    }

    if (!writeDataset(stem, &spec, records, 4)) {
        return;
    }

    EXPECT_OK(omcPhspFromFile(&phsp, stem));
    CHECK(phsp.newHistories == 0);
    CHECK(omcPhspCount(&phsp) == 4);
    omcPhspFree(&phsp);

    removeDataset(stem);
}

/* And a particle heading back the way it came by writing the particle type
 negative, the only place the sign of w is kept. */
static void test_negative_type_gives_negative_w(void) {

    const char *stem = "phsp_backwards";
    struct HeaderSpec spec = validSpec(1);
    struct RecordSpec record = validRecord();
    struct OmcPhsp phsp;
    struct OmcPhspRecord got;

    record.type = -OMC_PHSP_ELECTRON;

    if (!writeDataset(stem, &spec, &record, 1)) {
        return;
    }

    EXPECT_OK(omcPhspFromFile(&phsp, stem));
    EXPECT_OK(omcPhspGet(&phsp, 0, &got));

    CHECK(got.type == OMC_PHSP_ELECTRON);
    CHECK_CLOSE(got.w, -sqrt(1.0 - 0.01 - 0.04), 1e-6);

    omcPhspFree(&phsp);
    removeDataset(stem);
}

/* u and v are written as floats, and rounding can leave them describing a
 direction that is a little more than a unit vector. The format's own reader
 scales them back onto the unit circle and calls w zero rather than taking
 the square root of a negative number, and so does this one. */
static void test_uv_overflow_normalized(void) {

    const char *stem = "phsp_overflow";
    struct HeaderSpec spec = validSpec(1);
    struct RecordSpec record = validRecord();
    struct OmcPhsp phsp;
    struct OmcPhspRecord got;

    record.value[OMC_PHSP_U] = 0.8f;
    record.value[OMC_PHSP_V] = 0.8f;

    if (!writeDataset(stem, &spec, &record, 1)) {
        return;
    }

    EXPECT_OK(omcPhspFromFile(&phsp, stem));
    EXPECT_OK(omcPhspGet(&phsp, 0, &got));

    CHECK_CLOSE(got.u, 0.8/sqrt(1.28), 1e-6);
    CHECK_CLOSE(got.v, 0.8/sqrt(1.28), 1e-6);
    CHECK_CLOSE(got.w, 0.0, 1e-6);
    CHECK_CLOSE(got.u*got.u + got.v*got.v + got.w*got.w, 1.0, 1e-6);

    omcPhspFree(&phsp);
    removeDataset(stem);
}

static void test_extra_floats_and_longs_decoded(void) {

    const char *stem = "phsp_extradata";
    struct HeaderSpec spec = validSpec(1);
    struct RecordSpec record = validRecord();
    struct OmcPhsp phsp;
    struct OmcPhspRecord got;

    spec.nExtraFloat = 1;
    spec.nExtraLong = 1;
    spec.extraFloatType[0] = OMC_PHSP_FLOAT_ZLAST;
    spec.extraLongType[0] = OMC_PHSP_LONG_LATCH;

    record.extraFloat[0] = -33.5f;
    record.extraLong[0] = -7;   /* negative: the longs are signed */

    if (!writeDataset(stem, &spec, &record, 1)) {
        return;
    }

    EXPECT_OK(omcPhspFromFile(&phsp, stem));
    EXPECT_OK(omcPhspGet(&phsp, 0, &got));

    CHECK(phsp.header.extraFloatType[0] == OMC_PHSP_FLOAT_ZLAST);
    CHECK(phsp.header.extraLongType[0] == OMC_PHSP_LONG_LATCH);
    CHECK_CLOSE(got.extraFloat[0], -33.5, 1e-6);
    CHECK(got.extraLong[0] == -7);

    /* The particle itself is still read from the right place. */
    CHECK_CLOSE(got.weight, 0.75, 1e-6);

    omcPhspFree(&phsp);
    removeDataset(stem);
}

/* Read out of order, which is what sampling from a phase space file rather
 than replaying it needs. */
static void test_indexed_random_access(void) {

    const char *stem = "phsp_indexed";
    struct HeaderSpec spec = validSpec(3);
    struct RecordSpec records[3];
    struct OmcPhsp phsp;
    struct OmcPhspRecord got;

    for (int i = 0; i < 3; i++) {
        records[i] = validRecord();
        records[i].energy = (float)(i + 1);
    }

    if (!writeDataset(stem, &spec, records, 3)) {
        return;
    }

    EXPECT_OK(omcPhspFromFile(&phsp, stem));

    CHECK(omcPhspCount(&phsp) == 3);

    EXPECT_OK(omcPhspGet(&phsp, 2, &got));
    CHECK_CLOSE(got.energy, 3.0, 1e-6);
    EXPECT_OK(omcPhspGet(&phsp, 0, &got));
    CHECK_CLOSE(got.energy, 1.0, 1e-6);
    EXPECT_OK(omcPhspGet(&phsp, 1, &got));
    CHECK_CLOSE(got.energy, 2.0, 1e-6);

    omcPhspFree(&phsp);
    removeDataset(stem);
}

/* Or in order, popping the file like a stack until it runs out. */
static void test_next_pops_then_stops(void) {

    const char *stem = "phsp_pop";
    struct HeaderSpec spec = validSpec(3);
    struct RecordSpec records[3];
    struct OmcPhsp phsp;
    struct OmcPhspRecord got;

    for (int i = 0; i < 3; i++) {
        records[i] = validRecord();
        records[i].energy = (float)(i + 1);
    }

    if (!writeDataset(stem, &spec, records, 3)) {
        return;
    }

    EXPECT_OK(omcPhspFromFile(&phsp, stem));

    for (int i = 0; i < 3; i++) {
        CHECK(omcPhspNext(&phsp, &got) == 1);
        CHECK_CLOSE(got.energy, (double)(i + 1), 1e-6);
    }

    CHECK(omcPhspNext(&phsp, &got) == 0);
    CHECK(omcPhspNext(&phsp, &got) == 0);

    omcPhspRewind(&phsp);
    CHECK(omcPhspNext(&phsp, &got) == 1);
    CHECK_CLOSE(got.energy, 1.0, 1e-6);

    omcPhspFree(&phsp);
    removeDataset(stem);
}

static void test_get_out_of_range_fails(void) {

    const char *stem = "phsp_range";
    struct HeaderSpec spec = validSpec(3);
    struct RecordSpec records[3];
    struct OmcPhsp phsp;
    struct OmcPhspRecord got;

    for (int i = 0; i < 3; i++) {
        records[i] = validRecord();
    }

    if (!writeDataset(stem, &spec, records, 3)) {
        return;
    }

    EXPECT_OK(omcPhspFromFile(&phsp, stem));

    EXPECT_FAIL("ompMC:phsp:indexOutOfRange", omcPhspGet(&phsp, 3, &got));

    omcPhspFree(&phsp);
    removeDataset(stem);
}

/* How many particles there are is a question only the binary file can
 answer. The phase space IAEA publishes as index 700 announces one particle
 more than it holds, so a header and a file disagreeing has to be something
 the reader reports and works around rather than refuses -- and what it works
 around to is the file, since the particle the header adds is not there to be
 read. */
static void test_header_overcounting_particles_is_survivable(void) {

    const char *stem = "phsp_overcount";
    struct HeaderSpec spec = validSpec(3);
    struct RecordSpec records[2];
    unsigned char bytes[256];
    struct OmcPhsp phsp;
    struct OmcPhspRecord got;
    size_t packed;

    records[0] = validRecord();
    records[1] = validRecord();
    records[0].energy = 1.0f;
    records[1].energy = 2.0f;

    /* The header counts three particles and the checksum matches the two the
     file actually holds, which is the shape the published file has. */
    spec.checksum = 2*29;

    if (!writeHeaderFile(stem, &spec)) {
        return;
    }

    packed = packRecords(bytes, &spec, records, 2);
    if (!writePhspFile(stem, bytes, packed)) {
        return;
    }

    EXPECT_OK(omcPhspFromFile(&phsp, stem));

    CHECK(phsp.header.particles == 3);
    CHECK(omcPhspCount(&phsp) == 2);

    EXPECT_OK(omcPhspGet(&phsp, 1, &got));
    CHECK_CLOSE(got.energy, 2.0, 1e-6);

    /* And the particle the header made up is not there to be read. */
    EXPECT_FAIL("ompMC:phsp:indexOutOfRange", omcPhspGet(&phsp, 2, &got));

    omcPhspFree(&phsp);
    removeDataset(stem);
}

/* A file cut off in the middle of a record keeps the records that made it. */
static void test_partial_last_record_is_dropped(void) {

    const char *stem = "phsp_short";
    struct HeaderSpec spec = validSpec(2);
    struct RecordSpec records[2];
    unsigned char bytes[256];
    struct OmcPhsp phsp;
    size_t packed;

    records[0] = validRecord();
    records[1] = validRecord();

    if (!writeHeaderFile(stem, &spec)) {
        return;
    }

    packed = packRecords(bytes, &spec, records, 2);
    if (!writePhspFile(stem, bytes, packed - 4)) {
        return;
    }

    EXPECT_OK(omcPhspFromFile(&phsp, stem));

    CHECK(omcPhspCount(&phsp) == 1);

    omcPhspFree(&phsp);
    removeDataset(stem);
}

/* But a file without a whole record in it has nothing to offer. */
static void test_phsp_file_without_a_whole_record_fails(void) {

    const char *stem = "phsp_stub";
    struct HeaderSpec spec = validSpec(2);
    struct RecordSpec record = validRecord();
    unsigned char bytes[256];
    struct OmcPhsp phsp;

    if (!writeHeaderFile(stem, &spec)) {
        return;
    }

    packRecords(bytes, &spec, &record, 1);
    if (!writePhspFile(stem, bytes, 10)) {
        return;
    }

    EXPECT_FAIL("ompMC:phsp:fileSizeMismatch", omcPhspFromFile(&phsp, stem));

    removeDataset(stem);
}

/* Bytes after the last particle, on the other hand, cost nothing: say so and
 read the particles the header promised. */
static void test_trailing_bytes_warn_but_load(void) {

    const char *stem = "phsp_trailing";
    struct HeaderSpec spec = validSpec(1);
    struct RecordSpec record = validRecord();
    unsigned char bytes[256];
    struct OmcPhsp phsp;
    struct OmcPhspRecord got;
    size_t packed;

    if (!writeHeaderFile(stem, &spec)) {
        return;
    }

    packed = packRecords(bytes, &spec, &record, 1);
    bytes[packed] = 0xDE;
    bytes[packed + 1] = 0xAD;
    bytes[packed + 2] = 0xBE;

    if (!writePhspFile(stem, bytes, packed + 3)) {
        return;
    }

    EXPECT_OK(omcPhspFromFile(&phsp, stem));

    CHECK(omcPhspCount(&phsp) == 1);
    EXPECT_OK(omcPhspGet(&phsp, 0, &got));
    CHECK_CLOSE(got.energy, 2.5, 1e-6);

    omcPhspFree(&phsp);
    removeDataset(stem);
}

/* Every record is looked at while loading rather than when it is read, so
 that a file with something wrong in the middle of it is rejected at the door
 and omcPhspGet() has nothing left to check. */
static void test_bad_type_byte_fails_at_load(void) {

    const char *stem = "phsp_badtype";
    struct HeaderSpec spec = validSpec(2);
    struct RecordSpec records[2];
    struct OmcPhsp phsp;

    records[0] = validRecord();
    records[1] = validRecord();
    records[1].type = 6;        /* the format defines 1 to 5 */

    if (!writeDataset(stem, &spec, records, 2)) {
        return;
    }

    EXPECT_FAIL("ompMC:phsp:badParticleType", omcPhspFromFile(&phsp, stem));

    removeDataset(stem);
}

static void test_free_is_idempotent(void) {

    const char *stem = "phsp_free";
    struct HeaderSpec spec = validSpec(1);
    struct RecordSpec record = validRecord();
    struct OmcPhsp phsp;

    if (!writeDataset(stem, &spec, &record, 1)) {
        return;
    }

    EXPECT_OK(omcPhspFromFile(&phsp, stem));

    omcPhspFree(&phsp);
    CHECK(phsp.raw == NULL);
    CHECK(omcPhspCount(&phsp) == 0);

    omcPhspFree(&phsp);
    CHECK(phsp.raw == NULL);

    removeDataset(stem);
}

/*******************************************************************************
* The committed dataset
*
* Everything above reads a file this test file wrote, which leaves the two of
* them free to agree on something the format does not say. tests/data/small is
* a dataset written once by hand, so reading it checks the fixture builders as
* much as the reader: its header is the shape an IAEA writer emits, comments
* and prose blocks and all, and its ten particles cover the corners -- a
* constant z and weight, an extra long, histories opening on a negative
* energy, a backwards particle, and a direction whose u and v already use up
* the whole unit vector.
*******************************************************************************/

#ifndef OMPMC_TEST_DATA_DIR
    #define OMPMC_TEST_DATA_DIR "tests/data"
#endif

static void test_reads_committed_example(void) {

    char stem[512];
    struct OmcPhsp phsp;
    struct OmcPhspRecord got;

    snprintf(stem, sizeof(stem), "%s/small", OMPMC_TEST_DATA_DIR);

    EXPECT_OK(omcPhspFromFile(&phsp, stem));

    CHECK(phsp.header.fileType == 0);
    CHECK(phsp.header.byteOrder == 1234);
    CHECK(phsp.header.recordLength == 25);
    CHECK(phsp.header.checksum == 250);
    CHECK(phsp.header.particles == 10);
    CHECK(phsp.header.origHistories == 9);
    CHECK(omcPhspCount(&phsp) == 10);

    CHECK(phsp.header.typeCount[OMC_PHSP_PHOTON - 1] == 7);
    CHECK(phsp.header.typeCount[OMC_PHSP_ELECTRON - 1] == 2);
    CHECK(phsp.header.typeCount[OMC_PHSP_POSITRON - 1] == 1);

    CHECK(phsp.header.stored[OMC_PHSP_Z] == 0);
    CHECK(phsp.header.stored[OMC_PHSP_WEIGHT] == 0);
    CHECK_CLOSE(phsp.header.constant[OMC_PHSP_Z], 50.0, 1e-6);
    CHECK_CLOSE(phsp.header.constant[OMC_PHSP_WEIGHT], 1.0, 1e-6);
    CHECK(phsp.header.nExtraFloat == 0);
    CHECK(phsp.header.nExtraLong == 1);
    CHECK(phsp.header.extraLongType[0] == OMC_PHSP_LONG_NHIST);

    /* The first particle, which opens a history. */
    EXPECT_OK(omcPhspGet(&phsp, 0, &got));
    CHECK(got.type == OMC_PHSP_PHOTON);
    CHECK(got.newHistory == 1);
    CHECK_CLOSE(got.energy, 1.173, 1e-6);
    CHECK_CLOSE(got.x, 0.10, 1e-6);
    CHECK_CLOSE(got.y, 0.20, 1e-6);
    CHECK_CLOSE(got.z, 50.0, 1e-6);
    CHECK_CLOSE(got.u, 0.010, 1e-6);
    CHECK_CLOSE(got.v, 0.020, 1e-6);
    CHECK_CLOSE(got.w, sqrt(1.0 - 0.010*0.010 - 0.020*0.020), 1e-6);
    CHECK_CLOSE(got.weight, 1.0, 1e-6);
    CHECK(got.extraLong[0] == 1);

    /* The fifth was written with a negative particle type, so it is heading
     back towards the source. */
    EXPECT_OK(omcPhspGet(&phsp, 4, &got));
    CHECK(got.type == OMC_PHSP_PHOTON);
    CHECK(got.newHistory == 0);
    CHECK_CLOSE(got.energy, 0.250, 1e-6);
    CHECK(got.w < 0.0);
    CHECK_CLOSE(got.w, -sqrt(1.0 - 0.100*0.100 - 0.120*0.120), 1e-6);

    /* The seventh is the positron. */
    EXPECT_OK(omcPhspGet(&phsp, 6, &got));
    CHECK(got.type == OMC_PHSP_POSITRON);
    CHECK_CLOSE(got.energy, 0.511, 1e-6);

    /* The last travels in the plane: u and v use up the unit vector between
     them, leaving nothing for w. */
    EXPECT_OK(omcPhspGet(&phsp, 9, &got));
    CHECK(got.newHistory == 1);
    CHECK_CLOSE(got.u, 0.6, 1e-6);
    CHECK_CLOSE(got.v, 0.8, 1e-6);
    CHECK_CLOSE(got.w, 0.0, 1e-6);
    CHECK(got.extraLong[0] == 3);

    /* Popping the whole file finds the six histories the particles were
     written to hold, and every direction is a unit vector. */
    int histories = 0;
    int particles = 0;

    omcPhspRewind(&phsp);
    while (omcPhspNext(&phsp, &got)) {
        histories += got.newHistory;
        particles++;
        CHECK_CLOSE(got.u*got.u + got.v*got.v + got.w*got.w, 1.0, 1e-6);
        CHECK(got.energy > 0.0);
        CHECK(got.type >= OMC_PHSP_PHOTON && got.type <= OMC_PHSP_PROTON);
    }

    CHECK(particles == 10);
    CHECK(histories == 6);

    /* The same count the loader arrived at without decoding anything. */
    CHECK(phsp.newHistories == 6);

    omcPhspFree(&phsp);
}

int main(void) {

    printf("ompMC IAEA phase space reader tests\n\n");

    installFailCatcher();

    RUN(test_header_open_failure);
    RUN(test_header_reads_scalar_keywords);
    RUN(test_header_blocks_in_any_order);
    RUN(test_header_strips_comments_and_blanks);
    RUN(test_header_record_contents_flags);
    RUN(test_header_record_constants);
    RUN(test_header_extra_type_codes);
    RUN(test_header_w_flag_adds_no_bytes);
    RUN(test_header_record_length_mismatch_fails);
    RUN(test_header_missing_keyword_fails);
    RUN(test_header_rejects_big_endian_and_event_files);
    RUN(test_event_generator_header_rejected_before_anything_else);
    RUN(test_header_keeps_a_checksum_it_cannot_check);
    RUN(test_header_too_many_extras_fails);
    RUN(test_header_path_with_extension_accepted);
    RUN(test_header_per_type_counts);

    RUN(test_load_single_record_all_stored);
    RUN(test_constants_substituted);
    RUN(test_w_constant_not_reconstructed);
    RUN(test_negative_energy_marks_new_history);
    RUN(test_new_histories_are_counted);
    RUN(test_negative_type_gives_negative_w);
    RUN(test_uv_overflow_normalized);
    RUN(test_extra_floats_and_longs_decoded);
    RUN(test_indexed_random_access);
    RUN(test_next_pops_then_stops);
    RUN(test_get_out_of_range_fails);
    RUN(test_header_overcounting_particles_is_survivable);
    RUN(test_partial_last_record_is_dropped);
    RUN(test_phsp_file_without_a_whole_record_fails);
    RUN(test_trailing_bytes_warn_but_load);
    RUN(test_bad_type_byte_fails_at_load);
    RUN(test_free_is_idempotent);

    RUN(test_reads_committed_example);

    omcSetHost(NULL);

    printf("\n%d test groups, %d check failures\n", tests_run, tests_failed);

    return tests_failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

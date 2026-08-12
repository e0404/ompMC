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
 omc_phsp - Reading IAEA format phase space files.

 A phase space is a recording of the particles crossing a plane in an earlier
 simulation, and the IAEA format (INDC(NDS)-0484) stores one as a pair of
 files sharing a base name: an ASCII `.IAEAheader` saying what was recorded,
 and a binary `.IAEAphsp` holding the recordings themselves. Datasets in this
 format are published at https://www-nds.iaea.org/phsp/.

 The header decides the layout of a binary record. Each of the seven
 quantities x, y, z, u, v, w and the statistical weight is either stored per
 particle or constant for the whole file, and a file may carry extra floats
 and extra longs beyond those. So a record is not a fixed struct: it is
 whatever the header says it is, which is what struct OmcPhspHeader captures
 and omcPhspGet() decodes.

 Two of the fields are not stored plainly, which is worth knowing when
 reading the decoding code:

 - the particle type is a signed byte whose SIGN carries the sign of w. The
   magnitude of w itself is never stored at all, it is recovered from
   u and v as sqrt(1 - u^2 - v^2), unless the header declares w constant.
 - the energy is negative for the first particle of each new independent
   history, which is how a reader can tell how many histories it has
   consumed.

 Usage is the same shape as omc_spectrum: fill the struct from a file, use
 it, free it.

     struct OmcPhsp phsp;
     struct OmcPhspRecord particle;
     omcPhspFromFile(&phsp, "some/beam");
     while (omcPhspNext(&phsp, &particle)) { ... }
     omcPhspFree(&phsp);

 @warning The whole binary file is read into memory, which is fine for the
 published datasets of a few hundred MB but is worth a thought before opening
 something enormous. Particles are held in the packed form they have on disk
 and decoded on access, so the cost is the size of the file rather than a
 multiple of it, and a future buffered reader can replace the storage without
 the accessors below changing.
*****************************************************************************/

#ifndef OMC_PHSP_H
#define OMC_PHSP_H

#include <stdint.h>

/*! The particle types an IAEA record can carry, as stored in the first byte
 of the record. */
enum OmcPhspParticleType {
    OMC_PHSP_PHOTON = 1,
    OMC_PHSP_ELECTRON = 2,
    OMC_PHSP_POSITRON = 3,
    OMC_PHSP_NEUTRON = 4,
    OMC_PHSP_PROTON = 5
};

/*! What an extra float in a record means, from the header. */
enum OmcPhspExtraFloat {
    OMC_PHSP_FLOAT_USER = 0,    /**< user defined, generic */
    OMC_PHSP_FLOAT_XLAST = 1,   /**< x of the last interaction */
    OMC_PHSP_FLOAT_YLAST = 2,   /**< y of the last interaction */
    OMC_PHSP_FLOAT_ZLAST = 3    /**< z of the last interaction */
};

/*! What an extra long in a record means, from the header. */
enum OmcPhspExtraLong {
    OMC_PHSP_LONG_USER = 0,     /**< user defined, generic */
    OMC_PHSP_LONG_NHIST = 1,    /**< incremental history number */
    OMC_PHSP_LONG_LATCH = 2,    /**< EGS LATCH */
    OMC_PHSP_LONG_ILB = 3       /**< PENELOPE ILB */
};

/*! Index of a quantity in struct OmcPhspHeader::stored and
 struct OmcPhspHeader::constant, in the order the format stores them. */
enum OmcPhspVariable {
    OMC_PHSP_X = 0,
    OMC_PHSP_Y = 1,
    OMC_PHSP_Z = 2,
    OMC_PHSP_U = 3,
    OMC_PHSP_V = 4,
    OMC_PHSP_W = 5,
    OMC_PHSP_WEIGHT = 6,
    OMC_PHSP_NVARIABLES = 7
};

/*! The most extra floats or extra longs a record may carry, the same limit
 the IAEA reference implementation sets. */
#define OMC_PHSP_MAX_EXTRA 10

/*! What a `.IAEAheader` says about the file next to it. The prose sections a
 header also carries -- title, coordinate system, transport parameters,
 statistics -- are not kept; this is the part a reader needs. */
struct OmcPhspHeader {
    int fileType;               ///< 0 : a phase space file. 1, an event generator, is not read
    int byteOrder;              ///< 1234 : little endian, 4321 : big endian
    int recordLength;           ///< bytes per particle record

    /*! The size the binary file should have, in bytes. NOT #recordLength
     times #particles: published datasets exist where those two disagree. */
    unsigned long long checksum;

    /*! Particles the header says the binary file holds. What it actually
     holds is omcPhspCount(), and the two do differ in the wild. */
    unsigned long long particles;

    unsigned long long origHistories;   ///< histories that produced them

    /*! Particles of each type, indexed by enum OmcPhspParticleType minus
     one. Zero where the header does not say. */
    unsigned long long typeCount[5];

    /*! 1 : the quantity is stored per particle, 0 : it is the same for every
     particle and given by #constant. Indexed by enum OmcPhspVariable.

     @warning OMC_PHSP_W is the odd one out: w is never stored in a record,
     so a 1 here means "recover it from u and v", not "read it". */
    int stored[OMC_PHSP_NVARIABLES];

    /*! The value of each quantity that is not stored, indexed by enum
     OmcPhspVariable. Meaningless where #stored is 1. */
    float constant[OMC_PHSP_NVARIABLES];

    int nExtraFloat;            ///< extra floats per record, 0 to OMC_PHSP_MAX_EXTRA
    int nExtraLong;             ///< extra longs per record, 0 to OMC_PHSP_MAX_EXTRA

    int extraFloatType[OMC_PHSP_MAX_EXTRA];   ///< enum OmcPhspExtraFloat per extra float
    int extraLongType[OMC_PHSP_MAX_EXTRA];    ///< enum OmcPhspExtraLong per extra long
};

/*! One particle, decoded. */
struct OmcPhspRecord {
    int type;                   ///< enum OmcPhspParticleType, always positive
    int newHistory;             ///< 1 : the first particle of a new independent history

    double energy;              ///< kinetic energy in MeV, always positive
    double x, y, z;             ///< position in cm
    double u, v, w;             ///< direction cosines
    double weight;              ///< statistical weight

    float extraFloat[OMC_PHSP_MAX_EXTRA];    ///< first OmcPhspHeader::nExtraFloat are set
    int32_t extraLong[OMC_PHSP_MAX_EXTRA];   ///< first OmcPhspHeader::nExtraLong are set
};

/*! A phase space file, read into memory and ready to draw particles from. */
struct OmcPhsp {
    struct OmcPhspHeader header;        ///< what the `.IAEAheader` said

    /*! Particles held, counted from the size of the binary file rather than
     taken from the header. */
    unsigned long long nRecords;

    /*! Particles that open a new independent history, counted while loading.

     @warning Zero means the file does not mark histories AT ALL rather than
     that it holds none, and files like that are published: the particles a
     single history left behind cannot be told apart in one, so anything
     drawing from it has to treat every particle as its own history and will
     understate its own uncertainty by however much those particles are
     correlated. */
    unsigned long long newHistories;

    /*! The binary file, still in its packed on disk form. Decoded a record
     at a time by omcPhspGet(). */
    unsigned char *raw;

    unsigned long long cursor;          ///< the record omcPhspNext() returns next
};

/* Both readers below take the base name of the dataset, the way the IAEA
 files are named: "beam" reads beam.IAEAheader and beam.IAEAphsp. A path that
 already ends in either extension is accepted too, so that a name picked out
 of a file dialog can be passed straight through. */

/*! Read a `.IAEAheader` without touching the binary file beside it. Useful
 to find out how big a file is before deciding to read it.

 @param header Filled in from the header file.
 @param path Base name of the dataset, or the path of either of its files. */
void omcPhspHeaderFromFile(struct OmcPhspHeader *header, const char *path);

/*! Read a whole phase space file into memory.

 @param phsp Filled in from the dataset, ready for the accessors below.
 @param path Base name of the dataset, or the path of either of its files.

 @warning Everything that can go wrong with the file goes wrong here, and is
 reported through omcFail(), which does not return: a header that contradicts
 itself, a binary file with nothing readable in it, a record with a particle
 type the format does not define. That is deliberate -- it leaves
 omcPhspGet() with nothing left to check on a code path that may run once per
 history.

 @warning A file holding a different number of particles than its header
 announces is NOT one of those things. It is reported through omcLog() and
 read anyway, as many particles as are actually there, because the published
 datasets include one of those and refusing it would help nobody. This holds
 in both directions: the file is read to its end, so a header that counts
 too few does not cost you the rest of it any more than one that counts too
 many invents a particle. The header's count is only the first guess at how
 much memory to take. Callers that care should compare omcPhspCount() with
 struct OmcPhspHeader::particles. */
void omcPhspFromFile(struct OmcPhsp *phsp, const char *path);

/*! @param phsp The file to ask about.
 @return The number of particles it holds, counted from the size of the file
 rather than taken from the header. */
unsigned long long omcPhspCount(const struct OmcPhsp *phsp);

/*! Decode one particle, by position in the file.

 @param phsp The file to read from.
 @param index Which particle, from 0 to omcPhspCount() - 1.
 @param record Filled in with the particle. */
void omcPhspGet(const struct OmcPhsp *phsp, unsigned long long index,
                struct OmcPhspRecord *record);

/*! Decode the next particle, advancing the read position: the file used as a
 stack of particles to pop rather than an array to index.

 @param phsp The file to read from.
 @param record Filled in with the particle, untouched at the end of the file.
 @return 1 if a particle was read, 0 at the end of the file. */
int omcPhspNext(struct OmcPhsp *phsp, struct OmcPhspRecord *record);

/*! Put the read position of omcPhspNext() back to the first particle.

 @param phsp The file to rewind. */
void omcPhspRewind(struct OmcPhsp *phsp);

/*! Release the memory omcPhspFromFile() allocated. Safe to call twice.

 @param phsp The file to close. */
void omcPhspFree(struct OmcPhsp *phsp);

#endif

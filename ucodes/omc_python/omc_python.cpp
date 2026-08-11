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

/******************************************************************************
 omc_python - the compiled half of the Python interface, built as _ompmc.

 This is a host in the sense of omc_host.h, the sibling of ucodes/omc_matrad:
 it translates numpy arrays into the plain C structs the engines take, runs
 them, and hands the results back. Everything Python-facing that can be
 written in Python -- keyword arguments, validation, scipy conversion -- lives
 in ompmc/__init__.py instead, so this file stays small.

 THE GIL. Every argument is turned into plain C data while the GIL is held.
 The GIL is then released for the whole calculation, which is what lets the
 engines' OpenMP regions run at full speed: no worker thread ever touches a
 Python object. It is taken again only inside the progress trampoline, which
 the engines call from the master thread outside any parallel region -- the
 invariant omc_host.h states. Nothing below the release line may touch a
 Python object.

 CANCELLATION. A Python exception raised by the progress callback, including
 the KeyboardInterrupt that Ctrl-C leaves pending, must not unwind through the
 C frames of the engine. It is left set, the engine is asked to stop, and it
 surfaces once the call has returned normally.

 ERRORS. Shared C code reports fatal conditions through omcFail(), which must
 not return. Throwing a C++ exception through C frames is not something C
 compilers promise to support, so the sink stores the message and longjmp()s
 back to runGuarded(), which turns it into a Python exception. Everything the
 longjmp() skips over must therefore be trivially destructible: runGuarded()
 and the run functions it calls keep to plain structs and pointers, and every
 object that owns memory lives in the caller's frame.
*****************************************************************************/

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <csetjmp>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "omc_collimator.h"
#include "omc_engine_cube.h"
#include "omc_engine_dij.h"
#include "omc_engine_forward.h"
#include "omc_geom.h"
#include "omc_host.h"
#include "omc_phsp.h"
#include "omc_source_phsp.h"
#include "omc_spectrum.h"
#include "omc_utilities.h"
#include "omc_version.h"
#include "ompmc.h"

extern struct Media media;

/* Every host defines this; the core library reads it back to decide how much
 the transport code prints. */
int verbose_flag = 0;
}

namespace nb = nanobind;
using namespace nb::literals;

/******************************************************************************/
/* Host sinks */

static std::jmp_buf failJump;
static std::string failId;
static std::string failMessage;

static void pythonLogSink(int level, const char *message, void *user) {

    (void)user;

    /* Warnings always get through; the rest follows the verbosity asked for */
    if (level > OMC_LOG_WARNING && verbose_flag < level) {
        return;
    }

    nb::gil_scoped_acquire gil;
    try {
        nb::print(nb::str(message));
    } catch (...) {
        /* A broken sys.stdout must not take the calculation down */
        PyErr_Clear();
    }
}

static void pythonFailSink(const char *id, const char *message, void *user) {

    (void)user;

    failId = id ? id : "ompMC:error";
    failMessage = message ? message : "unknown error";

    std::longjmp(failJump, 1);
}

static void installHost() {

    struct OmcHost host;
    host.log = pythonLogSink;
    host.fail = pythonFailSink;
    host.user = nullptr;

    omcSetHost(&host);
}

/* Run fn(arg) with omcFail() turned into a false return. Holds nothing that
 needs destruction: the longjmp() skips over this frame's cleanup. */
static bool runGuarded(void (*fn)(void *), void *arg) {

    if (setjmp(failJump) != 0) {
        return false;
    }

    fn(arg);

    return true;
}

/******************************************************************************/
/* Progress trampoline */

struct ProgressState {
    PyObject *callable;         // borrowed, may be null
    bool cancelled;             // a Python exception is pending
};

static int callProgress(ProgressState *state, double fraction) {

    nb::gil_scoped_acquire gil;

    /* Ctrl-C cannot be delivered while the GIL is released, so this is where
     an interrupt surfaces. */
    if (PyErr_CheckSignals() != 0) {
        state->cancelled = true;
        return 0;
    }

    if (state->callable == nullptr) {
        return 1;
    }

    nb::object result;
    try {
        result = nb::borrow<nb::object>(state->callable)(fraction);
    } catch (nb::python_error &e) {
        e.restore();            // leave it set for the re-raise after the run
        state->cancelled = true;
        return 0;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        state->cancelled = true;
        return 0;
    }

    /* Returning False asks for the run to stop. Returning None -- what a
     callback that only draws a progress bar does -- does not. */
    if (result.is_none()) {
        return 1;
    }

    return nb::cast<bool>(result, false) ? 1 : 0;
}

/******************************************************************************/
/* Plain C views of the arguments, assembled while the GIL is held */

using Cube = nb::ndarray<const double, nb::ndim<3>, nb::f_contig, nb::device::cpu>;
using IntCube = nb::ndarray<const int32_t, nb::ndim<3>, nb::f_contig, nb::device::cpu>;
using Vector = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using IntVector = nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Triples = nb::ndarray<const double, nb::shape<-1, 3>, nb::f_contig, nb::device::cpu>;
using Grid = nb::ndarray<const double, nb::ndim<2>, nb::f_contig, nb::device::cpu>;

/* How the source energies are described. Points into the caller's arrays and
 strings, both of which outlive the run. */
struct SpectrumInput {
    int kind;                   // 0 monoenergetic, 1 histogram, 2 file
    double monoEnergy;

    const double *energy;
    const double *fluence;
    int nbins;
    double emin;
    int mode;

    const char *path;
};

static void buildSpectrum(struct OmcSpectrum *spectrum,
                          const SpectrumInput *input) {

    switch (input->kind) {
        case 1:
            omcSpectrumFromHistogram(spectrum, input->energy, input->fluence,
                                     input->nbins, input->emin, input->mode);
            break;
        case 2:
            omcSpectrumFromFile(spectrum, input->path);
            break;
        default:
            omcSpectrumMonoenergetic(spectrum, input->monoEnergy);
            break;
    }
}

/* Read the spectrum description out of the dict the Python side built. The
 strings and arrays it points at are kept alive by holding on to the dict. */
static SpectrumInput parseSpectrum(const nb::dict &spec, std::string &pathOut) {

    SpectrumInput input{};

    if (spec.contains("energy")) {
        auto energy = nb::cast<Vector>(spec["energy"]);
        auto fluence = nb::cast<Vector>(spec["fluence"]);

        if (energy.shape(0) == 0 || energy.shape(0) != fluence.shape(0)) {
            throw std::invalid_argument("spectrum energy and fluence must be "
                "non-empty and of the same length");
        }

        input.kind = 1;
        input.energy = energy.data();
        input.fluence = fluence.data();
        input.nbins = (int) energy.shape(0);
        input.emin = nb::cast<double>(spec["e_min"]);
        input.mode = nb::cast<int>(spec["mode"]);
    }
    else if (spec.contains("file")) {
        pathOut = nb::cast<std::string>(spec["file"]);
        input.kind = 2;
        input.path = pathOut.c_str();
    }
    else {
        input.kind = 0;
        input.monoEnergy = nb::cast<double>(spec["mono_energy"]);
    }

    return input;
}

/* The geometry, as pointers into the caller's numpy buffers */
struct GeometryInput {
    const double *density;
    const int32_t *material;
    const double *xBounds;
    const double *yBounds;
    const double *zBounds;
    int isize, jsize, ksize;
    char names[MXMED][60];
    int nmed;
};

static GeometryInput parseGeometry(const Cube &density, const IntCube &material,
                                   const Vector &xBounds, const Vector &yBounds,
                                   const Vector &zBounds,
                                   const std::vector<std::string> &materials) {

    GeometryInput geo{};

    if (materials.empty() || materials.size() > MXMED) {
        throw std::invalid_argument("between 1 and " + std::to_string(MXMED) +
            " materials are needed, got " + std::to_string(materials.size()));
    }
    for (size_t i = 0; i < materials.size(); i++) {
        if (materials[i].size() >= sizeof(geo.names[0])) {
            throw std::invalid_argument("material name '" + materials[i] +
                "' is too long");
        }
        std::strcpy(geo.names[i], materials[i].c_str());
    }
    geo.nmed = (int) materials.size();

    geo.isize = (int) density.shape(0);
    geo.jsize = (int) density.shape(1);
    geo.ksize = (int) density.shape(2);

    if (material.shape(0) != density.shape(0) ||
        material.shape(1) != density.shape(1) ||
        material.shape(2) != density.shape(2)) {
        throw std::invalid_argument(
            "density and material cubes must have the same shape");
    }
    if (xBounds.shape(0) != (size_t)geo.isize + 1 ||
        yBounds.shape(0) != (size_t)geo.jsize + 1 ||
        zBounds.shape(0) != (size_t)geo.ksize + 1) {
        throw std::invalid_argument("each bounds vector must have one more "
            "entry than the cube has voxels along that axis");
    }

    geo.density = density.data();
    geo.material = material.data();
    geo.xBounds = xBounds.data();
    geo.yBounds = yBounds.data();
    geo.zBounds = zBounds.data();

    return geo;
}

/* Copy the parsed geometry into the core's globals. No Python here. */
static void installGeometry(const GeometryInput *geo) {

    media.nmed = geo->nmed;
    for (int i = 0; i < geo->nmed; i++) {
        std::strcpy(media.med_names[i], geo->names[i]);
    }

    geometry.isize = geo->isize;
    geometry.jsize = geo->jsize;
    geometry.ksize = geo->ksize;

    /* The engines only read these, but struct Geom is the one the transport
     shares with the user codes and predates const correctness. */
    geometry.med_densities = (double *) geo->density;
    geometry.med_indices = (int *) geo->material;
    geometry.xbounds = (double *) geo->xBounds;
    geometry.ybounds = (double *) geo->yBounds;
    geometry.zbounds = (double *) geo->zBounds;

    omcGeomDetectSpacing();
}

static void applyInputItems(const nb::dict &items) {

    omcClearInputValues();

    for (auto item : items) {
        omcSetInputValue(nb::cast<std::string>(nb::str(item.first)).c_str(),
                         nb::cast<std::string>(nb::str(item.second)).c_str());
    }
}

static void cleanupPhysics() {

    cleanPhoton();
    cleanRayleigh();
    cleanPair();
    cleanElectron();
    cleanMscat();
    cleanSpin();
    cleanRegions();
}

/* Hand a vector's storage to numpy without copying it */
template <typename T>
static nb::ndarray<nb::numpy, T> adopt(std::vector<T> &&values) {

    auto *owned = new std::vector<T>(std::move(values));
    nb::capsule owner(owned, [](void *p) noexcept {
        delete (std::vector<T> *) p;
    });

    size_t shape[1] = { owned->size() };
    return nb::ndarray<nb::numpy, T>(owned->data(), 1, shape, owner);
}

/******************************************************************************/
/* The beamlet source, shared by calc_dij and calc_forward.

 The Fortran order of the triples means each column is contiguous, which is
 exactly the one-array-per-component layout the engine wants, so the arrays are
 used in place. They belong to the caller and have to outlive the call, which
 they do: nanobind keeps the Python objects alive for its duration. */

static void parseBeamletSource(struct OmcBeamletSource &src,
                               const IntVector &i_beam, const Triples &source,
                               const Triples &corner, const Triples &side1,
                               const Triples &side2) {

    src.nbeamlets = (int) i_beam.shape(0);
    src.ibeam = (const int *) i_beam.data();

    if (corner.shape(0) != (size_t)src.nbeamlets ||
        side1.shape(0) != (size_t)src.nbeamlets ||
        side2.shape(0) != (size_t)src.nbeamlets) {
        throw std::invalid_argument("corner, side1 and side2 must have one "
            "row per beamlet");
    }

    const size_t nbeams = source.shape(0);
    const size_t nbeamlets = (size_t) src.nbeamlets;

    src.xsource = source.data();
    src.ysource = source.data() + nbeams;
    src.zsource = source.data() + 2*nbeams;
    src.xcorner = corner.data();
    src.ycorner = corner.data() + nbeamlets;
    src.zcorner = corner.data() + 2*nbeamlets;
    src.xside1 = side1.data();
    src.yside1 = side1.data() + nbeamlets;
    src.zside1 = side1.data() + 2*nbeamlets;
    src.xside2 = side2.data();
    src.yside2 = side2.data() + nbeamlets;
    src.zside2 = side2.data() + 2*nbeamlets;

    for (size_t i = 0; i < nbeamlets; i++) {
        if (src.ibeam[i] < 0 || (size_t)src.ibeam[i] >= nbeams) {
            throw std::invalid_argument("i_beam entry " + std::to_string(i) +
                " is outside the " + std::to_string(nbeams) + " beams given");
        }
    }
}

/******************************************************************************/
/* The collimator, shared by every forward calculation.

 The transmission values are copied rather than borrowed: they are the only
 thing here small enough that copying is free, and the copy has to outlive the
 GIL release either way. The struct therefore has to live in the caller's
 frame -- the mask points into its own vector -- which is why this fills one
 in place instead of returning it. */

struct CollimatorInput {
    struct OmcApertureMask mask;
    std::vector<double> transmission;
    bool roulette;
    bool present;
};

static void parseCollimator(const nb::object &object, CollimatorInput &out) {

    out.present = false;
    out.roulette = false;
    std::memset(&out.mask, 0, sizeof(out.mask));

    if (object.is_none()) {
        return;
    }

    nb::dict spec = nb::cast<nb::dict>(object);

    /* Fortran ordered, so the first axis is contiguous -- which is the x runs
     fastest layout struct OmcApertureMask asks for. */
    auto values = nb::cast<Grid>(spec["transmission"]);

    out.mask.nx = (int) values.shape(0);
    out.mask.ny = (int) values.shape(1);

    const size_t ncells = (size_t) out.mask.nx*(size_t) out.mask.ny;
    out.transmission.assign(values.data(), values.data() + ncells);

    out.mask.z = nb::cast<double>(spec["z"]);
    out.mask.x0 = nb::cast<double>(spec["x0"]);
    out.mask.y0 = nb::cast<double>(spec["y0"]);
    out.mask.dx = nb::cast<double>(spec["dx"]);
    out.mask.dy = nb::cast<double>(spec["dy"]);
    out.mask.outside = nb::cast<double>(spec["outside"]);
    out.mask.transmission = out.transmission.data();

    out.roulette = nb::cast<bool>(spec["roulette"]);
    out.present = true;
}

/* Dress a parsed collimator as the modifier an engine takes, or hand back
 nothing at all for an open beam. @p modifier lives in the caller's frame. */
static const struct OmcBeamModifier *
useCollimator(CollimatorInput *input, struct OmcBeamModifier *modifier) {

    if (input == nullptr || !input->present) {
        return nullptr;
    }

    omcApertureMaskAsModifier(&input->mask, modifier);
    modifier->apply = input->roulette ? OMC_MODIFIER_ROULETTE
                                      : OMC_MODIFIER_WEIGHT;

    return modifier;
}

/******************************************************************************/
/* Dij */

struct DijContext {
    /* Results, in compressed sparse column order */
    std::vector<double> data;
    std::vector<int32_t> indices;
    std::vector<int64_t> indptr;
    std::vector<double> variance;
    bool wantVariance;

    ProgressState progress;
};

static void collectBeamlet(int ibeamlet, int nvoxels, const int *voxels,
                           const double *dose, const double *variance,
                           void *user) {

    (void)ibeamlet;

    DijContext *ctx = (DijContext *) user;

    for (int n = 0; n < nvoxels; n++) {
        ctx->data.push_back(dose[n]);
        ctx->indices.push_back(voxels[n]);
        if (ctx->wantVariance) {
            ctx->variance.push_back(variance ? variance[n] : 0.0);
        }
    }

    ctx->indptr.push_back((int64_t) ctx->data.size());
}

static int dijProgress(double fraction, void *user) {
    return callProgress(&((DijContext *) user)->progress, fraction);
}

/* Argument block for the guarded run: plain pointers only */
struct DijRun {
    const struct OmcDijOptions *options;
    const struct OmcBeamletSource *source;
    const SpectrumInput *spectrumInput;
    const GeometryInput *geometry;
    struct OmcDijCallbacks *callbacks;
    int beamletsDone;
};

static void runDij(void *arg) {

    DijRun *run = (DijRun *) arg;
    struct OmcSpectrum spectrum;

    installGeometry(run->geometry);
    initMediaData();
    buildSpectrum(&spectrum, run->spectrumInput);
    initRegions();
    initVrt();

    run->beamletsDone = omcCalcDij(run->options, run->source, &spectrum,
                                   run->callbacks);

    omcSpectrumFree(&spectrum);
    cleanupPhysics();
}

/******************************************************************************/
/* Cube */

struct CubeContext {
    ProgressState progress;
    int nbatch;
};

static int cubeProgress(int ibatch, int nbatch, uint64_t firstHistory,
                        void *user) {

    (void)firstHistory;

    CubeContext *ctx = (CubeContext *) user;

    return callProgress(&ctx->progress, (double)ibatch/(double)nbatch);
}

struct CubeRun {
    const struct OmcCubeOptions *options;
    struct OmcSsdSource *source;
    const SpectrumInput *spectrumInput;
    const GeometryInput *geometry;
    struct OmcCubeCallbacks *callbacks;
    double *dose;
    double *uncertainty;
    struct OmcCubeSummary *summary;
    int completed;
};

static void runCube(void *arg) {

    CubeRun *run = (CubeRun *) arg;
    struct OmcSpectrum spectrum;

    installGeometry(run->geometry);
    initMediaData();
    buildSpectrum(&spectrum, run->spectrumInput);
    initRegions();
    initVrt();
    omcSsdSourceInit(run->source);

    run->completed = omcCalcCube(run->options, run->source, &spectrum,
                                 run->dose, run->uncertainty,
                                 run->callbacks, run->summary);

    omcSpectrumFree(&spectrum);
    cleanupPhysics();
}

/******************************************************************************/
/* Forward */

struct ForwardContext {
    ProgressState progress;
};

static int forwardProgress(double fraction, void *user) {
    return callProgress(&((ForwardContext *) user)->progress, fraction);
}

struct ForwardRun {
    const struct OmcForwardOptions *options;
    const struct OmcBeamletSource *source;
    const double *weights;
    const SpectrumInput *spectrumInput;
    const GeometryInput *geometry;
    CollimatorInput *collimator;
    struct OmcForwardCallbacks *callbacks;
    double *dose;
    double *uncertainty;
    struct OmcForwardSummary *summary;
    struct OmcBeamletStats *stats;
    int charge;
    enum OmcSourceGeometry sourceGeometry;
    double sourceGaussianWidth;
    int completed;
};

static void runForward(void *arg) {

    ForwardRun *run = (ForwardRun *) arg;
    struct OmcSpectrum spectrum;

    installGeometry(run->geometry);
    initMediaData();
    buildSpectrum(&spectrum, run->spectrumInput);
    initRegions();
    initVrt();

    /* The engine takes any source; these are weighted beamlets. */
    struct OmcBeamletHistories histories;
    histories.sampler.source = run->source;
    histories.sampler.spectrum = &spectrum;
    histories.sampler.charge = run->charge;
    histories.sampler.geometry = run->sourceGeometry;
    histories.sampler.gaussianWidth = run->sourceGaussianWidth;
    histories.weights = run->weights;

    struct OmcSource source;
    omcBeamletHistoriesAsSource(&histories, &source);

    /* Usually nothing: with beamlets the collimation is already in the
     weights the caller handed over, and a mask on top of them is for the
     cases the weights cannot express -- a block, or a leaf that transmits. */
    struct OmcBeamModifier modifier;
    const struct OmcBeamModifier *use = useCollimator(run->collimator,
                                                      &modifier);

    run->completed = omcCalcForward(run->options, &source, use, run->dose,
                                    run->uncertainty, run->callbacks,
                                    run->summary);

    /* How the histories were shared out belongs to the beamlet source, so it
     is asked of it rather than found in the engine's summary. */
    omcBeamletHistoriesStats(&histories, run->stats);

    omcSpectrumFree(&spectrum);
    cleanupPhysics();
}

/******************************************************************************/
/* Forward, from a phase space file */

/* Where the phase space sits and how to draw from it. The path is a copy
 because the run reads it with the GIL released. */
struct PhspInput {
    std::string path;
    int order;                  // 0 replay in order, 1 draw at random
    uint64_t first;
    double rotation[9];
    double translation[3];
};

struct PhspRun {
    const struct OmcForwardOptions *options;
    const PhspInput *phsp;
    const GeometryInput *geometry;
    CollimatorInput *collimator;
    struct OmcForwardCallbacks *callbacks;
    double *dose;
    double *uncertainty;
    struct OmcForwardSummary *summary;

    /* Zeroed by the caller and freed by the caller, so that a file that
     fails to load halfway through is still released: the longjmp() out of
     omcFail() does not come back through here. */
    struct OmcPhsp *file;

    int completed;
};

static void runForwardPhsp(void *arg) {

    PhspRun *run = (PhspRun *) arg;

    installGeometry(run->geometry);
    initMediaData();
    initRegions();
    initVrt();

    /* No spectrum: a phase space carries the energy of every particle it
     holds, which is most of the reason for using one. */
    omcPhspFromFile(run->file, run->phsp->path.c_str());

    struct OmcPhspSampler sampler;
    std::memset(&sampler, 0, sizeof(sampler));
    sampler.phsp = run->file;
    sampler.order = run->phsp->order == 1 ? OMC_PHSP_RANDOM : OMC_PHSP_REPLAY;
    sampler.first = run->phsp->first;
    std::memcpy(sampler.transform.rotation, run->phsp->rotation,
                sizeof(sampler.transform.rotation));
    std::memcpy(sampler.transform.translation, run->phsp->translation,
                sizeof(sampler.transform.translation));

    struct OmcSource source;
    omcPhspSamplerAsSource(&sampler, &source);

    struct OmcBeamModifier modifier;
    const struct OmcBeamModifier *use = useCollimator(run->collimator,
                                                      &modifier);

    run->completed = omcCalcForward(run->options, &source, use, run->dose,
                                    run->uncertainty, run->callbacks,
                                    run->summary);

    cleanupPhysics();
}

/******************************************************************************/

NB_MODULE(_ompmc, m) {

    m.doc() = "Compiled core of the ompMC Python interface";

    m.attr("__version__") = OMPMC_VERSION_STRING;
    m.attr("MAX_MEDIA") = MXMED;

    m.def("calc_dij",
        [](Cube density, IntCube material, Vector x_bounds, Vector y_bounds,
           Vector z_bounds, std::vector<std::string> materials,
           IntVector i_beam, Triples source, Triples corner, Triples side1,
           Triples side2, nb::dict options, nb::dict inputItems,
           nb::dict spectrum, nb::object progress, int verbosity) {

        /* --- everything in this block runs with the GIL held --- */

        GeometryInput geo = parseGeometry(density, material, x_bounds,
                                          y_bounds, z_bounds, materials);

        std::string spectrumPath;
        SpectrumInput spectrumInput = parseSpectrum(spectrum, spectrumPath);

        struct OmcBeamletSource src;
        parseBeamletSource(src, i_beam, source, corner, side1, side2);

        struct OmcDijOptions opt;
        opt.nhist = nb::cast<int>(options["n_histories"]);
        opt.nbatch = nb::cast<int>(options["n_batches"]);
        opt.charge = nb::cast<int>(options["charge"]);
        opt.relDoseThreshold = nb::cast<double>(options["rel_dose_threshold"]);
        opt.sourceGeometry = nb::cast<bool>(options["gaussian_source"])
            ? OMC_SOURCE_GAUSSIAN : OMC_SOURCE_POINT;
        opt.sourceGaussianWidth = nb::cast<double>(options["source_width"]);
        opt.wantVariance = nb::cast<bool>(options["want_variance"]) ? 1 : 0;

        installHost();
        verbose_flag = verbosity;
        applyInputItems(inputItems);

        DijContext ctx;
        ctx.wantVariance = opt.wantVariance != 0;
        ctx.indptr.push_back(0);
        ctx.progress.callable = progress.is_none() ? nullptr : progress.ptr();
        ctx.progress.cancelled = false;

        struct OmcDijCallbacks callbacks;
        callbacks.beamlet = collectBeamlet;
        callbacks.progress = dijProgress;
        callbacks.user = &ctx;

        DijRun run{&opt, &src, &spectrumInput, &geo, &callbacks, 0};

        bool ok;
        {
            /* --- no Python beyond this point, except in the trampoline --- */
            nb::gil_scoped_release nogil;
            ok = runGuarded(&runDij, &run);
        }

        if (!ok) {
            throw std::runtime_error(failId + ": " + failMessage);
        }
        if (ctx.progress.cancelled) {
            throw nb::python_error();       // re-raises what the callback left
        }

        return nb::make_tuple(adopt(std::move(ctx.data)),
                              adopt(std::move(ctx.indices)),
                              adopt(std::move(ctx.indptr)),
                              adopt(std::move(ctx.variance)),
                              run.beamletsDone);
    },
    "density"_a, "material"_a, "x_bounds"_a,
    "y_bounds"_a, "z_bounds"_a, "materials"_a, "i_beam"_a,
    "source"_a, "corner"_a, "side1"_a, "side2"_a, "options"_a,
    "input_items"_a, "spectrum"_a, "progress"_a.none(), "verbosity"_a,
    "Dose influence matrix for a set of beamlets, as raw CSC arrays.");

    m.def("calc_cube",
        [](Cube density, IntCube material, Vector x_bounds, Vector y_bounds,
           Vector z_bounds, std::vector<std::string> materials,
           nb::dict options, nb::dict inputItems, nb::dict spectrum,
           nb::object progress, int verbosity) {

        GeometryInput geo = parseGeometry(density, material, x_bounds,
                                          y_bounds, z_bounds, materials);

        std::string spectrumPath;
        SpectrumInput spectrumInput = parseSpectrum(spectrum, spectrumPath);

        struct OmcSsdSource src{};
        src.ssd = nb::cast<double>(options["ssd"]);
        src.xinl = nb::cast<double>(options["x_min"]);
        src.xinu = nb::cast<double>(options["x_max"]);
        src.yinl = nb::cast<double>(options["y_min"]);
        src.yinu = nb::cast<double>(options["y_max"]);

        struct OmcCubeOptions opt;
        opt.nhist = nb::cast<int>(options["n_histories"]);
        opt.nbatch = nb::cast<int>(options["n_batches"]);
        opt.charge = nb::cast<int>(options["charge"]);
        opt.outputDose = nb::cast<bool>(options["output_dose"]) ? 1 : 0;

        installHost();
        verbose_flag = verbosity;
        applyInputItems(inputItems);

        const size_t gridsize = (size_t)geo.isize*(size_t)geo.jsize
                                *(size_t)geo.ksize;
        std::vector<double> dose(gridsize, 0.0);
        std::vector<double> uncertainty(gridsize, 0.0);

        CubeContext ctx;
        ctx.progress.callable = progress.is_none() ? nullptr : progress.ptr();
        ctx.progress.cancelled = false;
        ctx.nbatch = opt.nbatch;

        struct OmcCubeCallbacks callbacks;
        callbacks.batch = cubeProgress;
        callbacks.user = &ctx;

        struct OmcCubeSummary summary{};
        CubeRun run{&opt, &src, &spectrumInput, &geo, &callbacks,
                    dose.data(), uncertainty.data(), &summary, 0};

        bool ok;
        {
            nb::gil_scoped_release nogil;
            ok = runGuarded(&runCube, &run);
        }

        if (!ok) {
            throw std::runtime_error(failId + ": " + failMessage);
        }
        if (ctx.progress.cancelled) {
            throw nb::python_error();
        }
        if (!run.completed) {
            throw std::runtime_error("the calculation was stopped before any "
                "result was available");
        }

        return nb::make_tuple(adopt(std::move(dose)),
                              adopt(std::move(uncertainty)),
                              summary.nhist, summary.energyFraction);
    },
    "density"_a, "material"_a, "x_bounds"_a,
    "y_bounds"_a, "z_bounds"_a, "materials"_a, "options"_a, "input_items"_a,
    "spectrum"_a, "progress"_a.none(), "verbosity"_a,
    "Dose in every voxel from one collimated beam.");

    m.def("calc_forward",
        [](Cube density, IntCube material, Vector x_bounds, Vector y_bounds,
           Vector z_bounds, std::vector<std::string> materials,
           IntVector i_beam, Triples source, Triples corner, Triples side1,
           Triples side2, Vector weights, nb::dict options,
           nb::dict input_items, nb::dict spectrum, nb::object collimator,
           nb::object progress, int verbosity) {

        GeometryInput geo = parseGeometry(density, material, x_bounds,
                                          y_bounds, z_bounds, materials);

        std::string spectrumPath;
        SpectrumInput spectrumInput = parseSpectrum(spectrum, spectrumPath);

        struct OmcBeamletSource src;
        parseBeamletSource(src, i_beam, source, corner, side1, side2);

        if (weights.shape(0) != (size_t)src.nbeamlets) {
            throw std::invalid_argument("weights has " +
                std::to_string(weights.shape(0)) + " entries but there are " +
                std::to_string(src.nbeamlets) + " beamlets");
        }

        struct OmcForwardOptions opt;
        opt.nhist = nb::cast<int>(options["n_histories"]);
        opt.nbatch = nb::cast<int>(options["n_batches"]);
        opt.outputDose = nb::cast<bool>(options["output_dose"]) ? 1 : 0;

        installHost();
        verbose_flag = verbosity;
        applyInputItems(input_items);

        const size_t gridsize = (size_t)geo.isize*(size_t)geo.jsize
                                *(size_t)geo.ksize;
        std::vector<double> dose(gridsize, 0.0);
        std::vector<double> uncertainty(gridsize, 0.0);

        ForwardContext ctx;
        ctx.progress.callable = progress.is_none() ? nullptr : progress.ptr();
        ctx.progress.cancelled = false;

        struct OmcForwardCallbacks callbacks;
        callbacks.progress = forwardProgress;
        callbacks.user = &ctx;

        CollimatorInput collimatorInput;
        parseCollimator(collimator, collimatorInput);

        struct OmcForwardSummary summary{};
        struct OmcBeamletStats stats{};
        ForwardRun run{&opt, &src, weights.data(), &spectrumInput, &geo,
                       &collimatorInput,
                       &callbacks, dose.data(), uncertainty.data(), &summary,
                       &stats,
                       nb::cast<int>(options["charge"]),
                       nb::cast<bool>(options["gaussian_source"])
                           ? OMC_SOURCE_GAUSSIAN : OMC_SOURCE_POINT,
                       nb::cast<double>(options["source_width"]),
                       0};

        bool ok;
        {
            nb::gil_scoped_release nogil;
            ok = runGuarded(&runForward, &run);
        }

        if (!ok) {
            throw std::runtime_error(failId + ": " + failMessage);
        }
        if (ctx.progress.cancelled) {
            throw nb::python_error();
        }

        /* A progress callback that returned False leaves no exception behind,
         so what a voluntary stop means is the caller's decision, the same way
         calc_dij reports a short beamlet count. There is no partial result to
         hand back either way: the batches are averaged, so a run that stopped
         halfway is a dose with no meaning. */
        return nb::make_tuple(adopt(std::move(dose)),
                              adopt(std::move(uncertainty)),
                              run.completed != 0,
                              summary.nhist, stats.nsampled,
                              stats.nweighted,
                              stats.sampledWeight/stats.totalWeight,
                              summary.energyFraction, summary.blocked);
    },
    "density"_a, "material"_a, "x_bounds"_a,
    "y_bounds"_a, "z_bounds"_a, "materials"_a, "i_beam"_a,
    "source"_a, "corner"_a, "side1"_a, "side2"_a, "weights"_a, "options"_a,
    "input_items"_a, "spectrum"_a, "collimator"_a.none(), "progress"_a.none(),
    "verbosity"_a,
    "Dose in every voxel from a whole weighted set of beamlets.");

    m.def("calc_forward_phsp",
        [](Cube density, IntCube material, Vector x_bounds, Vector y_bounds,
           Vector z_bounds, std::vector<std::string> materials,
           nb::dict phsp, nb::dict options, nb::dict input_items,
           nb::object collimator, nb::object progress, int verbosity) {

        GeometryInput geo = parseGeometry(density, material, x_bounds,
                                          y_bounds, z_bounds, materials);

        PhspInput phspInput;
        phspInput.path = nb::cast<std::string>(phsp["path"]);
        phspInput.order = nb::cast<int>(phsp["order"]);
        phspInput.first = nb::cast<uint64_t>(phsp["first"]);

        auto rotation = nb::cast<std::vector<double>>(phsp["rotation"]);
        auto translation = nb::cast<std::vector<double>>(phsp["translation"]);

        if (rotation.size() != 9 || translation.size() != 3) {
            throw std::invalid_argument("the phase space transform needs a "
                "nine element rotation and a three element translation");
        }
        std::memcpy(phspInput.rotation, rotation.data(),
                    sizeof(phspInput.rotation));
        std::memcpy(phspInput.translation, translation.data(),
                    sizeof(phspInput.translation));

        struct OmcForwardOptions opt;
        opt.nhist = nb::cast<int>(options["n_histories"]);
        opt.nbatch = nb::cast<int>(options["n_batches"]);
        opt.outputDose = nb::cast<bool>(options["output_dose"]) ? 1 : 0;

        installHost();
        verbose_flag = verbosity;
        applyInputItems(input_items);

        const size_t gridsize = (size_t)geo.isize*(size_t)geo.jsize
                                *(size_t)geo.ksize;
        std::vector<double> dose(gridsize, 0.0);
        std::vector<double> uncertainty(gridsize, 0.0);

        ForwardContext ctx;
        ctx.progress.callable = progress.is_none() ? nullptr : progress.ptr();
        ctx.progress.cancelled = false;

        struct OmcForwardCallbacks callbacks;
        callbacks.progress = forwardProgress;
        callbacks.user = &ctx;

        CollimatorInput collimatorInput;
        parseCollimator(collimator, collimatorInput);

        /* Zeroed here rather than in the run, so that freeing it below is
         safe however far into the load the failure came. */
        struct OmcPhsp file;
        std::memset(&file, 0, sizeof(file));

        struct OmcForwardSummary summary{};
        PhspRun run{&opt, &phspInput, &geo, &collimatorInput, &callbacks,
                    dose.data(), uncertainty.data(), &summary, &file, 0};

        bool ok;
        {
            nb::gil_scoped_release nogil;
            ok = runGuarded(&runForwardPhsp, &run);

            /* The particles are the biggest thing a run of this kind holds --
             gigabytes for a published data set -- so they go back before the
             GIL does. */
            omcPhspFree(&file);
        }

        if (!ok) {
            throw std::runtime_error(failId + ": " + failMessage);
        }
        if (ctx.progress.cancelled) {
            throw nb::python_error();
        }

        return nb::make_tuple(adopt(std::move(dose)),
                              adopt(std::move(uncertainty)),
                              run.completed != 0,
                              summary.nhist, summary.started, summary.blocked,
                              summary.energyFraction);
    },
    "density"_a, "material"_a, "x_bounds"_a, "y_bounds"_a, "z_bounds"_a,
    "materials"_a, "phsp"_a, "options"_a, "input_items"_a,
    "collimator"_a.none(), "progress"_a.none(), "verbosity"_a,
    "Dose in every voxel from the particles of an IAEA phase space file.");
}

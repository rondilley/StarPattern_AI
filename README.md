# Star Pattern AI

AI-powered autonomous discovery pipeline for patterns in astronomical star fields. Combines evolutionary parameter optimization, multi-survey data acquisition (SDSS, Gaia, MAST, ZTF), GPU-accelerated pattern detection with learned meta-detection and compositional pipeline evolution, and LLM-as-strategist architecture for token-efficient closed-loop discovery.

## What It Does

Star Pattern AI autonomously scans the sky looking for patterns across multiple domains:

1. **Gravitational lenses** -- Einstein rings, arcs, and multiply-imaged sources
2. **Galaxy morphology anomalies** -- unusual shapes, CAS/Gini/M20 outliers, Sersic profile residuals
3. **Galaxy interaction signatures** -- tidal tails, merger double-nuclei, color anomalies
4. **Stellar distribution anomalies** -- overdensities, voids, clustering (Voronoi/Clark-Evans)
5. **Kinematic structures** -- co-moving groups, stellar streams, runaway stars (Gaia proper motions)
6. **Time-domain variability** -- periodic variables, eclipsing binaries, transients, AGN (ZTF light curves)
7. **Multi-scale features** -- wavelet-detected extended emission, nebular structures
8. **Stellar populations** -- CMD analysis, blue stragglers, multiple populations
9. **Emergent patterns** -- novel detection strategies discovered via evolved compositional pipelines

The pipeline evolves both its detection parameters (72-gene genome) and detection strategies (variable-length pipeline genomes) using genetic algorithms. A learned meta-detector replaces the linear ensemble with non-linear scoring trained via active learning. Findings are cross-referenced against SIMBAD/NED/TNS catalogs. LLMs serve as periodic strategists (~99% token reduction vs per-detection calls), with local classifiers and evaluators handling routine decisions.

## Requirements

- Python 3.10+
- CUDA-capable GPU (optional, falls back to CPU)
- API keys for at least one LLM provider (OpenAI, Anthropic, Google, xAI)

## Installation

```bash
pip install -e .
```

With optional dependencies:

```bash
pip install -e ".[dev]"       # pytest, black, ruff, mypy
pip install -e ".[gpu]"       # CuPy for GPU acceleration
pip install -e ".[ml-extra]"  # UMAP, HDBSCAN
pip install -e ".[local]"     # llama.cpp for local LLMs
pip install -e ".[report]"    # PDF report generation
```

## API Keys

Place API key files in the project root using the `*.key.txt` naming convention:

```
openai.key.txt      # OpenAI API key
claude.key.txt      # Anthropic API key
gemini.key.txt      # Google Gemini API key
xai.key.txt         # xAI (Grok) API key
```

Each file contains the raw API key string, nothing else. These files are gitignored. Providers are auto-discovered at runtime from whichever key files exist.

## Usage

All commands run through the CLI:

```bash
# Fetch SDSS data for a specific sky region
python -m star_pattern.cli fetch --ra 180.0 --dec 45.0 --radius 3.0

# Fetch 50 random high-galactic-latitude regions
python -m star_pattern.cli fetch --random 50 --min-gal-lat 20

# Run pattern detection on a FITS image
python -m star_pattern.cli detect --input image.fits

# Batch detect on a directory of FITS files
python -m star_pattern.cli detect --input data/ --batch --output results/

# Evolve detection parameters over 50 generations
python -m star_pattern.cli evolve --generations 50 --population 40

# Run autonomous discovery for 8 hours with LLM analysis
python -m star_pattern.cli discover --hours 8 --with-llm

# Run autonomous discovery for 100 cycles
python -m star_pattern.cli discover --cycles 100

# Analyze a detection result with LLM hypothesis and debate
python -m star_pattern.cli analyze --input results/pattern.json --with-debate

# Train a lens detection model
python -m star_pattern.cli train --task lens --data data/lenses/ --epochs 100

# Set up local LLM backend
python -m star_pattern.cli setup-local
```

Add `-v` for verbose logging or `-c path/to/config.json` to use a custom config.

## Configuration

Default settings are in `config.json`. Key sections:

| Section | Controls |
|---|---|
| `data` | Survey sources, cache directory, search radius |
| `detection` | Source extraction threshold, Gabor filter params, ensemble weights |
| `evolution` | Population size, generations, mutation/crossover rates, fitness weights |
| `llm` | Key directory, token limits, debate rounds, consensus settings |
| `pipeline` | Output directory, checkpoint interval, max cycles |

## Detection Pipeline

The detection pipeline runs 14 specialized detectors plus learned meta-detection:

- **Classical CV** -- Gabor filter banks, FFT power spectrum analysis, Hough arc detection
- **Source extraction** -- SEP for detection with photutils fallback; fixed-aperture photometry and fixed-window second moments for flux and shape (see Reproducibility below)
- **Morphology** -- CAS statistics, Gini coefficient, M20, ellipticity from moments
- **Anomaly detection** -- Isolation Forest on feature embeddings
- **Lens detection** -- Central source finding, arc detection in annular sectors, ring completeness scoring
- **Distribution analysis** -- Voronoi tessellation, 2-point correlation, Clark-Evans statistic, KDE overdensity
- **Galaxy features** -- Tidal feature detection, merger candidates, color anomaly flagging
- **Kinematic analysis** -- Co-moving groups (DBSCAN), stellar streams (RANSAC in 4D), runaway stars
- **Transient detection** -- Astrometric excess noise, photometric variability, parallax anomalies
- **Sersic profile fitting** -- Galaxy morphology classification, residual substructure detection
- **Wavelet multi-scale** -- A-trous decomposition, multi-scale source detection
- **Stellar populations** -- CMD analysis, main sequence/RGB/blue straggler identification
- **Variability analysis** -- ZTF light curves, Lomb-Scargle periodograms, outburst detection
- **Temporal differencing** -- Multi-epoch image differencing for new sources, brightening, fading, and motion

After the 14 detectors, a **FeatureFusionExtractor** builds a 65-D feature vector, and a **MetaDetector** (linear -> GBM -> neural net) provides learned non-linear scoring. **ComposedPipelines** (evolved sequences of image operations) discover detection strategies not hard-coded in any detector.

GPU acceleration is available for the wavelet and classical detectors through `star_pattern.utils.hardware`, which supports CUDA and ROCm via PyTorch and degrades to the CPU path when no accelerator is present. Run `star-pattern gpu-check` to see what the current machine exposes.

## Reproducibility

The same image analysed twice produces the same catalog, byte for byte.

That is not free. SEP's deblender is not reproducible: given a byte-identical array it returns identical source positions and counts, then assigns blended pixels to neighbours differently on every call. Measured on a synthetic field with SEP 1.4.1, 16 of 38 sources had flux varying by up to 9.7% between runs, and one source in 38 flipped between the star and galaxy classifications. No SEP version fixes this (1.4.1, 1.4.0 and the sep-pjw 1.3.8 fork are identical; 1.2.1 has no wheel for Python 3.12) and no deblend parameter fixes it without switching deblending off, which costs about a third of the detections.

The pipeline therefore uses SEP for detection only, and measures flux and shape itself:

- **Flux** comes from `sep.sum_circle` at a fixed aperture (default 5 pixels) rather than SEP's segmentation flux. A fixed aperture recovers a median 92% of the segmentation flux, and 64% at the 10th percentile, so extended sources are measured more conservatively. Brightness ranking correlates with the old segmentation flux at r = 0.94.
- **Shape** comes from flux-weighted second moments in a fixed window, 1.6x the photometric aperture, rather than SEP's `a`, `b` and `theta`. The window is deliberately wider than the aperture: measuring shape in the same circle used for photometry caps the visible extent and collapses the star/galaxy split.

`sep` is pinned to `==1.4.1` so this characterised behaviour cannot shift silently. `tests/test_source_extraction_determinism.py` and `tests/test_ensemble_characterization.py` hold the guarantee, the latter comparing full serialized output against golden fixtures to 1e-9.

## Statistical Confidence

Detections fall into two evidence families, and the two are never mixed:

- **Tail probability** -- the detector supplies a physical measurement with a real null model (Gaussian SNR, Poisson counts, binomial rates, chi-squared residuals, Lomb-Scargle false-alarm probability). These carry a `p_value` and are the only members of the region-wide Benjamini-Hochberg FDR family.
- **Heuristic** -- the detector supplies a unitless 0-1 score with no null distribution behind it (isolation-forest scores, Hough vote counts, ring completeness, score fallbacks). These carry a `heuristic_score`, an explicitly null `p_value`, and are triaged against a score cutoff rather than a significance threshold.

Reports label every detection with its `evidence_basis`, and tail-family findings rank above heuristic findings. A heuristic score is a triage signal for deciding what to look at, not a claim about the false-positive rate.

Positions are cross-referenced against SIMBAD, NED, and TNS. Each result records which catalogs actually answered: when `coverage_complete` is false, the absence of a match is not evidence of novelty.

## Evolutionary Search

Detection parameters are encoded as a 72-gene genome covering source extraction, Gabor filters, anomaly detection, lens detection, morphology, distribution analysis, galaxy features, kinematic analysis, transient detection, sersic, wavelet, stellar population, variability, ensemble weights, and meta/representation/compositional parameters. A separate variable-length PipelineGenome encodes evolved detection strategies. The genetic algorithm uses tournament selection, elitism, adaptive mutation, experience replay, and the following fitness function:

$$\text{Fitness} = 0.35 \cdot \text{anomaly} + 0.25 \cdot \text{significance} + 0.15 \cdot \text{novelty} + 0.1 \cdot \text{diversity} + 0.15 \cdot \text{recovery}$$

Twelve preset genomes (lens, morphology, distribution, balanced, sensitive, kinematic, transient, sersic, wavelet, population, variability, temporal) seed the detection population. Eight preset pipeline genomes seed the compositional pipeline population.

## LLM Integration

Three LLM evaluation modes, all using real API calls to multiple providers:

1. **Hypothesis generation** -- Converts a detection into a physical mechanism with testable predictions
2. **Adversarial debate** -- Advocate argues the pattern is real, challenger argues artifact, judge renders verdict
3. **Consensus scoring** -- Multiple LLMs independently rate significance on 1-10 scale, results combined via Borda count

Supported providers: OpenAI (GPT-4o), Anthropic (Claude Sonnet 4), Google (Gemini 2.5 Flash), xAI (Grok 2), and local models via llama.cpp.

## Testing

```bash
# Run full test suite (730 tests, includes real API calls)
python -m pytest tests/ -v

# Run without LLM tests (no API usage)
python -m pytest tests/ --ignore=tests/test_llm_hypothesis.py

# Run only a specific test file
python -m pytest tests/test_detection.py -v
```

Tests use real data sources and real LLM providers. No mocks. Tests that require network access or API keys use `pytest.skip()` when resources are unavailable.

## Project Structure

```
src/star_pattern/
    core/           Core data types (config, FITS, sky regions, catalogs)
    data/           Multi-survey acquisition (SDSS, Gaia, MAST, ZTF) + FITS/catalog caching
    detection/      Pattern detection (14 detectors, ensemble, feature fusion,
                    meta-detector, compositional pipelines, local classifier/evaluator)
    discovery/      Evolutionary search (72-gene genome, pipeline genome, fitness,
                    GA engine, pipeline co-evolution, presets)
    distributed/    Master/slave work distribution (protocol, dispatch, bridge)
    llm/            LLM integration (strategy advisor, hypothesis, debate, consensus,
                    providers, model registry)
    ml/             Machine learning (backbone, embeddings, losses, models, training,
                    representation manager)
    evaluation/     Validation (metrics, confidence, cross-reference, statistics,
                    synthetic injection)
    visualization/  Output (overlays, mosaics, reports)
    pipeline/       Orchestration (autonomous discovery, active learning, batch)
    utils/          Shared utilities (logging, retry, run management)
    utils/hardware/ Accelerators (GPU/NPU detection, GPU array operations)
```

## License

MIT

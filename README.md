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

The pipeline evolves both its detection parameters (54-gene genome) and detection strategies (variable-length pipeline genomes) using genetic algorithms. A learned meta-detector replaces the linear ensemble with non-linear scoring trained via active learning. Findings are cross-referenced against SIMBAD/NED/TNS catalogs. LLMs serve as periodic strategists (~99% token reduction vs per-detection calls), with local classifiers and evaluators handling routine decisions.

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

The detection pipeline runs 13 specialized detectors plus learned meta-detection:

- **Classical CV** -- Gabor filter banks, FFT power spectrum analysis, Hough arc detection
- **Source extraction** -- SEP (primary) with photutils fallback
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

After the 13 detectors, a **FeatureFusionExtractor** builds a ~60-D feature vector, and a **MetaDetector** (linear -> GBM -> neural net) provides learned non-linear scoring. **ComposedPipelines** (evolved sequences of image operations) discover detection strategies not hard-coded in any detector.

## Evolutionary Search

Detection parameters are encoded as a 54-gene genome covering source extraction, Gabor filters, anomaly detection, lens detection, morphology, distribution analysis, galaxy features, kinematic analysis, transient detection, sersic, wavelet, stellar population, variability, ensemble weights, and meta/representation/compositional parameters. A separate variable-length PipelineGenome encodes evolved detection strategies. The genetic algorithm uses tournament selection, elitism, adaptive mutation, experience replay, and the following fitness function:

$$\text{Fitness} = 0.35 \cdot \text{anomaly} + 0.25 \cdot \text{significance} + 0.15 \cdot \text{novelty} + 0.1 \cdot \text{diversity} + 0.15 \cdot \text{recovery}$$

Eleven preset genomes (lens, morphology, distribution, balanced, sensitive, kinematic, transient, sersic, wavelet, population, variability) seed the detection population. Eight preset pipeline genomes seed the compositional pipeline population.

## LLM Integration

Three LLM evaluation modes, all using real API calls to multiple providers:

1. **Hypothesis generation** -- Converts a detection into a physical mechanism with testable predictions
2. **Adversarial debate** -- Advocate argues the pattern is real, challenger argues artifact, judge renders verdict
3. **Consensus scoring** -- Multiple LLMs independently rate significance on 1-10 scale, results combined via Borda count

Supported providers: OpenAI (GPT-4o), Anthropic (Claude Sonnet 4), Google (Gemini 2.5 Flash), xAI (Grok 2), and local models via llama.cpp.

## Testing

```bash
# Run full test suite (387 tests, includes real API calls)
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
    detection/      Pattern detection (13 detectors, ensemble, feature fusion,
                    meta-detector, compositional pipelines, local classifier/evaluator)
    discovery/      Evolutionary search (54-gene genome, pipeline genome, fitness,
                    GA engine, pipeline co-evolution, presets)
    llm/            LLM integration (hypothesis, debate, consensus, providers)
    ml/             Machine learning (backbone, embeddings, losses, models, training,
                    representation manager)
    evaluation/     Validation (metrics, cross-reference, statistics, synthetic injection)
    visualization/  Output (sky plots, overlays, mosaics, reports)
    pipeline/       Orchestration (autonomous discovery, active learning, batch)
    utils/          Shared utilities (GPU, logging, retry, run management)
```

## License

MIT

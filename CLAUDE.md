# star_pattern_AI

## Project Overview

AI-powered autonomous discovery pipeline for patterns in astronomical star fields. Combines evolutionary parameter optimization, multi-survey data acquisition, GPU-accelerated pattern detection, and LLM-as-strategist architecture for token-efficient closed-loop discovery.

## Architecture

Full architecture documentation: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

- `src/star_pattern/` -- Main package (105 source files across 12 subpackages)
  - `core/` -- Data types: PipelineConfig (dataclass configs including MetaDetectorConfig, RepresentationConfig, CompositionalConfig, TemporalConfig), WideFieldConfig, SurveyConfig, FITSImage (astropy FITS I/O + WCS), SkyRegion (coordinate handling), EpochImage (multi-epoch image with MJD/band metadata), StarCatalog (with merge/dedup), CatalogEntry (with to_dict/from_dict serialization), TileGrid (hex-packed sky tiling), HEALPixSurvey (systematic sky coverage with healpy)
  - `data/` -- Multi-survey data acquisition via astroquery: SDSSDataSource (+ Stripe 82 multi-epoch via `fetch_epoch_images()`), GaiaDataSource (TAP+), MASTDataSource (HST/JWST, configurable max_observations, + multi-epoch via `fetch_epoch_images()`), ZTFDataSource (IRSA TAP light curves + IBE epoch image cutouts, g/r/i bands, stores light curves in CatalogEntry.properties), DataSource ABC (with default `fetch_epoch_images()` returning `{}`), DataCache (SHA256-keyed FITS + catalog + epoch caching across runs, source-specific epoch keys: ztf_epoch/mast_epoch/sdss_epoch), DataPipeline (multi-source orchestrator, generic temporal fetch from all sources with per-source isolation), Mosaicker (reproject-based image stitching), WideFieldPipeline (tile+fetch+mosaic orchestrator). All data sources cache catalog queries to avoid redundant network calls on repeated runs.
  - `detection/source_extraction.py` -- SEP for detection (including deblending), but photometry and shape are measured here, not taken from SEP. SEP's deblender is not reproducible: identical input gives identical positions and counts but different pixel-to-source assignment on every call, so its `flux`, `npix`, `a`, `b` and `theta` move between runs. Flux comes from `sep.sum_circle` at a fixed aperture (`aperture_radius_px`, default 5px) and shape from flux-weighted second moments in a wider fixed window (`moment_radius_px`, default 1.6x the aperture). Both are deterministic. The moment window must stay wider than the photometric aperture or the star/galaxy split degenerates to "everything is a star"
  - `detection/` -- Pattern detection: 14 specialized detectors (ClassicalDetector, SourceExtractor, MorphologyAnalyzer, AnomalyDetector, LensDetector, DistributionAnalyzer, GalaxyDetector, ProperMotionAnalyzer, TransientDetector, SersicAnalyzer, WaveletAnalyzer, StellarPopulationAnalyzer, VariabilityAnalyzer, TemporalDetector) + `base.py` (DetectorSpec registry, DetectionContext, RECOVERABLE_DETECTOR_EXCEPTIONS) + EnsembleDetector (registry-driven single loop, pixel-scale-aware, extracts 65-D rich_features via FeatureFusionExtractor, optional MetaDetector scoring, respects per-detector enable gates from genome, score renormalized over the detectors that actually ran) + FeatureFusionExtractor (cross-detector feature extraction) + MetaDetector (learned non-linear scoring: linear -> GBM at 50 labels -> neural net at 200 labels) + ComposedPipeline (variable-length detection pipelines from 10 primitive operations) + LocalClassifier (rule-based, zero tokens) + LocalEvaluator (SNR/agreement-based, zero tokens)
  - `discovery/` -- Evolutionary search: DetectionGenome (72 genes: 48 detector params + 12 enable gates + 6 temporal + 6 meta/representation/compositional), FitnessEvaluator (5-component with synthetic injection recovery + type-diversity bonus, uses rich_features when available), EvolutionaryDiscovery (GA engine with adaptive mutation + experience replay + LLM-guided population seeding + active-learning weight injection + pipeline co-evolution), PipelineGenome (variable-length 2-5 ops, structural + parametric mutation), 12 preset detection genomes + 8 preset pipeline genomes
  - `llm/` -- LLM-as-strategist: StrategyAdvisor (periodic batch review, ~1,000 tokens/session), TokenTracker (budget enforcement), LLMCache (SHA256-keyed response caching), models.py (central model-identifier registry with verification dates). Reachable only through `analyze --with-debate`: HypothesisGenerator, PatternDebate, PatternConsensus. LLMSearchGuide has no caller at all and is a deletion candidate
  - `llm/providers/` -- Provider system: LLMProvider ABC (with generate_tracked/generate_cached), ProviderDiscovery (auto-scan *.key.txt), OpenAI/Claude/Gemini/xAI/LlamaCpp implementations
  - `ml/` -- Machine learning: BackboneWrapper (EfficientNet/ResNet/ZooBot), FeatureExtractor, FocalLoss/DiceLoss, SimpleUNet/LensNet/AstroClassifier, BYOL pretrainer, Trainer, RepresentationManager (orchestrates BackboneWrapper + FeatureExtractor + SSLPretrainer + EmbeddingAnomalyDetector into the pipeline, with BYOL retrain and embedding anomaly scoring)
  - `evaluation/` -- Validation: Anomaly (per-detection dataclass: type, detector, pixel/sky coords, score, confidence, group_id, properties), PatternResult (with anomalies list, region_confidence), ConfidenceScore (p-value-based statistical confidence with physical basis and annotation), ConfidenceEvaluator (per-detector confidence scoring: 13 methods mapping physical quantities to p-values, quality-floor filtering, BH-FDR correction, spatial grouping via Union-Find), signal_to_noise, detection_significance, CatalogCrossReferencer (SIMBAD/NED/TNS), bootstrap/KS/Anderson-Darling/permutation tests, SyntheticInjector
  - `visualization/` -- Output: WCS-aware sky plots, detection overlays, anomaly-centric discovery mosaic (dynamic panel count for all quality-passing anomalies up to 100, sorted by confidence, compact layout when >30 panels, confidence labels, signal quality filter, extended feature recentering, per-feature contrast stretch), DiscoveryReport (markdown + JSON + mosaic + histogram, per-anomaly table with location/type/score/confidence/group, evidence breakdown with annotations, group summaries with Fisher combined p-values)
  - `pipeline/` -- Orchestration: AutonomousDiscovery (main loop with fast SIGINT shutdown -- first CTRL-C stops after current phase, second CTRL-C exits immediately; local classification/evaluation, per-anomaly extraction with confidence scoring and quality-floor filtering replacing arbitrary caps, BH-FDR correction, spatial grouping, periodic LLM strategy sessions, adaptive evolution with LLM-seeded variants + pipeline co-evolution, meta-detector scoring, representation learning, active learning, token tracking, image saving, report generation, optional wide-field mode, optional HEALPix survey mode), ActiveLearner (feedback-driven retraining, ensemble weight learning, meta-detector sample feeding, strategy integration, persistence, adaptive query strategy), BatchProcessor
  - `distributed/` -- Master/slave work distribution over asyncio sockets: MasterDispatcher, SlaveServer, MasterBridge, wire protocol (hmac-authenticated, gzip-framed), DistributedConfig. Driven by `discover --slaves` and the `serve` command
  - `utils/` -- Structured logging, retry_with_backoff (sync+async, with permanent-failure short circuit so 401/404 are not retried), RunManager (checkpoints/state/images)
  - `utils/hardware/` -- Accelerator layer re-exported through `utils/gpu.py`: `backends.py` (GPUBackend enum, CUDA/ROCm detection, cached device and array-module selection, hardware_summary), `npu.py` (NPUBackend, Linux amdxdna probe, ONNX Runtime providers and sessions), `ops.py` (torch-based gpu_fft2_power, gpu_separable_convolve, gpu_fftconvolve_batch, gpu_edge_magnitude, each returning None with a CPU fallback in the caller)

## Key Commands

```bash
python -m star_pattern.cli fetch --ra 180 --dec 45 --radius 3     # Download data
python -m star_pattern.cli fetch-wide --ra 180 --dec 45 --field-radius 15 -o /tmp/wide  # Wide-field mosaic
python -m star_pattern.cli detect --input image.fits               # Run detection
python -m star_pattern.cli evolve --generations 50                 # Evolve parameters
python -m star_pattern.cli discover --hours 8 --with-llm           # Autonomous discovery (with ZTF)
python -m star_pattern.cli discover --hours 8 --no-ztf             # Discovery without ZTF light curves
python -m star_pattern.cli discover --wide-field 10 --cycles 2     # Wide-field discovery
python -m star_pattern.cli discover --survey --nside 8 --cycles 3  # HEALPix survey discovery
python -m star_pattern.cli survey-status --state-file output/runs/.../survey_state.json  # Survey progress
python -m star_pattern.cli analyze --input pattern.json --with-debate  # LLM analysis
python -m star_pattern.cli train --task lens --data ./data --epochs 20  # Train a backbone
python -m star_pattern.cli gpu-check                               # Report GPU/NPU status
python -m star_pattern.cli setup-local                             # Install a local GGUF model
python -m star_pattern.cli serve --host 0.0.0.0 --port 7827        # Run as a distributed worker
python -m star_pattern.cli discover --slaves host1:7827,host2:7827 # Distributed discovery
python -m pytest tests/ -v                                         # Run 730 tests
python -m pytest tests/ -m "not network and not llm"               # Offline subset (what CI runs)
```

## Conventions

- Python 3.11+, type hints on all public interfaces
- Dataclasses for all configuration (PipelineConfig, DataConfig, DetectionConfig, EvolutionConfig, LLMConfig, WideFieldConfig, SurveyConfig)
- ABCs for extensible interfaces (DataSource, LLMProvider)
- All 72 detection parameters are genome-tunable via the GA (48 detector + 12 enable gates + 6 temporal + 6 meta/representation/compositional)
- LLM providers auto-discovered from `*.key.txt` files in project root (openai.key.txt, claude.key.txt, gemini.key.txt, xai.key.txt)
- FITS data and catalog queries cached in `output/cache/` with SHA256-keyed index
- Run artifacts in `output/runs/{timestamp}/` with JSON checkpoints
- No mocks in tests -- real API calls with pytest.skip() when unavailable
- No emoji/icons/symbols in code or output (Rule 1 in .claude/claude-code-rules.md)
- All API calls must have try/except with specific exception types
- Fitness function: $F = 0.35 \cdot \text{anomaly} + 0.25 \cdot \text{significance} + 0.15 \cdot \text{novelty} + 0.1 \cdot \text{diversity} + 0.15 \cdot \text{recovery}$

## Dependencies

Core: numpy, scipy, matplotlib, astropy, astroquery, photutils, sep, torch, torchvision, scikit-learn, openai, anthropic, google-generativeai, click, rich, tqdm, Pillow, requests

Optional: cupy-cuda12x (GPU), astropy-healpix (HEALPix survey), umap-learn + hdbscan (ML extra), llama-cpp-python + huggingface_hub (local LLMs), reproject>=0.13 (wide-field mosaicking), reportlab (PDF reports), pytest + black + ruff + mypy (dev)

## LLM Provider Models

Defaults live in one place: `src/star_pattern/llm/providers/models.py`. Do not
edit model identifiers from memory -- a plausible but wrong identifier fails
at call time with a 404 that reads like a network problem. Each entry records
when it was last verified.

| Provider | Model | Key File | Live check 2026-08-01 |
|---|---|---|---|
| OpenAI | gpt-4o | openai.key.txt | works |
| Anthropic | claude-sonnet-5 | claude.key.txt | works |
| xAI | grok-4.5 | xai.key.txt | works |
| Google | gemini-2.5-flash | gemini.key.txt | KEY REJECTED (API_KEY_INVALID) |
| Local | *.gguf in models/ | (no key needed) | not configured |
| TNS (cross-match) | n/a, bot key only | tns.key.txt (optional) | not configured |

The Gemini key is correctly formed (39 characters, `AIzaSy` prefix) but the
API rejects it. Replace the key, or enable the Generative Language API on
the Google Cloud project it belongs to. Until then the pipeline runs with
three providers, which still satisfies the two-provider minimum for debate
and consensus.

To re-check a provider's model identifier, ask its API rather than guessing.
The xAI default `grok-2-latest` was retired without the `-latest` suffix
following the line forward, and the API answered "Model not found":

```python
from openai import OpenAI
OpenAI(api_key=..., base_url="https://api.x.ai/v1").models.list()
```

Anthropic models from Opus 4.7 onward reject `temperature`, `top_p` and
`top_k`; the Claude provider omits the sampling parameter for those models
rather than sending a value that returns 400.

## Workflow

- Enter plan mode for any non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, stop and re-plan immediately -- do not keep pushing a broken approach
- Use subagents liberally to keep the main context window clean; offload research, exploration, and parallel analysis; one task per subagent for focused execution
- When given a bug report, fix it autonomously -- point at logs, errors, failing tests, then resolve them; zero context switching required from the user
- For non-trivial changes, pause and ask "is there a more elegant way?"; skip this for simple, obvious fixes
- Track task progress via TaskCreate/TaskUpdate tools with checkable items; mark items complete as you go
- Simplicity first: make every change as simple as possible, minimal code impact
- Find root causes; no temporary fixes; senior developer standards

## Self-Improvement

- After any correction from the user, update `VIBE_HISTORY.md` with the pattern and lesson learned
- Write rules for yourself that prevent the same mistake from recurring
- Review `VIBE_HISTORY.md` for relevant lessons at the start of complex tasks

## Communication

- Focus on substance over praise; skip unnecessary compliments
- Engage critically: question assumptions, identify issues, offer counterpoints
- Be direct about problems instead of softening criticism
- Prioritize accuracy and honesty over validation
- Ground agreement in evidence and reason, not reflexive validation
- Challenge problematic approaches even if not asked for criticism

### Code Feedback Checklist

Before code-related responses, verify:
1. Have I questioned questionable assumptions?
2. Have I identified potential bugs or security issues?
3. Have I checked if this duplicates existing code?
4. Have I been direct about problems instead of dancing around them?
5. Have I provided evidence/reasoning for my positions?

## Project Rules

Enforced via Claude Code hooks in `.claude/hooks/` (registered in `.claude/settings.json`):

1. No emoji/icons/symbols in code or output
2. Never declare success without testing
3. No stubs, placeholders, fake data, or mocks
4. Never push secrets to git (*.key.txt is gitignored)
5. All API calls must have error handling
6. Update CLAUDE.md on architecture changes
7. Update VIBE_HISTORY.md with lessons learned
8. Rules cannot be deleted or weakened

Full rules: `.claude/claude-code-rules.md`

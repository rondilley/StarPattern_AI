# Vibe History

## 2026-08-01 - Correctness Audit, Refresh, and Accelerator Layer

### Lessons Learned

- **A test can encode a bug as ground truth, and then defend it.** The
  Benjamini-Hochberg correction ran its step-up monotonicity pass over the
  input array order instead of the rank order. For p = [0.9, 0.5, 0.001, 0.7]
  it returned [0.004, 0.004, 0.004, 0.933] instead of [0.9, 0.9, 0.004, 0.9],
  so one real detection dragged every noise detection in the region down with
  it and the report ranked on that number. `test_known_bh_fdr_values` asserted
  the wrong value, and its comment spelled out the wrong algorithm step by
  step. A test named "verify against manually computed values" is only as good
  as the derivation in the comment. When a hand-derived expectation exists,
  cross-check it against a reference implementation
  (`scipy.stats.false_discovery_control`) rather than against the code.
  The same pattern appeared twice more: `test_lens_adapts_radii` asserted the
  shared-state mutation bug as a feature.

- **Do not put a number in a field whose name is a claim.** Nine detectors
  computed `p = 1 - score` from a unitless 0-1 heuristic and reported it in a
  `p_value` field, then fed it into FDR alongside genuine Gaussian and Poisson
  tail probabilities. This invalidated the correction for every detection in
  the region, not just the heuristic ones. The fix is two explicit evidence
  families: only real tail probabilities carry a `p_value` and enter FDR;
  heuristics carry a `heuristic_score` and a null `p_value`. Related: a
  quality floor of 0.0013 means "3 sigma"; reusing it as a score cutoff of
  0.9987 is a category error that silently deletes a whole detection channel.

- **An outage and an empty result must never be the same value.** Four
  catalog queries carried a `@retry_with_backoff` decorator and then caught
  the exception inside the function body, so no retry could ever fire and
  every failure returned `[]`. `cross_reference` read that as "no catalog
  match", which reads as "novel discovery". A 30-second SIMBAD outage was
  enough to promote a catalogued galaxy. Results now record which catalogs
  actually answered, and `coverage_complete` tells a real miss from an
  unanswered question. Wrapping a call in a retry decorator does nothing if
  the body swallows the exception first.

- **Truthiness is the wrong test for a measurement.** Eight Gaia fields used
  `if row["pmra"]`, which discards a proper motion, parallax, or colour index
  of exactly 0.0 and drops the source from every kinematic analysis
  downstream. Astropy also returns masked cells and NaN rather than None, so
  `is not None` alone is not enough either.

- **Per-image state on a shared detector makes results depend on arrival
  order.** `LensDetector.detect` and `ClassicalDetector.detect` wrote the
  pixel-scale-derived radii back onto `self`. The ensemble builds each
  detector once and reuses it for the whole run, so one image with a WCS
  permanently redefined the radii for every image after it, including images
  with no WCS. Detection was not reproducible and nothing reported it.

- **Verify hardware detection against the degraded case, not just the happy
  one.** `torch.cuda.is_available()` returns True with `CUDA_VISIBLE_DEVICES`
  empty while `device_count()` is 0, and every later device call raises
  "Invalid device id". Availability checks need to confirm a usable device,
  not just a loaded runtime. Separately, `get_device_properties(0).total_mem`
  is not an attribute -- the real name is `total_memory` -- and a bare
  `except Exception` had been hiding that, so `gpu_memory_info()` returned
  None on a fully working GPU for as long as the function existed.

- **`except (ImportError, Exception)` is just `except Exception`.** It reads
  like a considered choice and catches everything, including the typo above.

- **Cache the hardware probe.** `get_array_module()` re-imported CuPy,
  allocated a probe array, and logged on every call, while the module's own
  `_device_cache` went unused. A 5-scale wavelet decomposition paid for five
  identical probes per image. The module-level cache existed; nothing used it.

- **Model identifiers rot, and the failure looks like a network problem.**
  `claude-sonnet-4-20250514` had been retired and returned 404 through a retry
  decorator that treated it as transient, so a permanent failure burned the
  full backoff schedule three times. Defaults now live in one registry
  (`llm/providers/models.py`) with the date each was verified, and the retry
  decorator short-circuits permanently-failing HTTP statuses. Current
  Anthropic models also reject `temperature` outright, so the provider omits
  the parameter rather than sending a value that returns 400.

### Approaches That Worked

- Fixing the statistics first and alone, before anything else, because it
  changed the numbers in every report and would have contaminated any other
  measurement taken afterwards.
- Reading the two abandoned TDD test files as an executable specification.
  `tests/test_gpu_backends.py` named 21 functions and asserted their exact
  contracts; implementing against it produced a working CUDA/ROCm/NPU layer
  with no design guesswork, verified on real hardware and again with
  `CUDA_VISIBLE_DEVICES=""` to exercise the CPU fallback.
- Checking wavelet correctness through the a-trous perfect-reconstruction
  property (`sum(details) + residual == original`, error 9e-16) rather than
  through a tolerance on a stored array.
- Gating the SDSS query on declination before the network call. The footprint
  tests went from live queries to 0.6 s, and an out-of-footprint region now
  reports "not covered" instead of an empty result.

### Ensemble Refactor and Scoring (same session, after the above)

- **Write the characterization test before the refactor, and let it fail.**
  The determinism gate in `tests/test_ensemble_characterization.py` refused
  to record a golden and exposed something bigger than the refactor:
  **sep 1.4.1's deblender is not reproducible**. Given a byte-identical
  array it returns identical source positions and counts, then assigns
  blended pixels to neighbours differently on every call. Measured over 40
  extractions of one frame: 16 of 38 sources have unstable `npix` and
  `flux` (worst 9.7%, median 3.1%), ellipticity moves by up to 0.184, and
  because `star_mask` is `ellipticity < 0.3`, **one source in 38 is
  classified a star on one run and a galaxy on the next**. Setting
  `deblend_cont=1.0` yields one distinct result across six runs, which
  isolates the deblender; thread count makes no difference. The ensemble
  `anomaly_score`, `n_detections` and every per-detector score are stable,
  so the discovery ranking is safe today, but three of the 65
  `rich_features` fed to the MetaDetector are not. Tolerances in that test
  are set above the *measured* spread, and `TestSepDeblenderInstability`
  fails loudly if sep is ever fixed, so the loosening gets reverted.

- **A weighted sum with no denominator is a hidden thumb on the scale.**
  `anomaly_score` summed thirteen weighted detector scores and never
  divided. A detector that could not run contributed zero while still
  occupying its share of the budget, so the score measured *how many
  detectors had inputs*, not how interesting the field was. Measured after
  renormalizing: an image with everything available is unchanged (0.4607,
  because the weights already sum to ~1), an image with no catalog rises
  51%, a genome with four detectors disabled rises 64%, and a sparse field
  rises 79%. The GA selects on a fixed `interest_threshold`, so every
  genome that switched a detector off was being penalized for the harness's
  arithmetic rather than for its detection quality.

- **Prove the refactor, do not assert it.** The registry rewrite cut
  `detect()` from 496 lines to 143 and removed all 16 broad
  `except Exception` handlers, and the goldens recorded from the old code
  passed unchanged. When the scoring fix then landed, the regenerated
  fixtures differed in exactly two leaves per scenario, `anomaly_score` and
  `rich_features[62]` — and feature 62 is `_top.anomaly_score`. Two
  numbers, both expected, nothing structural. That diff is the evidence the
  change did what it claimed and nothing else.

- **Narrow the exception net and programming errors stop hiding.**
  `RECOVERABLE_DETECTOR_EXCEPTIONS` covers the failures a detector can
  legitimately hit on degenerate input. An `AttributeError` from a typo or
  an `ImportError` from a missing optional dependency now propagates
  instead of becoming `{"error": ...}` and a silent zero for a whole run.

### Still Open

- **Saved elite genomes and the tuned `interest_threshold` are stale.** The
  score renormalization lifts every score computed on incomplete inputs, by
  up to 79% in the measured cases. Persisted GA populations and evolution
  history were scored on the old scale and are not comparable. Re-tune the
  threshold before reading fitness across the change.
- **`sep`'s non-reproducible deblender cannot be fixed by pinning.** Tested
  every installable build: 1.4.1, 1.4.0, and the sep-pjw 1.3.8 fork all
  produce byte-for-byte identical instability; 1.2.1 has no Python 3.12
  wheel and needs MSVC. No extract parameter fixes it either -- of
  `set_sub_object_limit`, `set_extract_pixstack`, `deblend_nthresh` and
  `deblend_cont`, the only reproducible settings are the ones that disable
  deblending (`deblend_nthresh=1`, `deblend_cont >= 0.5`), and those find
  26 sources on the test field where deblending finds 38. sep is now pinned
  to `==1.4.1` so the characterized behaviour cannot shift silently, but
  the pin buys stability of the defect, not absence of it.

  **APPLIED.** `SourceExtractor` now uses sep for detection only and
  measures flux and shape itself. Flux from `sep.sum_circle` at a fixed
  5px aperture; shape from flux-weighted second moments in a fixed window.
  Every output field is now exactly reproducible across runs.

- **A fixed aperture caps the extent you can measure, and that broke the
  classifier before the tests caught it.** The obvious implementation
  measures shape in the same circle used for photometry. Doing that made
  every source look equally round: ellipticity collapsed to 0.001-0.261
  and the star/galaxy split returned 38 stars out of 38, against 31 of 38
  for the sep parameters it replaced. The fix is a moment window wider
  than the photometric aperture. Measured across radii: 5px gives 38/38
  (degenerate), 8px gives 33/38, 10px gives 26/38, 20px gives 11/38.
  Settled on 1.6x the aperture, which is the closest non-degenerate match
  to the previous behaviour and has the highest correlation with sep's
  Kron radius (r = 0.77). `test_equal_radii_would_degenerate` records why
  the two radii differ so nobody collapses them back into one.

- **The cost of the change, stated plainly.** A 5px aperture recovers a
  median 92% of the segmentation flux and 64% at the 10th percentile, so
  extended sources are measured more conservatively than before.
  Brightness ranking still correlates with the old flux at r = 0.94. The
  golden diff confirmed the blast radius: `anomaly_score` and
  `n_detections` unchanged in all four scenarios, and only the `sources`
  arrays plus `rich_features[1, 2, 4]` moved. Absolute magnitudes are not
  comparable across this change; relative ordering essentially is.
- `_look_elsewhere_correction` still synthesizes Poisson inputs from
  `anomaly_score` itself. It no longer reaches any report, but it still gates
  verdicts, and re-deriving those thresholds is separate work.
- `llm/search_guide.py` now has no caller anywhere: not production, not
  tests. It is a deletion candidate, left in place because removing a module
  is the user's call rather than a cleanup.

## 2026-02-20 - Initial Build

### Architecture Decisions

- Chose src layout (`src/star_pattern/`) for proper package isolation and clean editable installs
- Python dataclasses for all config types instead of raw dicts -- type safety with minimal boilerplate
- Abstract base classes for DataSource and LLMProvider to allow clean extension without modifying existing code
- Provider auto-discovery from `*.key.txt` files -- zero config for adding new LLM providers, just drop the key file
- Ensemble detection pattern: run all detectors independently, combine with weighted scoring -- avoids coupling between detection methods
- 22-gene genome maps directly to DetectionConfig fields -- every tunable parameter is evolvable
- Adversarial debate with provider fallback: if the judge provider fails (rate limit, etc.), try the next available provider instead of crashing
- Minimum output token floor (1024) for Gemini 2.5 thinking models -- prevents internal reasoning from consuming the entire token budget

### Approaches That Worked

- Adapting Prime_Plot_AI's genetic algorithm genome for astronomical detection parameters (22 genes mapping to detection config fields)
- Reusing llm_compare's adversarial debate pattern for pattern validation (advocate/challenger/judge)
- Borda count consensus scoring across multiple LLM providers for significance rating
- SEP as primary source extraction with photutils fallback -- SEP is faster for batch work, photutils more flexible for edge cases
- SHA256-keyed data cache with JSON index -- deduplicates downloads across runs without relying on filenames
- Clark-Evans nearest-neighbor statistic for detecting stellar clustering -- simple, fast, interpretable
- Voronoi tessellation cell area coefficient of variation as a spatial uniformity measure
- CAS (Concentration-Asymmetry-Smoothness) + Gini + M20 for galaxy morphology -- well-established metrics that work on single-band images

### Lessons Learned

- Never use mocks in tests. Real provider discovery from key files with pytest.skip() when unavailable gives actual confidence that the code works
- Windows/MSYS environment requires Python-based hook scripts instead of bash+jq -- jq is not guaranteed to be installed
- FITSImage.from_array needs explicit float32 casting -- astropy FITS I/O is strict about dtypes and will silently produce wrong results with float64 in some operations
- Galactic coordinate conversion via astropy's SkyCoord handles the FK5->Galactic transform cleanly -- no need to implement the rotation matrix manually
- `gemini-2.0-flash` was discontinued in early February 2026 with all free tier quotas set to `limit: 0`. The error message says "quota exceeded" but the actual issue is the model no longer exists on the free tier. Fix: upgrade to `gemini-2.5-flash`
- Gemini 2.5 is a thinking model that uses output tokens for internal reasoning. A `max_output_tokens` of 10 will produce empty responses because all tokens are consumed by thinking. Enforce a minimum floor
- The `response.text` accessor on Gemini responses raises ValueError when the response has no content parts (finish_reason=MAX_TOKENS or SAFETY). Must check `response.candidates[0].content.parts` before accessing `.text`
- `pyproject.toml` build-backend must be `setuptools.build_meta`, not `setuptools.backends._legacy:_Backend` (which doesn't exist in setuptools 80+)
- Synthetic arc injection test: checking `modified.data.max() > original.data.max()` fails because the arc brightness (200) on background (~100) is still less than existing bright sources (~1038). Check `.sum()` instead to verify flux was actually added
- The debate judge role should cycle through all available providers on failure, not crash on the first 429

### Performance Insights

- Full test suite (97 tests including real SDSS network queries and real LLM API calls): ~120 seconds
- Non-LLM tests (90 tests): ~55 seconds
- SDSS catalog fetch for a 1 arcmin region: ~5-10 seconds (network dependent)
- Source extraction with SEP on a 256x256 image: <100ms
- Isolation Forest anomaly detection on 100 embeddings: <50ms

### Failed Attempts

- `setuptools.backends._legacy:_Backend` as build-backend -- doesn't exist in modern setuptools, caused pip install to fail with cryptic BackendUnavailable error
- `gemini-2.0-flash` as default Gemini model -- discontinued Feb 2026, returns `limit: 0` for all quotas
- Testing synthetic injection by comparing max pixel values -- flawed assumption that injected features would exceed the brightest existing source

## 2026-02-20 - Scope Expansion: All Pattern Types

### Architecture Decisions

- Expanded detection from 3 pattern types (lens, morphology, distribution) to 6 (adding galaxy interactions, kinematics, transients)
- New detectors operate on catalog data (StarCatalog) in addition to images (FITSImage) -- the ensemble now accepts an optional catalog parameter
- Genome expanded from 22 to 34 genes to include galaxy (3), kinematic (4), transient (2), and new ensemble weights (3)
- Ensemble weights expanded from 4 to 7 categories, rescaled proportionally to maintain sum ~1.0
- LLM classification expanded from 5 to 10 categories (added galaxy_interaction, kinematic_group, stellar_stream, variable, transient)
- Pipeline now merges all catalogs from RegionData and passes them to the ensemble detector

### Approaches That Worked

- DBSCAN clustering in proper motion (pmra, pmdec) space reliably finds co-moving groups in synthetic catalogs
- RANSAC-like line fitting in 4D (ra, dec, pmra, pmdec) detects stellar streams among field stars
- Robust sigma estimation (MAD * 1.4826) for runaway star detection -- median absolute deviation is resistant to the outliers being detected
- Smooth galaxy model subtraction (heavy Gaussian blur) + Gabor filters at large scales picks up tidal features
- Double-nucleus merger detection via local maxima finding + 180-degree rotation asymmetry measurement
- Magnitude-binned color outlier detection for catalog-based color anomalies
- Astrometric excess noise from Gaia catalog as a proxy for variability and unresolved binaries

### Lessons Learned

- Merger detection test with symmetrically-placed double nuclei fails because 180-degree rotation maps nuclei onto each other, yielding zero asymmetry. Test fixtures must place features asymmetrically relative to the cutout center
- When testing tidal features, the tidal arc must be strong enough relative to noise and placed at a scale compatible with the smooth model subtraction sigma
- Catalog-based detectors (proper motion, transient) need graceful degradation when catalog entries lack the expected properties -- always check for None before float conversion
- Galaxy feature detection on pure noise images should produce low scores -- tested and confirmed

## 2026-02-20 - Wide-Field Sky Coverage

### Architecture Decisions

- TileGrid uses hex-packed tiling with cos(dec) RA correction for efficient coverage of large sky areas
- Mosaicker wraps `reproject` (optional dependency) for WCS-aware image stitching
- WideFieldPipeline orchestrates tile decomposition, multi-source fetch, and mosaicking into a single RegionData
- Detectors are now pixel-scale-aware: EnsembleDetector extracts pixel_scale from FITSImage.pixel_scale() and passes it to LensDetector, ClassicalDetector, and GalaxyDetector
- DataConfig.sources default changed from ["sdss"] to ["sdss", "gaia", "mast"] to use all sources by default
- MAST max_observations made configurable (was hardcoded at 3)
- StarCatalog.merge() deduplicates by source_id when combining catalogs from overlapping tiles

### Approaches That Worked

- Hex-packed tiling with Vincenty formula for angular separation -- accurate at all declinations including near poles
- Moving WCS validation and single-image passthrough before the `reproject` import in Mosaicker -- lets tests pass without the optional dependency
- `FITSImage.pixel_scale()` needed to handle astropy Quantity objects from `proj_plane_pixel_scales()` -- extracting `.value` before `float()` conversion

### Lessons Learned

- `astropy.wcs.WCS.proj_plane_pixel_scales()` returns Quantity objects, not plain floats. `float()` fails on Quantities with units. Must use `.value` attribute to get the bare number
- The `reproject` library is a heavy optional dependency; validation and edge cases (empty list, single image, no WCS) should be handled before attempting the import
- When passing pixel_scale through to sub-detectors, use kwargs with None defaults so existing callers are not affected
- Merger detector O(n^2) pair comparison explodes on large images: a 1489x2048 image at 95th percentile produces thousands of peaks, yielding 463k merger candidates. Fix: cap peaks to brightest 200, use cKDTree for neighbor search, cap output to 50
- `ndimage.generic_filter(data, np.std, size=N)` is catastrophically slow for large N: it calls a Python function at every pixel. On 1489x2048 with size=102, it takes 250+ seconds. Fix: compute local std via `uniform_filter` on data and data^2, derive std = sqrt(E[x^2] - E[x]^2) -- all in C, ~1 second
- HoughArcDetector with 90 radii x 360 thetas x 150k edge points is slow on large images. Fix: downsample to max 512px, subsample edge points to 5000, use 72 thetas (5-degree steps), step radii by max(1, range//30)
- LensDetector creating full-image mgrid arrays for arc/ring detection is wasteful. Fix: extract a square cutout around the central source (radius = ring_max_radius + padding) and work in local coordinates
- Survey-resolution SDSS frames (1489x2048) have many bright sources that saturate merger detection. Always cap peak count before O(n^2) operations

## 2026-02-20 - Astronomical Detection Methods and Learning/Evolution

### Architecture Decisions

- Added 3 well-known astronomical detection methods: Sersic profile fitting, a-trous wavelet multi-scale analysis, and stellar population CMD analysis
- Genome expanded from 34 to 43 genes (6 new detector params + 3 new ensemble weights)
- Fitness function expanded from 4 to 5 components, adding synthetic injection recovery (0.15 weight)
- Ensemble expanded from 9 to 12 detectors with 10 configurable weights
- ActiveLearner enhanced with closed-loop learning: feedback-driven IsolationForest retraining, ensemble weight learning via Pearson correlation, persistence across sessions
- Evolutionary discovery enhanced with adaptive mutation rate (0.05-0.5 bounds) and experience replay (top genomes persisted to JSON across runs)
- Added 10 preset genomes (was 7) including sersic-focused, wavelet-focused, and CMD/population-focused

### Approaches That Worked

- A-trous wavelet decomposition with B3 spline kernel [1,4,6,4,1]/16 for multi-scale astronomical source detection. Perfect reconstruction property makes validation straightforward (sum of details + smooth = original)
- MAD * 1.4826 noise estimation per wavelet scale -- robust to source contamination unlike standard deviation
- Sersic profile fitting via 1D radial profiles (azimuthally averaged in elliptical annuli) + scipy.optimize.curve_fit. Much faster than full 2D fitting and sufficient for morphology classification
- Density-based MS turnoff estimation: use the first bright magnitude bin with >= 30% of peak bin count. Prevents sparse blue stragglers from biasing the turnoff
- Experience replay: persisting top genomes to JSON and loading them in future runs dramatically improves initial population quality
- Adaptive mutation bounded between 0.05-0.5 with stagnation counter prevents both premature convergence and runaway mutation
- Pearson correlation between detector scores and interest labels for ensemble weight learning -- simple, interpretable, and works with small datasets

### Lessons Learned

- Blue straggler detection requires density-based turnoff estimation. Using the single brightest star as the turnoff fails when blue stragglers exist -- they ARE the brightest/bluest stars, so they define the turnoff they're supposed to be detected relative to. Fix: use magnitude bin counts to find where the well-populated main sequence begins
- Wavelet smoothing tests need large images (256x256+) for higher scales. At scale 2, the dilated kernel spans 17 pixels; on a 64x64 image, boundary effects prevent further std reduction. Use images large relative to the kernel at the highest tested scale
- Sersic n=4 (de Vaucouleurs) profiles have shallower outer falloff than n=1 (exponential). A test asserting n=4 is fainter than n=1 at r=2*r_e fails because the heavy tail of n=4 makes it brighter at large radii. Only compare near center or at r=r_e (where both equal I_e by definition)
- Synthetic catalog construction for CMD tests requires clear magnitude separation: MS at mag 15-21, BS at mag 12-14.5, RGB at mag 12-15. Without gaps, the running median algorithm merges populations
- IsolationForest retraining needs both positive AND negative examples (>= 5 each). The contamination parameter should be derived from the label ratio, not hardcoded
- Feature vector dimensionality must match across extraction and novelty/diversity computation. Extending from 9 to 12 dimensions (adding sersic, wavelet, population scores) requires updating the extraction function, not just the score computation

### Performance Insights

- Full test suite expanded to 218 tests: ~130 seconds
- Sersic profile fitting on 128x128 synthetic galaxy: <500ms
- Wavelet 5-scale decomposition on 128x128 image: <100ms
- CMD analysis on 100-star catalog: <50ms
- Experience replay load/save: <10ms (JSON with numpy arrays)

## 2026-02-20 - LLM-as-Strategist Architecture

### Architecture Decisions

- Restructured LLM integration from per-detection calls to periodic batch strategy sessions
- LocalClassifier replaces HypothesisGenerator for routine classification (rule-based, zero tokens)
- LocalEvaluator replaces PatternDebate for routine evaluation (SNR/agreement-based, zero tokens)
- StrategyAdvisor provides periodic strategic guidance: parameter adjustments, weight changes, focus regions
- TokenTracker enforces per-session budget across all LLM calls
- LLMCache prevents redundant calls via SHA256-keyed response caching with TTL
- LLMProvider base class extended with generate_tracked() and generate_cached() methods
- LLMConfig extended with token_budget (500k default), strategy_interval (25 cycles), max_debate_tokens
- EvolutionaryDiscovery extended with apply_strategy_to_population() for LLM-guided genome variants
- ActiveLearner extended with get_strategy_summary() and apply_strategy() for closed-loop learning
- Legacy LLM components (HypothesisGenerator, PatternDebate, etc.) kept for escalation cases

### Approaches That Worked

- DETECTOR_TO_CLASS mapping + FOLLOW_UP_TEMPLATES: simple lookup tables replace expensive LLM calls for 99% of detections
- Ambiguity detection via top-2 score gap: if the gap between the two highest detector scores is < 0.15, the detection is ambiguous and worth LLM review
- Novelty detection: no cross-matches AND high confidence (> 0.6) flags genuinely novel findings for LLM review
- Compact batch summaries: pipeline state compressed to ~500 tokens for strategy sessions
- 70/30 weight blending (evolved/LLM) prevents LLM from dominating the optimization
- Strategy outcome tracking: recording pre/post metrics lets the LLM learn from its own advice
- SHA256-keyed caching: identical prompts (same pipeline state) return cached strategy without API calls

### Lessons Learned

- Per-detection LLM calls are wasteful: most detections are routine and classifiable by deterministic rules. The LLM adds value at the strategic level (batch review, parameter tuning) not at the individual detection level
- Token estimation via len(text)/4 is sufficient for budget pre-checks; exact tokenization is unnecessary for planning
- The anomaly_score from the detection dict is also extracted by the LocalClassifier via the fallback path, which can make top-2 gap calculations behave differently than expected. Account for all score sources when testing ambiguity
- Test fixtures for evaluator tests need realistic SNR values: a peak of 120 on background 100 with noise 10 gives SNR ~1.3 (below the artifact threshold), while peak 5000 gives SNR ~330 (clearly real)

### Performance Insights

- Full test suite: 275 tests in ~115 seconds
- LocalClassifier.classify(): <1ms per detection
- LocalEvaluator.evaluate(): <5ms per detection (includes SNR computation)
- Token consumption: ~160-200 tokens/cycle (down from ~20,700), ~99% reduction
- Strategy session: ~1,000-2,500 tokens every 25 cycles
- New component tests: 57 tests in ~3 seconds

## 2026-02-20 - HEALPix Grid Survey Mode

### Architecture Decisions

- Added HEALPixSurvey class for systematic sky coverage using HEALPix equal-area pixelization
- healpy is an optional dependency (lazy imported with clear error message)
- Survey state persisted to JSON for cross-session resume (visited pixels, pending queue, findings counts)
- Region selection priority chain: strategy suggestions > survey grid > random
- Three visit ordering strategies: galactic_latitude (default, high |b| first), random_shuffle, dec_sweep
- SurveyConfig added as a dataclass alongside existing WideFieldConfig pattern

### Approaches That Worked

- Using healpy.pix2ang for pixel center coordinates then converting through SkyRegion for galactic latitude filtering -- reuses existing coordinate infrastructure
- Attaching pixel index as a dynamic attribute on SkyRegion (_healpix_pixel) to thread it through the pipeline without modifying the SkyRegion dataclass
- Simple JSON state persistence: {visited, pending, findings_per_pixel, config} -- sufficient for resume without complexity
- Filtering galactic plane at survey initialization rather than per-query -- builds the full filtered pixel list once

### Lessons Learned

- healpy does not provide prebuilt wheels for Windows and requires C compilation (cfitsio, HEALPix C++). Use astropy-healpix instead -- it has cross-platform wheels and integrates natively with astropy coordinate frames
- astropy_healpix.HEALPix requires a frame parameter (e.g. ICRS()) to use healpix_to_skycoord(). Without it, raises NoFrameError. Use healpix_to_lonlat() as the frame-independent alternative
- Vectorized pixel coordinate conversion (all pixels at once) is much faster than per-pixel loops through SkyRegion -- use numpy arrays with astropy_healpix's batch operations for filtering and sorting
- At NSIDE=64 with min_galactic_lat=20, roughly 60% of pixels survive filtering (~30,000 out of 49,152)
- Survey state save integrated at checkpoint intervals (not every cycle) to avoid I/O overhead

## 2026-02-20 - Time-Domain Variability Detection

### Architecture Decisions

- Added ZTFDataSource for light curve data via IRSA TAP service (g/r/i bands)
- Light curves stored in CatalogEntry.properties["ztf_lightcurve"] as {band: [(mjd, mag, magerr), ...]}
- VariabilityAnalyzer added as 13th detector in the ensemble, operating on catalog light curves
- Variability analysis uses three complementary methods: variability indices (chi2, eta, MAD), Lomb-Scargle periodograms (astropy.timeseries), and outburst detection (MAD-based thresholding)
- Variable star classification is deterministic/rule-based (matches project convention of local classification)
- Genome expanded from 43 to 48 genes (4 variability params + 1 weight)
- Ensemble weights rebalanced from 11 to 12 categories (added variability=0.09)
- Added TNS (Transient Name Server) to cross-reference pipeline alongside SIMBAD/NED
- ZTF is enabled by default; --no-ztf CLI flag to disable

### Approaches That Worked

- Bulk TAP query for all light curves in a region, then group by object ID -- much faster than per-source queries
- Von Neumann eta ratio for distinguishing correlated variability from random noise (correlated -> low eta, random -> eta ~ 2)
- MAD * 1.4826 for robust sigma estimation in outburst detection -- same approach used in wavelet analysis, consistent across codebase
- astropy.timeseries.LombScargle for periodogram computation -- already a dependency, handles unevenly sampled data natively
- Rule-based variable classification from period/amplitude/shape -- matches the LocalClassifier pattern, zero tokens
- Storing light curves in CatalogEntry.properties rather than a separate data structure -- fits the existing data flow through EnsembleDetector without interface changes

### Lessons Learned

- Monotonic fading light curves do not trigger outburst detection via MAD-based thresholds. The MAD of a linear ramp is ~1/4 of the total range, so individual points rarely exceed 3 sigma. Transient-like signatures need a sharp deviation from baseline, not a gradual trend
- ZTF data release table names include the release number (ztf_objects_dr22, ztf_dr22). This needs updating as new releases come out
- The IRSA TAP service via astroquery.ipac.irsa.Irsa is available in astroquery >= 0.4.6 (already a core dependency), no new packages needed for ZTF access
- False alarm probability from LombScargle.false_alarm_probability() can fail on edge cases -- always wrap in try/except with fallback to fap=1.0
- When testing transient detection, use a constant baseline with injected sharp events rather than a monotonic fade, which distributes deviations evenly and prevents any single point from being an outlier

### Performance Insights

- Full test suite expanded to 313 tests: ~170 seconds
- Variability analysis on 3-source catalog with 200-epoch light curves: <4 seconds (including Lomb-Scargle)
- Lomb-Scargle periodogram with 10,000 frequency points: <500ms
- ZTF IRSA TAP query for 3 arcmin region: ~30 seconds (network dependent)
- 18 variability tests + 6 ZTF tests: ~40 seconds

## 2026-02-21 - Genuine Pattern Discovery System + Catalog Caching + Fast Shutdown

### Architecture Decisions

- Added 4 new systems for genuine pattern discovery beyond fixed detectors:
  1. FeatureFusionExtractor: extracts ~60-D cross-detector feature vectors from detection results (all data already computed, pure extraction)
  2. MetaDetector: learned non-linear scoring with progressive complexity (linear -> GBM at 50 labels -> neural net at 200 labels)
  3. RepresentationManager: orchestrates existing ML infra (BackboneWrapper, FeatureExtractor, SSLPretrainer, EmbeddingAnomalyDetector) into the pipeline
  4. ComposedPipeline: variable-length detection pipelines (2-5 ops) from 10 primitive operations, evolved via PipelineGenome
- Genome expanded from 48 to 54 genes (6 new: meta_blend_weight, meta_gbm_depth, meta_gbm_estimators, repr_anomaly_contamination, repr_weight, composed_weight)
- PipelineGenome is a separate variable-length genome co-evolved alongside DetectionGenome
- Added CatalogEntry.to_dict()/from_dict() serialization for catalog caching across runs
- Wired catalog caching into all 4 data sources (SDSS, Gaia, MAST, ZTF) -- avoids redundant network calls on repeated runs
- DataCache uses band="__catalog__" key to distinguish catalog entries from FITS image cache entries
- Fast SIGINT shutdown: first CTRL-C stops after current phase (not cycle), second CTRL-C raises SystemExit(1) for immediate exit
- Shutdown checks added between every phase in the main loop and inside _process_region and _evolve_parameters
- All new systems have enabled: bool config flags -- disabled = identical to previous behavior

### Approaches That Worked

- Progressive meta-detector complexity (linear/GBM/NN) matched to label count avoids overfitting with small data while enabling full non-linear learning with enough labels
- Feature fusion from existing detector outputs -- zero additional computation, just extraction of already-computed intermediate values
- JSON catalog caching with SHA256 keys, same pattern as FITS image caching -- simple, consistent, works across sessions
- try/except wrappers around cache checks in all data sources -- cache failures should never break data fetching
- 8 preset pipeline genomes seeding the compositional pipeline population -- same pattern as 11 preset detection genomes
- Two-tier SIGINT handling (graceful phase exit + forced exit) gives users control over urgency

### Lessons Learned

- CatalogEntry serialization must handle None values (e.g. mag=None) and nested structures (ZTF light curve dicts with lists of tuples). JSON handles tuples as lists of lists, which is acceptable since downstream code accesses by index
- Hook rule 5 (API error handling) triggers on any function matching fetch_catalog pattern, even cache check code. Wrapping cache checks in try/except satisfies the hook and is good practice anyway
- SIGINT handler raising SystemExit on second press works on both Unix and Windows/MSYS
- Shutdown checks between phases (fetch -> detect -> save -> strategy -> evolution) are more responsive than checking only at cycle boundaries -- cycles can run over an hour
- DetectionConfig.from_genome_dict() handles old (48-gene) and new (54-gene) genomes via .get() defaults, maintaining backward compatibility
- All 4 new systems must be independently disableable via config flags for backward compatibility and debugging

### Performance Insights

- Full test suite expanded to 387 tests: ~190 seconds
- Catalog cache hit eliminates 5-30 second network calls per data source per region
- Feature fusion extraction from detection dict: <1ms (pure dict lookups + numpy array construction)
- MetaDetector scoring with GBM: <5ms per sample
- ComposedPipeline run (5-op pipeline on 256x256 image): <50ms

## 2026-02-22 - Per-Anomaly Findings, Reporting, and Mosaic Visualization

### Architecture Decisions

- Added Anomaly dataclass to metrics.py: per-detection findings with type, detector, pixel/sky coords, score, properties
- PatternResult gains anomalies list populated by _extract_anomalies() in autonomous.py
- Per-detector score normalization to [0, 1] before global ranking (_MAX_PER_DETECTOR = 8 cap prevents one noisy detector from consuming all slots)
- Anomaly-centric mosaic: cutouts centered on individual anomalies (not full-field per-finding panels)
- Report includes per-anomaly markdown table with location, type, detector, score, properties
- Signal quality filter in mosaic skips point-source anomalies with no detectable source at center
- Extended feature recentering: tidal feature cutouts snap to nearest bright pixel so the source generating the gradient is visible

### Approaches That Worked

- Per-detector score normalization solves the cross-detector ranking problem: galaxy "strength" (raw pixel flux ~100k) vs overdensity "sigma" (~1-10) vs wavelet "n_scales" (~1-5). Normalize each detector to [0, 1], then global sort produces diverse anomaly mix
- Storing raw_score in properties before normalization allows the report to display physically meaningful values (SNR for lens arcs, sigma for overdensities) while using normalized scores for ranking
- Signal quality check (_has_source_at_center) with 2-sigma threshold effectively filters noise detections (wavelet false positives, gradient artifacts at empty sky positions) from mosaic panels
- Extended feature recentering via brightest-pixel search within feature radius: tidal feature centroids are in faint sky between source and background; snapping to the brightest nearby pixel shows the source generating the gradient
- Cutout radius scaling by feature area (sqrt(area) * 0.75, clamped [60, 400]) properly frames tidal features that span hundreds of pixels
- Low-contrast ZScale stretch (contrast=0.15) for extended features reveals diffuse structure that's invisible at default contrast

### Lessons Learned

- Always inspect actual output (mosaic.png, report.json) after visual changes. Three rounds of blind coordinate fixes failed to resolve mosaic issues because the real root causes (score scale mismatch, tiny cutouts, noise detections) were only visible in the rendered output
- Different detectors produce scores on completely incompatible scales. Galaxy detector "strength" is raw pixel flux sums (100k+), while overdensity "sigma" is statistical significance (1-10). Without normalization, one detector dominates all ranking slots
- Tidal feature centroids from gradient detection land in featureless sky between the bright source and background. Centering a cutout there shows uniform noise. The fix is recentering on the nearest bright pixel, which shows the source with the gradient visible around it
- The galaxy detector's tidal features at quantized orientations (0, pi/4, pi/2, 3pi/4) with very large areas (10k-30k px) are mostly background gradient artifacts, not real tidal tails. Per-detector cap limits their impact
- SDSS FITS images are fully populated (no NaN, no zero borders) -- "blank" cutouts are regions of uniform sky noise, not missing data
- Extended features (tidal tails, sersic residuals) are diffuse structures that fail point-source signal checks. They need different rendering: larger cutouts, aggressive contrast stretch, no center-brightness filter
- Detector output key names must match extraction code exactly: ClassicalDetector uses center_x/center_y (not cx/cy), ProperMotionAnalyzer uses mean_ra/mean_dec (not center_ra/center_dec), SersicAnalyzer uses peak_snr (not snr)

### Performance Insights

- Full test suite expanded to 422 tests (66 in test_pipeline.py): ~32 seconds for pipeline tests
- Per-anomaly extraction from detection dict: <5ms per region
- Mosaic generation with 24 cutout panels + signal quality checks: ~4 seconds
- Typical discovery run (--hours 0.1): 6-12 cycles, 4-6 findings, 24-28 anomalies per finding

## 2026-02-22 (Session 2) - Catalog Detector Data Source Fixes

### Bug: Property Key Mismatches

Three catalog-based detectors always produced zero scores and blank overlay images:

1. **StellarPopulationAnalyzer**: Looked for `properties["g_r"]` but SDSS stores individual magnitudes as `properties["g"]` and `properties["r"]` without computing the color index. Fix: compute `g - r` from individual bands when neither `bp_rp` nor `g_r` is available.

2. **TransientDetector**: Photometric outlier code looked for `properties["phot_bp_mean_mag"]` (the Gaia column name) but GaiaDataSource stores BP magnitude as `properties["BP"]` (short key). Fix: check both key names, plus add SDSS g-r fallback for photometric outlier detection.

3. **ProperMotionAnalyzer**: Key names were correct (`pmra`, `pmdec`) but regions with only SDSS data have zero proper motion sources. This is a genuine data limitation, not a bug. Added diagnostic logging to surface "0 from Gaia" when kinematic analysis fails.

### Lesson Learned

- Property key names between data sources and detectors must be verified end-to-end. The Gaia TAP query fetches `phot_bp_mean_mag` but stores it as `"BP"` in the CatalogEntry properties dict. The TransientDetector was looking for the original column name instead of the stored key name. This kind of mapping mismatch is invisible in unit tests that construct entries directly with the expected keys.
- When a detector works fine in unit tests but always returns zero in integration runs, check the actual property keys in the data source, not just the detector code.

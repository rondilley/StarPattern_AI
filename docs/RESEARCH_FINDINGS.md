# Research Findings: AI/ML Pattern Detection in Astrophotography and Star Fields

**Date**: 2026-02-20
**Status**: Comprehensive literature and tool survey

---

## Table of Contents

1. [Published Papers & Research](#1-published-papers--research)
2. [Existing Tools & Software](#2-existing-tools--software)
3. [Data Sources & APIs](#3-data-sources--apis)
4. [Pattern Types in Star Fields](#4-pattern-types-in-star-fields)
5. [Algorithms & Approaches](#5-algorithms--approaches)
6. [Hypothesis Refinement](#6-hypothesis-refinement)
7. [Actionable Next Steps](#7-actionable-next-steps)

---

## 1. Published Papers & Research

### 1.1 Comprehensive Surveys

| Paper | Key Finding | URL |
|-------|-------------|-----|
| **Machine Learning in Stellar Astronomy: Progress up to 2024** (arXiv, Feb 2025) | Comprehensive survey of ML in stellar classification, parameter inference, and pattern recognition. Publications surged from <5/year pre-2019 to 25+/year by 2022. | https://arxiv.org/html/2502.15300v1 |
| **Review of AI Applications in Astronomical Data Processing** (2024) | Neural network for pulsar identification (SPINN) achieves 0.64% false-positive rate, reducing candidates by 4 orders of magnitude. | http://www.ati.ac.cn/en/article/pdf/preview/10.61977/ati2024001.pdf |
| **A Review of Unsupervised Learning in Astronomy** (2024) | Surveys unsupervised methods across clustering, dimensionality reduction, and density estimation for astronomical data. | https://arxiv.org/html/2406.17316v1 |

### 1.2 Anomaly Detection

| Paper | Key Finding | URL |
|-------|-------------|-----|
| **Astronomaly at Scale: Searching for Anomalies Amongst 4 Million Galaxies** (MNRAS, 2024) | Applied active anomaly detection to ~4M galaxy images from DECaLS. Found 1,635 anomalies in top 2,000 sources including 8 strong gravitational lens candidates and 1,609 merger candidates. | https://academic.oup.com/mnras/article/529/1/732/7612998 |
| **Anomaly Detection to Identify Transients in LSST Time Series Data** (MNRAS, 2024) | Active Anomaly Discovery (AAD) algorithm tailored for LSST broker integration. Uses real-bogus ML classifiers to improve filtering. | https://academic.oup.com/mnras/article/543/1/351/8249279 |
| **Exploring the Universe with SNAD: Anomaly Detection in Astronomy** (2024) | Two-step pipeline: ML anomaly detection + human expert analysis across wavelengths. Uses Isolation Forest, Active Anomaly Discovery, PineForest. | https://arxiv.org/html/2410.18875v1 |
| **Anomaly Detection in Hyper Suprime-Cam Galaxy Images with GANs** (MNRAS) | WGAN learns distribution of normal galaxy images; anomalies detected via reconstruction error and discriminator features. | https://academic.oup.com/mnras/article/508/2/2946/6369368 |
| **Detecting Outliers in Astronomical Images with Deep Generative Networks** (MNRAS) | Deep generative networks for unsupervised anomaly detection in galaxy images. | https://dx.doi.org/10.1093/mnras/staa1647 |
| **The ROAD to Discovery: ML Anomaly Detection in Radio Astronomy Spectrograms** (A&A, 2023) | Context-prediction self-supervised loss for LOFAR spectrograms. ROAD outperforms autoencoding models. | https://www.aanda.org/articles/aa/full_html/2023/12/aa47182-23/aa47182-23.html |

### 1.3 Galaxy Morphology Classification

| Paper | Key Finding | URL |
|-------|-------------|-----|
| **Galaxy Morphological Classification with Deformable CNNs** (AJ, 2024) | Pruned ResNet-18 with attention layers achieves 94.5% accuracy on GZ DECaLS 7-class taxonomy. | https://iopscience.iop.org/article/10.3847/1538-3881/ad10ab |
| **Extended Catalogue of Galaxy Morphology Using Deep Learning in S-PLUS DR3** (MNRAS, 2024) | Large-scale morphological catalog generated with deep learning. | https://academic.oup.com/mnras/article/528/3/4188/7492270 |
| **Morphological Classification of Galaxies: Comparing 3-way and 4-way CNNs** | 83% accuracy (3-class) and 81% (4-class) on 14,034 SDSS images. Custom CNN outperforms existing architectures. | https://arxiv.org/abs/2106.01571 |
| **Improved Galaxy Morphology Classification with CNNs** (Universe, 2024) | Zhu et al. achieved 95.2% accuracy using ResNet on 28,790 images. | https://www.mdpi.com/2218-1999/10/6/230 |

### 1.4 Gravitational Lensing Detection

| Paper | Key Finding | URL |
|-------|-------------|-----|
| **ML-Based Gravitational Lens Identification with LOFAR** (MNRAS) | CNN achieves 95.3% true positive rate with 0.008% false positive rate for lensed events. | https://academic.oup.com/mnras/article/517/1/1156/6649832 |
| **Detecting Gravitational Lenses: Interpretability and Sensitivity** (MNRAS) | F1 scores 0.83-0.91; 76% recall for compound arcs, 52% for double rings. | https://academic.oup.com/mnras/article/512/3/3464/6544650 |
| **DeepGraviLens: Multi-Modal Architecture** (Neural Computing and Applications, 2023) | Multi-modal approach combining image and metadata for lens classification. | https://link.springer.com/article/10.1007/s00521-023-08766-9 |
| **Finding Strong Gravitational Lenses Through Self-Attention** (A&A, 2022) | Self-attention encoders outperform CNNs by high margin for lens detection. | https://www.aanda.org/articles/aa/full_html/2022/08/aa42463-21/aa42463-21.html |
| **Substructure Detection in Realistic Strong Lensing with ML** (2024) | ML applied to detect dark matter substructure in lensing images. | https://arxiv.org/html/2401.16624 |

### 1.5 Stellar Streams and Tidal Features

| Paper | Key Finding | URL |
|-------|-------------|-----|
| **Stream Automatic Detection with CNNs** (2025) | CNNs automate detection of faint stellar streams in galactic halos. Requires surface brightness limits >29 mag/arcsec^2 in r-band. | https://arxiv.org/html/2503.17202 |

### 1.6 Transient Detection

| Paper | Key Finding | URL |
|-------|-------------|-----|
| **AI Breakthrough for Cosmic Events (Oxford/Google Cloud)** (Nature Astronomy, 2025) | LLM classifies cosmic events with ~93% accuracy using only 15 example images. Human panel rated descriptions as highly coherent. | https://www.ox.ac.uk/news/2025-10-08-ai-breakthrough-helps-astronomers-spot-cosmic-events-just-handful-examples |
| **Real-Time FRB Detection with Deep Learning** (Breakthrough Listen/NVIDIA, 2025) | 7% better accuracy than existing pipelines, 10x fewer false positives, 600x speed improvement. | https://astrobiology.com/2025/11/revolutionary-ai-system-achieves-600x-speed-breakthrough-in-detection-of-signals-from-space.html |
| **Real-time Detection of Anomalies in Large-Scale Transient Surveys** (MNRAS) | Framework for real-time anomaly detection in survey data streams. | https://academic.oup.com/mnras/article/517/1/393/6705438 |
| **Real-time Light Curve Classification with Modified Semi-supervised VAE** (AJ, 2025) | Semi-supervised variational autoencoder for WFST light curve classification. | https://iopscience.iop.org/article/10.3847/1538-3881/adcac0 |

### 1.7 Self-Supervised & Foundation Models

| Paper | Key Finding | URL |
|-------|-------------|-----|
| **Enabling Unsupervised Discovery Through Self-Supervised Representations** (MNRAS, 2024) | BYOL applied to Galaxy Zoo DECaLS images. Self-supervised features enable both morphology grouping and anomaly detection without labels. | https://academic.oup.com/mnras/article/530/1/1274/7640046 |
| **AstroCLIP: Cross-Modal Foundation Model for Galaxies** (MNRAS, 2024) | Vision transformer + DINOv2 creates shared latent space for galaxy images and spectra. First cross-modal astronomical foundation model. | https://academic.oup.com/mnras/article/531/4/4990/7697182 |
| **AstroDINO: Self-Supervised Learning on Astronomical Images** (Stanford CS231n, 2025) | Applies DINO framework to astronomical image representation learning. | https://cs231n.stanford.edu/2025/papers/text_file_840589796-AstroDINO_final.pdf |
| **Towards an Astronomical Foundation Model for Stars** (MNRAS, 2024) | Transformer-based model trained across multiple surveys for stellar inference tasks. | https://academic.oup.com/mnras/article/527/1/1494/7291945 |

### 1.8 Transfer Learning

| Paper | Key Finding | URL |
|-------|-------------|-----|
| **Transfer Learning for Transient Classification: ZTF to LSST** (MNRAS Letters, 2025) | ZTF pretrained model achieves equivalent performance with only 5% labeled data. 94% baseline performance with 30% training data for LSST. | https://academic.oup.com/mnrasl/article/542/1/L132/8193425 |
| **Examining Vision Foundation Models for Classification and Detection in Astronomy** (A&A, 2025) | Foundation model transfer learning outperforms training from scratch even with frozen backbone. | https://www.aanda.org/articles/aa/full_html/2025/11/aa53691-25/aa53691-25.html |
| **Leveraging Transfer Learning for Astronomical Image Analysis** (2024) | ResNet152V2 achieves 92.09% accuracy classifying galaxies, stars, quasars from SDSS. | https://arxiv.org/html/2411.18206v1/ |
| **Effective Fine-Tuning of Vision-Language Models for Galaxy Morphology** (2024) | Vision-language model fine-tuning reduces training epochs by 83% with 12% accuracy improvement. | https://arxiv.org/html/2411.19475v1 |

### 1.9 Cosmic Web Structure Detection

| Paper | Key Finding | URL |
|-------|-------------|-----|
| **Detecting Filamentary Pattern in the Cosmic Web: SDSS Catalogue** (MNRAS) | Algorithms for detecting cosmic web filaments from galaxy distributions. | https://academic.oup.com/mnras/article/438/4/3465/1107139 |
| **The Persistent Cosmic Web and Its Filamentary Structure** (MNRAS) | DisPerSE algorithm for scale-free, parameter-free identification of voids, walls, filaments, clusters. Uses discrete Morse theory. | https://academic.oup.com/mnras/article/414/1/350/1090746 |
| **Cosmic Web Classification Through Stochastic Topological Ranking** (2025) | ASTRA algorithm classifies galaxies into cosmic web structures for large spectroscopic surveys. | https://ui.adsabs.harvard.edu/abs/2025RASTI...4...32F/abstract |

### 1.10 Diffusion Models in Astronomy

| Paper | Key Finding | URL |
|-------|-------------|-----|
| **Radio-Astronomical Image Reconstruction with Conditional DDPM** (A&A, 2024) | Conditional denoising diffusion model for deconvolution of radio astronomy dirty images. | https://www.aanda.org/articles/aa/full_html/2024/03/aa47948-23/aa47948-23.html |
| **Galaxy Image Super-Resolution with Diffusion Network** (2024) | GD-Net applies global attention DDPM for galaxy image super-resolution. | https://www.sciencedirect.com/science/article/abs/pii/S095219762401995X |

---

## 2. Existing Tools & Software

### 2.1 Core Astronomy Python Ecosystem

| Tool | Description | Install / URL |
|------|-------------|---------------|
| **AstroPy** | Core package for astronomy in Python. FITS I/O, coordinate systems, unit conversions, cosmology calculations. | `pip install astropy` / https://www.astropy.org/ |
| **Photutils** | Astropy-affiliated package for source detection, aperture/PSF photometry, background estimation, image segmentation, radial profiles. | `pip install photutils` / https://photutils.readthedocs.io/ |
| **Astroquery** | Unified Python interface to query 30+ astronomical databases (SDSS, Gaia, VizieR, MAST, SkyView, etc.). | `pip install astroquery` / https://astroquery.readthedocs.io/ |
| **SEP** | Python/C library wrapping Source Extractor algorithms. Operates on in-memory arrays (no FITS/config files). Fast source detection and photometry. | `pip install sep` / https://github.com/kbarbary/sep |
| **reproject** | Reprojection of astronomical images between coordinate systems. | `pip install reproject` |

### 2.2 Source Detection & Astrometry

| Tool | Description | URL |
|------|-------------|-----|
| **SExtractor (Source Extractor)** | Gold standard for source extraction from astronomical images. Includes neural-network-based star/galaxy separation. Written in C. | https://github.com/astromatic/sextractor |
| **Astrometry.net** | Automatic astrometric calibration. Submit unknown image, receive WCS solution + object list. | https://github.com/dstndstn/astrometry.net / http://nova.astrometry.net/ |
| **DAOStarFinder** | Part of Photutils. DAOPHOT-style star finding via local density maxima fitting. | Included in Photutils |
| **StarFinder / IRAFStarFinder** | PSF-fitting star detection algorithms within Photutils. | Included in Photutils |

### 2.3 ML-Based Astronomical Analysis Tools

| Tool | Description | URL |
|------|-------------|-----|
| **Astronomaly** | Framework for active anomaly detection in astronomical data (images, light curves, spectra). Python backend + JS frontend. Easily extensible. | https://github.com/MichelleLochner/astronomaly |
| **ZooBot** | Pretrained Bayesian CNN for galaxy morphology. Fine-tunable with minimal data. PyTorch + TensorFlow. Trained on millions of Galaxy Zoo labels. | https://github.com/mwalmsley/zoobot / https://zoobot.readthedocs.io/ |
| **AstroCLIP** | Cross-modal foundation model. Embeds galaxy images + spectra into shared latent space. Uses DINOv2 backbone. | https://github.com/PolymathicAI/AstroCLIP |
| **SNAD Tools (coniferest)** | Anomaly detection package with Isolation Forest, Active Anomaly Discovery, PineForest algorithms. | https://github.com/snad-space/zwad |
| **YOLO-CIANNA** | YOLO-inspired source detector customized for astronomical images. Handles high-density small objects in radio data. | https://arxiv.org/abs/2402.05925 |
| **DeepGraviLens** | Multi-modal architecture for classifying gravitational lensing data. | https://link.springer.com/article/10.1007/s00521-023-08766-9 |
| **DisPerSE** | Discrete Persistent Structures Extractor. Scale-free identification of cosmic web features (voids, walls, filaments, clusters). | http://www2.iap.fr/users/sousbie/web/html/indexd41d.html |

### 2.4 Citizen Science Platforms

| Platform | Description | URL |
|----------|-------------|-----|
| **Zooniverse** | 2.7M+ registered volunteers, 450+ peer-reviewed publications. Hosts Galaxy Zoo and dozens of astronomy projects. | https://www.zooniverse.org/ |
| **Galaxy Zoo** | Galaxy morphology classification by citizen scientists. Dataset of ~1M SDSS galaxies. Galaxy Zoo 2: 243,434 labeled images. Now integrated with ZooBot AI. | https://www.zooniverse.org/projects/zookeeper/galaxy-zoo |
| **SNAD Viewer** | Centralized multi-dimensional view of astronomical objects from ZTF data releases for anomaly review. | https://snad.space/ |

### 2.5 Image Processing for Astrophotography

| Tool | Description |
|------|-------------|
| **StarXTerminator** | Deep learning star detection/removal using CNN trained on astronomical images. |
| **GraXpert** | AI-automated gradient detection and removal for astrophotography. |
| **PixInsight** | Professional astrophotography processing with some ML integration. |

---

## 3. Data Sources & APIs

### 3.1 Major Sky Surveys

| Survey | Access Method | Data Volume | URL |
|--------|--------------|-------------|-----|
| **SDSS (Sloan Digital Sky Survey)** | CAS (SQL queries), SAS (FITS files), astroquery.sdss, Bulk download via Globus | 35% of sky, DR17 latest | https://www.sdss4.org/dr17/data_access/ |
| **Gaia (ESA)** | TAP+ service, astroquery.gaia, VizieR | 1.8B sources, DR3 | https://gea.esac.esa.int/archive/ |
| **HST (Hubble)** | MAST API, z.MAST API, ESA HST Archive | Decades of observations | https://archive.stsci.edu/ |
| **ZTF (Zwicky Transient Facility)** | IRSA, public data releases (DR1-DR4+) | Northern sky, time-domain | https://www.ztf.caltech.edu/ztf-public-releases.html |
| **DECaLS (Dark Energy Camera Legacy Survey)** | Legacy Survey viewer, direct download | Deep imaging, southern sky | https://www.legacysurvey.org/ |
| **Hyper Suprime-Cam (HSC)** | HSC-SSP public data releases | Deep wide-field imaging | https://hsc.mtk.nao.ac.jp/ssp/ |
| **2MASS** | IRSA, astroquery, VizieR | Full sky near-IR | https://irsa.ipac.caltech.edu/Missions/2mass.html |
| **WISE/NEOWISE** | IRSA | Full sky mid-IR | https://irsa.ipac.caltech.edu/Missions/wise.html |
| **Pan-STARRS** | MAST, PS1 archive | 3/4 sky optical | https://panstarrs.stsci.edu/ |

### 3.2 Upcoming / Active Surveys

| Survey | Status | Key Feature |
|--------|--------|-------------|
| **Vera Rubin Observatory / LSST** | First light achieved; data pipeline active | ~10M alerts/night, 60-second processing. Brokers: ALeRCE, Lasair, Pitt-Google, SNAPS |
| **Euclid (ESA)** | Operational | Galaxy morphology + weak lensing, integrated with Galaxy Zoo |
| **Roman Space Telescope** | In development | Wide-field IR, microlensing surveys |

### 3.3 Programmatic Access Methods

```python
# SDSS images via astroquery
from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord
pos = SkyCoord('0h8m05.63s +14d50m23.3s', frame='icrs')
images = SDSS.get_images(pos, band='g')

# Gaia catalog query
from astroquery.gaia import Gaia
job = Gaia.launch_job("SELECT TOP 100 * FROM gaiadr3.gaia_source WHERE ...")
results = job.get_results()

# MAST / Hubble archive
from astroquery.mast import Observations
obs = Observations.query_criteria(obs_collection='HST', target_name='M31')

# SkyView all-sky image cutouts
from astroquery.skyview import SkyView
images = SkyView.get_images(position='M31', survey='DSS')

# VizieR catalog access (20,000+ catalogs)
from astroquery.vizier import Vizier
result = Vizier.query_region(pos, radius='0d6m0s', catalog='II/246')
```

### 3.4 Image Formats

| Format | Description | Python Library |
|--------|-------------|----------------|
| **FITS** (Flexible Image Transport System) | Standard astronomical image format. Supports multi-HDU, WCS metadata, tables. | `astropy.io.fits` |
| **HDF5** | Used by some surveys for large datasets | `h5py` |
| **JPEG/PNG** | Compressed preview images (lossy, not for science) | `PIL/Pillow` |

### 3.5 Key Archive Endpoints

- **MAST API**: https://mast.stsci.edu/api/v0/ (HST, JWST, TESS, Kepler, etc.)
- **ESA HST Archive**: https://hst.esac.esa.int/ehst/
- **Hubble on AWS Open Data**: https://registry.opendata.aws/hst/
- **IRSA**: https://irsa.ipac.caltech.edu/ (ZTF, 2MASS, WISE, Spitzer)
- **CDS VizieR**: https://vizier.cds.unistra.fr/
- **NASA ADS**: https://ui.adsabs.harvard.edu/ (Paper search)

---

## 4. Pattern Types in Star Fields

### 4.1 Known Structural Patterns

| Pattern Type | Description | Detection Difficulty |
|-------------|-------------|---------------------|
| **Star Clusters (Open)** | Gravitationally bound groups of 100-10,000 stars sharing common origin. Described by King models (core radius + tidal radius). | Easy |
| **Star Clusters (Globular)** | Dense, ancient spherical clusters of 100K-1M stars. Anomalous peak in radial distribution at ~40,000 ly from galactic center. | Easy |
| **Stellar Streams** | Tidal debris from disrupted dwarf galaxies/clusters. Extremely low surface brightness (<29 mag/arcsec^2). | Hard |
| **Gravitational Lenses (Strong)** | Einstein rings, arcs, multiple images of background objects. Compound arcs, double rings. | Medium |
| **Gravitational Lenses (Weak)** | Statistical shape distortions in background galaxies. | Hard |
| **Cosmic Web Filaments** | Elongated structures connecting galaxy clusters. Detected via DisPerSE, SpineWeb, MMF algorithms. | Hard |
| **Voids** | Near-empty regions bounded by filaments and walls. | Medium |
| **Galaxy Mergers** | Tidal tails, bridges, distorted morphologies. | Medium |
| **Planetary Nebulae** | Ring/bipolar structures from dying stars. | Easy |
| **Supernova Remnants** | Expanding shell structures. | Medium |

### 4.2 Mathematical Relationships in Stellar Distributions

- **King Model**: Describes radial density profile of star clusters using core radius (r_c) and tidal radius (r_t)
- **Sersic Profile**: I(r) = I_e * exp(-b_n * [(r/r_e)^(1/n) - 1]) describes galaxy light profiles
- **Power-law mass function**: Cluster initial mass function follows two-part power law
- **Two-point correlation function**: Statistical measure of galaxy/star clustering
- **Voronoi tessellation**: Used to identify over/under-dense regions
- **Minkowski functionals**: Topological descriptors of point distributions

### 4.3 Scientifically Interesting Anomalies

| Anomaly Type | Why Interesting |
|-------------|----------------|
| **Unusual galaxy morphologies** | May indicate rare evolutionary stages or unknown physical processes |
| **Strong lens candidates** | Probe dark matter distribution, measure Hubble constant |
| **Dark matter substructure in lensing** | Tests cold dark matter predictions |
| **Stellar streams disruptions/gaps** | Evidence of dark matter subhalo interactions |
| **Hypervelocity stars** | Ejected by supermassive black holes |
| **Unusual transients** | New classes of explosive/eruptive phenomena |
| **Void galaxies** | Galaxies in cosmic voids evolve differently |
| **Stellar collision transients** | Rare merger events between stars |
| **Red dwarf flares** | Habitability implications for exoplanets |
| **Unidentified extended structures** | Could reveal new physical phenomena |

---

## 5. Algorithms & Approaches

### 5.1 Supervised Learning

| Method | Application | Key Result |
|--------|-------------|------------|
| **ResNet / ResNet-152V2** | Galaxy morphology, star/galaxy/quasar classification | 92-95% accuracy |
| **Deformable CNNs** | Galaxy morphology with GZ DECaLS | 94.5% accuracy |
| **YOLOv8/v9/v10** | Source detection, deep sky object detection, star/constellation recognition | mAP50 up to 95.7% |
| **U-Net** | Satellite streak removal from astronomical images | Outperforms traditional methods |
| **Deformable DETR, Faster RCNN** | Star localization (compared against SExtractor) | Competitive with classical methods |
| **RAPID (RNNs)** | Real-time photometric transient classification | Updates predictions as new data arrives |

### 5.2 Self-Supervised Learning (SSL)

| Method | Application | Key Advantage |
|--------|-------------|---------------|
| **BYOL (Bootstrap Your Own Latent)** | Galaxy representation learning from Galaxy Zoo DECaLS | No labels needed; learned features useful for both morphology grouping and anomaly detection |
| **DINOv2 / AstroDINO** | Vision transformer backbone for galaxy image embeddings | Learns general visual representations transferable to many downstream tasks |
| **AstroCLIP** | Cross-modal embeddings (image + spectra) | Shared latent space enables semantic search, redshift estimation, morphology classification without fine-tuning |
| **Context Prediction SSL** | Radio astronomy spectrogram representation (ROAD) | Outperforms autoencoders for anomaly detection in LOFAR data |

### 5.3 Unsupervised Anomaly Detection

| Method | Application | Tool |
|--------|-------------|------|
| **Isolation Forest** | Outlier detection in feature space | SNAD coniferest, scikit-learn |
| **Active Anomaly Discovery (AAD)** | Interactive anomaly detection with human-in-the-loop | SNAD, Astronomaly |
| **PineForest** | Enhanced isolation forest variant | SNAD coniferest |
| **Autoencoders (reconstruction error)** | Anomalous galaxy detection | Custom implementations |
| **WGAN (reconstruction + discriminator)** | Galaxy anomaly detection in HSC images | Custom |
| **One-class SVM** | Novelty detection | scikit-learn |
| **Local Outlier Factor** | Density-based anomaly detection | scikit-learn |

### 5.4 GAN-Based Approaches

- **WGAN for anomaly detection**: Train generator on normal galaxy distribution. Anomalies identified by (a) poor reconstruction quality and (b) outlying discriminator features.
- **Diffusion models replacing GANs**: 2024-2025 trend shows diffusion models are superior for image reconstruction, denoising, and artifact removal in astronomy.
- **Conditional DDPM**: Applied to radio image deconvolution and galaxy super-resolution.

### 5.5 Transfer Learning Strategies

- **ImageNet pretraining + fine-tuning**: Works but astronomical-domain pretraining (e.g., ZooBot) is significantly better.
- **Cross-survey transfer**: ZTF -> LSST models maintain 94% performance with 30% training data.
- **Few-shot / zero-shot**: LLMs (Gemini) achieve 93% accuracy with only 15 examples for transient classification.
- **Foundation model approach**: Pretrain on unlabeled astronomical data, fine-tune for specific tasks. Reduces labeled data needs by 70-95%.

### 5.6 Evolutionary / Genetic Algorithms

- Applied to **gravitational microlensing light curve fitting** (parameter optimization)
- Used for **binary asteroid orbital parameter estimation** (N-body problem fitting)
- **Stellar structure modeling** via parallel genetic algorithms
- **Pulsar and exoplanet detection** in time series data
- Best suited for **optimization problems** rather than pattern recognition per se

### 5.7 Cosmic Web Detection Algorithms

| Algorithm | Method | URL/Reference |
|-----------|--------|---------------|
| **DisPerSE** | Discrete Morse theory + persistence for scale-free filament/void/cluster detection | Built on topological data analysis |
| **SpineWeb** | Watershed segmentation of density field | |
| **Multi-scale Morphology Filter (MMF)** | Scale-space technology for node/filament/wall/void classification | |
| **ASTRA** | Stochastic topological ranking for galaxy classification into web structures | 2025 paper |

---

## 6. Hypothesis Refinement

### 6.1 Most Promising Pattern Types to Search For

Based on the literature, the following are ranked by scientific value x feasibility:

**Tier 1: High Impact, Achievable with Current Tools**
1. **Strong gravitational lens candidates** - Astronomaly found 8 candidates in 4M galaxies. CNNs achieve 95%+ TPR. Large sky coverage remains unsearched.
2. **Galaxy merger candidates** - Same Astronomaly run found 1,609 candidates. Tidal features detectable with standard CNNs.
3. **Unusual galaxy morphologies** - ZooBot + anomaly detection pipeline can flag novel morphologies from existing survey data.

**Tier 2: High Impact, Requires More Sophisticated Pipeline**
4. **Stellar streams in resolved star fields** - CNN detection is emerging (2025 paper) but requires deep imaging and careful background subtraction.
5. **Dark matter substructure via lensing perturbations** - Active research area; ML methods being developed (2024 paper).
6. **Novel transient classification** - LLM-based approaches (2025 Oxford/Google) open new possibilities with minimal training data.

**Tier 3: Exploratory / Discovery-Oriented**
7. **Unknown extended structures** - Unsupervised anomaly detection on large image datasets. No prior hypothesis needed.
8. **Statistical anomalies in stellar distributions** - Voronoi tessellation + density analysis to find unexplained over/under-densities.
9. **Cross-wavelength anomalies** - Objects that look normal in one band but anomalous in another.

### 6.2 Formulating Testable Sub-Hypotheses

| Hypothesis | Test Method | Success Criteria |
|-----------|-------------|-----------------|
| H1: "Pretrained foundation models can identify novel patterns in star fields without task-specific training" | Apply AstroCLIP/ZooBot embeddings + Isolation Forest to survey images | Discovery of objects flagged as anomalous that are confirmed by spectroscopic follow-up or cross-referencing catalogs |
| H2: "Unsupervised clustering of astronomical image embeddings reveals physically meaningful groupings" | BYOL/DINOv2 embeddings + t-SNE/UMAP + HDBSCAN on 100K+ images | Clusters correspond to known morphological types AND contain at least one previously uncatalogued group |
| H3: "Active anomaly detection can find rare objects more efficiently than random search" | Compare Astronomaly pipeline vs random sampling on identical dataset | Anomaly pipeline finds 2x+ more interesting objects in first N samples (as validated by expert review) |
| H4: "Transfer learning from natural images underperforms domain-specific pretraining for astronomical anomaly detection" | Compare ImageNet-pretrained vs ZooBot-pretrained backbone for anomaly detection | Domain-specific model achieves higher AUC-ROC for known anomaly types |
| H5: "Star field density patterns contain structure beyond Poisson noise that correlates with known physical phenomena" | Voronoi tessellation + persistence homology on star catalogs from Gaia DR3 | Detected structures match known stellar streams, moving groups, or OB associations |

### 6.3 What Constitutes Proof / Disproof

**For pattern detection validity:**
- **Proof**: Pattern is statistically significant (p < 0.001 above noise), reproducible across independent observations/surveys, and has a plausible physical explanation OR represents a genuinely novel phenomenon confirmed by independent observers.
- **Disproof**: Pattern disappears when tested on different survey data, is attributable to instrumental artifacts, selection effects, or statistical noise.

**For ML model utility:**
- **Proof**: Model identifies known objects with high precision/recall AND finds previously uncatalogued objects that are confirmed through follow-up.
- **Disproof**: Model's anomaly scores are uncorrelated with scientific interest as judged by domain experts.

**Controls needed:**
- Inject synthetic anomalies into test data to measure detection rate
- Compare against random baseline and classical methods
- Test on multiple surveys to rule out survey-specific artifacts
- Perform statistical significance tests with proper multiple-comparison correction

---

## 7. Actionable Next Steps

### 7.1 Recommended Initial Pipeline

```
Step 1: Data Acquisition
    - Use astroquery to download SDSS/DECaLS cutout images (start with 10K-100K)
    - Download corresponding Gaia DR3 star catalogs for same regions
    - Store as FITS with WCS metadata

Step 2: Feature Extraction
    - Option A: Use ZooBot pretrained model as feature extractor (frozen backbone)
    - Option B: Use AstroCLIP for multi-modal embeddings
    - Option C: Classical features via Photutils/SEP (flux, size, ellipticity, etc.)

Step 3: Anomaly Detection
    - Apply Isolation Forest to feature embeddings
    - Implement Astronomaly active learning loop
    - Rank objects by anomaly score

Step 4: Validation
    - Cross-reference top anomalies against SIMBAD, NED, VizieR catalogs
    - Visual inspection of top candidates
    - Flag genuinely novel objects for follow-up

Step 5: Iteration
    - Incorporate expert feedback via active learning
    - Expand to larger dataset
    - Test on different survey data
```

### 7.2 Quick-Start Tool Installation

```bash
pip install astropy photutils astroquery sep
pip install scikit-learn torch torchvision
pip install zoobot  # pretrained galaxy morphology models
pip install umap-learn hdbscan  # dimensionality reduction + clustering

# Clone Astronomaly
git clone https://github.com/MichelleLochner/astronomaly.git

# Clone SNAD anomaly detection
git clone https://github.com/snad-space/zwad.git
```

### 7.3 Key GitHub Repositories

| Repository | Purpose |
|-----------|---------|
| https://github.com/MichelleLochner/astronomaly | Anomaly detection framework |
| https://github.com/mwalmsley/zoobot | Pretrained galaxy morphology CNN |
| https://github.com/PolymathicAI/AstroCLIP | Cross-modal foundation model |
| https://github.com/snad-space/zwad | ZTF anomaly detection pipeline |
| https://github.com/astromatic/sextractor | Source extraction |
| https://github.com/dstndstn/astrometry.net | Astrometric calibration |
| https://github.com/astropy/photutils | Source detection and photometry |
| https://github.com/kbarbary/sep | SExtractor as Python library |
| https://github.com/jobovy/gaia_tools | Gaia data access tools |

### 7.4 Recommended Reading Order

1. Start with the survey: [Machine Learning in Stellar Astronomy: Progress up to 2024](https://arxiv.org/html/2502.15300v1)
2. Understand anomaly detection: [Astronomaly at Scale](https://academic.oup.com/mnras/article/529/1/732/7612998)
3. Learn foundation models: [AstroCLIP paper](https://academic.oup.com/mnras/article/531/4/4990/7697182)
4. Study self-supervised approach: [Enabling Unsupervised Discovery](https://academic.oup.com/mnras/article/530/1/1274/7640046)
5. Review practical tools: [ZooBot documentation](https://zoobot.readthedocs.io/)
6. Explore cosmic web: [DisPerSE methodology](https://academic.oup.com/mnras/article/414/1/350/1090746)

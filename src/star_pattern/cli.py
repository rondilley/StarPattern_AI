"""CLI interface for star_pattern_AI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table as RichTable

from star_pattern.utils.logging import get_logger

console = Console()
logger = get_logger("cli")


@click.group()
@click.option("--config", "-c", default="config.json", help="Path to config file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx: click.Context, config: str, verbose: bool) -> None:
    """Star Pattern AI - Autonomous discovery of patterns in star fields."""
    from star_pattern.core.config import PipelineConfig
    from star_pattern.utils.logging import setup_logging

    setup_logging(level="DEBUG" if verbose else "INFO")

    ctx.ensure_object(dict)
    config_path = Path(config)
    if config_path.exists():
        ctx.obj["config"] = PipelineConfig.from_file(config_path)
    else:
        ctx.obj["config"] = PipelineConfig()


@cli.command()
@click.option("--ra", type=float, required=False, help="Right ascension (degrees)")
@click.option("--dec", type=float, required=False, help="Declination (degrees)")
@click.option("--radius", type=float, default=3.0, help="Search radius (arcmin)")
@click.option("--sources", type=str, default="sdss", help="Comma-separated data sources")
@click.option("--bands", type=str, default="r", help="Comma-separated bands")
@click.option("--random", "n_random", type=int, default=0, help="Fetch N random regions")
@click.option("--min-gal-lat", type=float, default=20.0, help="Min galactic latitude")
@click.pass_context
def fetch(
    ctx: click.Context,
    ra: float | None,
    dec: float | None,
    radius: float,
    sources: str,
    bands: str,
    n_random: int,
    min_gal_lat: float,
) -> None:
    """Fetch astronomical data for a sky region."""
    from star_pattern.core.sky_region import SkyRegion
    from star_pattern.data.sdss import SDSSDataSource
    from star_pattern.data.cache import DataCache

    config = ctx.obj["config"]
    cache = DataCache(config.data.cache_dir)
    band_list = [b.strip() for b in bands.split(",")]

    # Build source objects
    source_map = {
        "sdss": lambda: SDSSDataSource(cache=cache),
    }

    source_list = [s.strip() for s in sources.split(",")]
    active_sources = []
    for s in source_list:
        if s in source_map:
            active_sources.append(source_map[s]())
        else:
            console.print(f"[yellow]Unknown source: {s}[/yellow]")

    if not active_sources:
        console.print("[red]No valid data sources specified[/red]")
        return

    # Build regions
    regions: list[SkyRegion] = []
    if n_random > 0:
        import numpy as np

        rng = np.random.default_rng()
        for _ in range(n_random):
            regions.append(SkyRegion.random(min_galactic_lat=min_gal_lat, radius=radius, rng=rng))
        console.print(f"Generated {n_random} random regions (|b| > {min_gal_lat}°)")
    elif ra is not None and dec is not None:
        regions.append(SkyRegion(ra=ra, dec=dec, radius=radius))
    else:
        console.print("[red]Specify --ra/--dec or --random N[/red]")
        return

    # Fetch
    for region in regions:
        console.print(f"\n[bold]Fetching ({region.ra:.4f}, {region.dec:.4f}) r={region.radius}'[/bold]")
        for source in active_sources:
            data = source.fetch_region(region, bands=band_list)
            if data.has_images():
                for band, img in data.images.items():
                    console.print(f"  [green]{source.name}/{band}[/green]: {img.shape}")
            if data.has_catalogs():
                for name, cat in data.catalogs.items():
                    console.print(f"  [blue]{name} catalog[/blue]: {len(cat)} entries")

    console.print("\n[bold green]Fetch complete.[/bold green]")


@cli.command(name="fetch-wide")
@click.option("--ra", type=float, required=True, help="Center RA (degrees)")
@click.option("--dec", type=float, required=True, help="Center Dec (degrees)")
@click.option("--field-radius", type=float, required=True, help="Field radius (arcmin)")
@click.option("--tile-radius", type=float, default=3.0, help="Per-tile radius (arcmin)")
@click.option("--overlap", type=float, default=0.2, help="Tile overlap fraction")
@click.option("--output", "-o", required=True, help="Output directory")
@click.option("--bands", type=str, default="r", help="Comma-separated bands")
@click.pass_context
def fetch_wide(
    ctx: click.Context,
    ra: float,
    dec: float,
    field_radius: float,
    tile_radius: float,
    overlap: float,
    output: str,
    bands: str,
) -> None:
    """Fetch and mosaic a wide-field sky region."""
    from star_pattern.core.config import WideFieldConfig
    from star_pattern.data.wide_field import WideFieldPipeline

    config = ctx.obj["config"]
    config.wide_field = WideFieldConfig(
        tile_radius_arcmin=tile_radius,
        overlap_fraction=overlap,
    )

    band_list = [b.strip() for b in bands.split(",")]

    console.print(
        f"[bold]Wide-field fetch: "
        f"({ra:.4f}, {dec:.4f}) r={field_radius}'[/bold]"
    )
    console.print(
        f"  Tile radius: {tile_radius}', overlap: {overlap}"
    )

    pipeline = WideFieldPipeline(config)
    result = pipeline.fetch_wide_field(
        center_ra=ra,
        center_dec=dec,
        field_radius_arcmin=field_radius,
        bands=band_list,
    )

    out_path = Path(output)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save mosaicked FITS images
    for band, img in result.images.items():
        fits_path = out_path / f"mosaic_{band}.fits"
        img.save(fits_path)
        console.print(f"  [green]{band}[/green]: {img.shape} -> {fits_path}")

        # Save PNG overlay
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            norm_img = img.normalize("zscale")
            ax.imshow(norm_img.data, cmap="gray", origin="lower")
            ax.set_title(f"Wide-field mosaic: {band}")
            png_path = out_path / f"mosaic_{band}.png"
            fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            console.print(f"  Overlay: {png_path}")
        except Exception as e:
            logger.debug(f"PNG save failed: {e}")

    # Save catalog summary
    for src, cat in result.catalogs.items():
        console.print(f"  [blue]{src} catalog[/blue]: {len(cat)} entries")

    console.print(
        f"\n[bold green]Wide-field fetch complete.[/bold green] "
        f"{result.metadata.get('n_tiles', '?')} tiles mosaicked."
    )


@cli.command()
@click.option("--input", "-i", "input_path", required=True, help="FITS file or directory")
@click.option("--type", "-t", "det_type", default="all", help="Detection type (all, classical, anomaly, lens)")
@click.option("--batch", is_flag=True, help="Process all FITS in directory")
@click.option("--output", "-o", default=None, help="Output directory")
@click.pass_context
def detect(ctx: click.Context, input_path: str, det_type: str, batch: bool, output: str | None) -> None:
    """Run pattern detection on FITS images."""
    from star_pattern.core.fits_handler import FITSImage
    from star_pattern.detection.ensemble import EnsembleDetector

    config = ctx.obj["config"]
    path = Path(input_path)

    if batch and path.is_dir():
        files = sorted(path.glob("*.fits")) + sorted(path.glob("*.fits.gz"))
    elif path.is_file():
        files = [path]
    else:
        console.print(f"[red]Not found: {input_path}[/red]")
        return

    console.print(f"Processing {len(files)} file(s) with detection type: {det_type}")

    detector = EnsembleDetector(config.detection)
    results = []

    for f in files:
        console.print(f"\n[bold]{f.name}[/bold]")
        try:
            img = FITSImage.from_file(f)
            result = detector.detect(img)
            results.append(result)
            console.print(f"  Anomaly score: {result.get('anomaly_score', 0):.4f}")
            console.print(f"  Detections: {result.get('n_detections', 0)}")
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")

    if output:
        out_path = Path(output)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "detections.json", "w") as fout:
            json.dump(results, fout, indent=2, default=str)
        console.print(f"\nResults saved to {out_path / 'detections.json'}")

        # Save overlay images for each detection
        from star_pattern.visualization.pattern_overlay import overlay_sources
        import matplotlib.pyplot as plt

        for i, (fpath, result) in enumerate(zip(files, results)):
            try:
                img = FITSImage.from_file(fpath)
                sources = result.get("sources", {})
                if sources:
                    fig = overlay_sources(img, sources)
                    fig.savefig(
                        str(out_path / f"{fpath.stem}_overlay.png"),
                        dpi=150, bbox_inches="tight",
                    )
                    plt.close(fig)
                    console.print(f"  Overlay: {fpath.stem}_overlay.png")
            except Exception as e:
                logger.debug(f"Overlay save failed: {e}")


@cli.command()
@click.option("--generations", "-g", type=int, default=50, help="Number of generations")
@click.option("--population", "-p", type=int, default=30, help="Population size")
@click.option("--resume", type=str, default=None, help="Resume from run directory")
@click.pass_context
def evolve(ctx: click.Context, generations: int, population: int, resume: str | None) -> None:
    """Run evolutionary parameter optimization."""
    from star_pattern.discovery.evolutionary import EvolutionaryDiscovery
    from star_pattern.utils.run_manager import RunManager

    config = ctx.obj["config"]
    config.evolution.generations = generations
    config.evolution.population_size = population

    if resume:
        run_mgr = RunManager(run_name=resume)
    else:
        run_mgr = RunManager(base_dir=config.output_dir)

    console.print(f"[bold]Evolutionary search: {generations} gen x {population} pop[/bold]")

    engine = EvolutionaryDiscovery(config, run_manager=run_mgr)
    best = engine.run()

    console.print(f"\n[bold green]Best fitness: {best.fitness:.4f}[/bold green]")
    console.print(f"Run saved to: {run_mgr.run_dir}")


@cli.command()
@click.option("--hours", type=float, default=None, help="Max runtime in hours")
@click.option("--cycles", type=int, default=None, help="Max discovery cycles")
@click.option("--with-llm", is_flag=True, help="Enable LLM hypothesis generation")
@click.option("--wide-field", type=float, default=None, help="Wide-field radius (arcmin)")
@click.option("--survey", is_flag=True, help="Enable HEALPix grid survey mode")
@click.option("--nside", type=int, default=64, help="HEALPix NSIDE resolution (default 64)")
@click.option(
    "--survey-order",
    type=click.Choice(["galactic_latitude", "random_shuffle", "dec_sweep"]),
    default="galactic_latitude",
    help="Survey visit order",
)
@click.option("--with-ztf/--no-ztf", default=True, help="Include ZTF light curves")
@click.option("--slaves", type=str, default=None,
              help="Comma-separated slave addresses (host:port,...) for distributed mode")
@click.option("--auth-token", "discover_auth_token", default="",
              help="Shared auth token for distributed communication")
@click.pass_context
def discover(
    ctx: click.Context,
    hours: float | None,
    cycles: int | None,
    with_llm: bool,
    wide_field: float | None,
    survey: bool,
    nside: int,
    survey_order: str,
    with_ztf: bool,
    slaves: str | None,
    discover_auth_token: str,
) -> None:
    """Run autonomous discovery pipeline."""
    from star_pattern.pipeline.autonomous import AutonomousDiscovery
    from star_pattern.utils.run_manager import RunManager

    config = ctx.obj["config"]
    if cycles:
        config.max_cycles = cycles

    if not with_ztf and "ztf" in config.data.sources:
        config.data.sources = [s for s in config.data.sources if s != "ztf"]

    # Configure distributed mode if slaves specified
    if slaves:
        from star_pattern.distributed.config import DistributedConfig
        addresses = [a.strip() for a in slaves.split(",") if a.strip()]
        config.distributed = DistributedConfig(
            mode="master",
            slave_addresses=addresses,
            auth_token=discover_auth_token,
        )
        console.print(f"  Distributed mode: {len(addresses)} slave(s) configured")

    run_mgr = RunManager(base_dir=config.output_dir)
    console.print("[bold]Starting autonomous discovery...[/bold]")

    pipeline = AutonomousDiscovery(config, run_manager=run_mgr, use_llm=with_llm)
    if wide_field is not None:
        pipeline.set_wide_field(wide_field)
        console.print(f"  Wide-field mode: {wide_field}' radius")
    if survey:
        from star_pattern.core.config import SurveyConfig
        survey_config = SurveyConfig(nside=nside, order=survey_order)
        pipeline.set_survey(survey_config)
        console.print(
            f"  Survey mode: NSIDE={nside}, order={survey_order}"
        )
    findings = pipeline.run(max_hours=hours)

    # Summary table
    if findings:
        table = RichTable(title="Top Findings")
        table.add_column("Type")
        table.add_column("RA")
        table.add_column("Dec")
        table.add_column("Score")
        for f in sorted(findings, key=lambda x: x.anomaly_score, reverse=True)[:10]:
            table.add_row(
                f.detection_type,
                f"{f.region_ra:.4f}",
                f"{f.region_dec:.4f}",
                f"{f.anomaly_score:.4f}",
            )
        console.print(table)

    console.print(f"\n[bold green]Discovery complete.[/bold green]")
    console.print(f"Run saved to: {run_mgr.run_dir}")
    console.print(f"Images: {run_mgr.images_dir}")
    console.print(f"Reports: {run_mgr.reports_dir}")


@cli.command()
@click.option("--input", "-i", "input_path", required=True, help="Pattern result JSON")
@click.option("--with-debate", is_flag=True, help="Run adversarial debate")
@click.pass_context
def analyze(ctx: click.Context, input_path: str, with_debate: bool) -> None:
    """Analyze a detection result with LLMs."""
    from star_pattern.llm.hypothesis import HypothesisGenerator
    from star_pattern.llm.providers.discovery import ProviderDiscovery

    config = ctx.obj["config"]

    with open(input_path) as f:
        pattern_data = json.load(f)

    discovery = ProviderDiscovery(key_dir=config.llm.key_dir)
    providers = discovery.discover()

    if not providers:
        console.print("[red]No LLM providers found. Check *.key.txt files.[/red]")
        return

    console.print(f"Using {len(providers)} LLM provider(s)")

    gen = HypothesisGenerator(providers[0], config.llm)
    hypothesis = gen.generate(pattern_data)
    console.print(f"\n[bold]Hypothesis:[/bold]\n{hypothesis}")

    if with_debate and len(providers) >= 2:
        from star_pattern.llm.debate import PatternDebate

        debate = PatternDebate(providers, config.llm)
        verdict = debate.run(pattern_data)
        console.print(f"\n[bold]Debate Verdict:[/bold]\n{verdict}")


@cli.command()
@click.option("--task", type=click.Choice(["lens", "morphology", "anomaly"]), required=True)
@click.option("--data", "data_dir", required=True, help="Training data directory")
@click.option("--epochs", type=int, default=100, help="Training epochs")
@click.option("--batch-size", type=int, default=16, help="Batch size")
@click.pass_context
def train(ctx: click.Context, task: str, data_dir: str, epochs: int, batch_size: int) -> None:
    """Train a detection model."""
    from star_pattern.ml.train import Trainer

    config = ctx.obj["config"]
    console.print(f"[bold]Training {task} model for {epochs} epochs[/bold]")

    trainer = Trainer(task=task, data_dir=data_dir, epochs=epochs, batch_size=batch_size)
    trainer.run()


@cli.command(name="survey-status")
@click.option("--state-file", required=True, help="Path to survey_state.json")
def survey_status(state_file: str) -> None:
    """Show HEALPix survey progress."""
    state_path = Path(state_file)
    if not state_path.exists():
        console.print(f"[red]State file not found: {state_file}[/red]")
        return

    with open(state_path) as f:
        state = json.load(f)

    config_data = state.get("config", {})
    visited = state.get("visited", [])
    pending = state.get("pending", [])
    findings = state.get("findings_per_pixel", {})

    n_visited = len(visited)
    n_pending = len(pending)
    n_total = n_visited + n_pending
    pct = (n_visited / n_total * 100.0) if n_total > 0 else 0.0
    n_with_findings = sum(1 for v in findings.values() if v > 0)
    total_findings = sum(findings.values())

    table = RichTable(title="HEALPix Survey Status")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("NSIDE", str(config_data.get("nside", "?")))
    table.add_row("Order", config_data.get("order", "?"))
    table.add_row("Min galactic lat", f"{config_data.get('min_galactic_lat', '?')} deg")
    table.add_row("Total pixels", str(n_total))
    table.add_row("Visited", str(n_visited))
    table.add_row("Remaining", str(n_pending))
    table.add_row("Progress", f"{pct:.1f}%")
    table.add_row("Pixels with findings", str(n_with_findings))
    table.add_row("Total findings", str(total_findings))

    console.print(table)


@cli.command(name="setup-local")
def setup_local() -> None:
    """Set up local LLM backend."""
    console.print("[bold]Setting up local LLM...[/bold]")
    try:
        from star_pattern.llm.providers.llamacpp_provider import LlamaCppProvider

        provider = LlamaCppProvider.setup_default()
        console.print(f"[green]Local LLM ready: {provider.model_name}[/green]")
    except Exception as e:
        console.print(f"[red]Setup failed: {e}[/red]")
        console.print("Install: pip install star-pattern-ai[local]")


@cli.command()
@click.option("--host", default="0.0.0.0", help="Listen address")
@click.option("--port", type=int, default=7827, help="Listen port")
@click.option("--auth-token", default="", help="Shared auth token")
@click.pass_context
def serve(ctx: click.Context, host: str, port: int, auth_token: str) -> None:
    """Start as a slave worker node, listening for work from a master."""
    import asyncio
    from star_pattern.distributed.config import DistributedConfig
    from star_pattern.distributed.slave import SlaveServer

    config = ctx.obj["config"]
    config.distributed = DistributedConfig(
        mode="slave",
        listen_host=host,
        listen_port=port,
        auth_token=auth_token,
    )

    console.print(
        f"[bold]Starting slave worker on {host}:{port}...[/bold]"
    )
    server = SlaveServer(config)
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        console.print("\n[bold]Slave worker stopped.[/bold]")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()

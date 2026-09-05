#!/usr/bin/env python3
"""Headless command-line interface for UVR separation.

Reuses the unmodified GUI separation engines (UVR.py/separate.py) via the
api/ package's HeadlessRoot shim -- no Tkinter display required. Suitable
for scripting, batch jobs, or servers.

Examples:
    python cli.py --list-models
    python cli.py --download-model mdx "UVR-MDX-NET Inst HQ 3"
    python cli.py -i song.wav -o ./out --arch mdx --model UVR-MDX-NET-Inst_HQ_3
    python cli.py -i song.wav -o ./out --arch vr --model 1_HP-UVR --cpu
"""
from __future__ import annotations

import argparse
import sys
import time


class _StdoutJob:
    """Minimal stand-in for api.jobs.Job -- the only interface run_separation
    needs is .id, .update(**kwargs), and .append_log(text)."""

    id = "cli"

    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self._last_progress = -1.0

    def update(self, **kwargs):
        if self.quiet:
            return
        if "progress" in kwargs:
            pct = int(kwargs["progress"] * 100)
            if pct != self._last_progress:
                self._last_progress = pct
                print(f"\r  progress: {pct}%", end="", flush=True)
        if "message" in kwargs:
            print(f"\n{kwargs['message']}")

    def append_log(self, text: str):
        if not self.quiet:
            print(text, end="")


def cmd_list_models(args):
    from api.registry import list_models

    for m in list_models():
        mark = "x" if m.installed else " "
        print(f"[{mark}] {m.arch.value:6s} {m.name}")


def cmd_download_model(args):
    from api.schemas import Arch
    from api.registry import download_model

    try:
        arch = Arch(args.arch)
    except ValueError:
        print(f"Unknown arch '{args.arch}'. Use one of: vr, mdx, demucs", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading {args.display_name} ({arch.value})...")
    files = download_model(arch, args.display_name)
    print("Done:")
    for f in files:
        print(f"  {f}")


def cmd_separate(args):
    from api.schemas import Arch, OutputFormat, SeparationOptions
    from api.separation import run_separation

    try:
        arch = Arch(args.arch)
    except ValueError:
        print(f"Unknown arch '{args.arch}'. Use one of: vr, mdx, demucs", file=sys.stderr)
        sys.exit(1)

    use_gpu = True if args.gpu else False if args.cpu else None

    opts = SeparationOptions(
        arch=arch,
        model_name=args.model,
        primary_stem_only=args.primary_only,
        secondary_stem_only=args.secondary_only,
        output_format=OutputFormat(args.format),
        normalization=args.normalize,
        denoise=args.denoise,
        use_gpu=use_gpu,
    )

    job = _StdoutJob(quiet=args.quiet)
    t0 = time.monotonic()
    try:
        outputs = run_separation(args.input, args.output, opts, job)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone in {time.monotonic() - t0:.1f}s. Output files:")
    for out in outputs:
        print(f"  [{out.stem}] {out.filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Headless UVR separation (no GUI required).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list-models", help="List available and installed models")
    p_list.set_defaults(func=cmd_list_models)

    p_dl = sub.add_parser("download-model", help="Download a model by its catalog display name")
    p_dl.add_argument("arch", choices=["vr", "mdx", "demucs"])
    p_dl.add_argument("display_name", help='e.g. "UVR-MDX-NET Inst HQ 3"')
    p_dl.set_defaults(func=cmd_download_model)

    p_sep = sub.add_parser("separate", help="Separate an audio file into stems")
    p_sep.add_argument("-i", "--input", required=True, help="Input audio file")
    p_sep.add_argument("-o", "--output", required=True, help="Output directory")
    p_sep.add_argument("--arch", required=True, choices=["vr", "mdx", "demucs"])
    p_sep.add_argument("--model", required=True, help="Model basename (see list-models)")
    p_sep.add_argument("--format", default="WAV", choices=["WAV", "FLAC", "MP3"])
    p_sep.add_argument("--gpu", action="store_true", help="Force GPU (error if unavailable)")
    p_sep.add_argument("--cpu", action="store_true", help="Force CPU")
    p_sep.add_argument("--normalize", action="store_true")
    p_sep.add_argument("--denoise", action="store_true")
    p_sep.add_argument("--primary-only", action="store_true", dest="primary_only")
    p_sep.add_argument("--secondary-only", action="store_true", dest="secondary_only")
    p_sep.add_argument("-q", "--quiet", action="store_true")
    p_sep.set_defaults(func=cmd_separate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

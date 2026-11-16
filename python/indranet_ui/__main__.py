"""python -m indranet_ui model.safetensors: explore a model in the browser."""
import argparse
import sys
import webbrowser


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m indranet_ui",
                                 description="IndraNet's interpretability UI: a local web page on the tfi engine.")
    ap.add_argument("models", nargs="+", metavar="model",
                    help="model files (.safetensors from indranet.export or indranet_hf) or folders of them; "
                         "the page lists them all and opens the first")
    ap.add_argument("--port", type=int, default=8765, help="port (default 8765; 0 picks a free one)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to listen on (default 127.0.0.1: this computer only; the page has no login)")
    ap.add_argument("--fp16", action="store_true", help="keep the weights in half precision (half the memory)")
    ap.add_argument("--device", choices=["cpu", "both", "gpu"], help="where the engine computes (default both)")
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    ap.add_argument("--verbose", action="store_true", help="log every request")
    args = ap.parse_args(argv)
    try:
        import tfi
    except ImportError as e:
        print(e, file=sys.stderr)
        return 1
    from .server import serve

    if args.device:
        tfi.set_device(args.device)

    def ready(url: str) -> None:
        print(f"indranet_ui: {url}  (Ctrl-C stops)", flush=True)
        if args.open:
            webbrowser.open(url)

    print(f"loading {args.models[0]} ...", flush=True)
    try:
        serve(args.models, host=args.host, port=args.port, fp16=args.fp16, quiet=not args.verbose, ready=ready)
    except (OSError, RuntimeError, ValueError) as e:
        print(f"indranet_ui: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

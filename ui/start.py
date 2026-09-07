from __future__ import annotations

import argparse

from ui.backend.app import app


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the GMM experiment dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

"""Run the server: ``python -m research_mcp [--transport stdio|http]``."""

from __future__ import annotations

import argparse
import logging
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Research Paper MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default=os.environ.get("RESEARCH_TRANSPORT", "stdio"))
    parser.add_argument("--host", default=os.environ.get("RESEARCH_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("RESEARCH_PORT", "8000")))
    parser.add_argument("--data-dir", default=None, help="Library location (env RESEARCH_DATA_DIR)")
    args = parser.parse_args()
    if args.data_dir:
        os.environ["RESEARCH_DATA_DIR"] = args.data_dir
    # stdout carries the MCP protocol in stdio mode, so logs go to stderr.
    logging.basicConfig(level=os.environ.get("RESEARCH_LOG_LEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s")

    if args.transport == "stdio":
        from .server import mcp

        mcp.run("stdio")
        return

    import uvicorn

    from .http_app import create_app

    if args.host not in ("127.0.0.1", "localhost") and not os.environ.get("RESEARCH_API_TOKEN"):
        logging.warning("Listening on %s without RESEARCH_API_TOKEN: anyone on the network can use the server.", args.host)
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

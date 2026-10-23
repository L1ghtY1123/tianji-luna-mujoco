"""Tianji Luna MuJoCo entry point."""
from .app import create_app as create_workbench


def create_app():
    return create_workbench(demo_mode=True)


def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8351)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)

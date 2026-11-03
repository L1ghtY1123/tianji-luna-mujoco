"""Run the existing HTTP contract with project-owned resource caching."""
import argparse
import logging
import os

from tianji_foundationpose.backend import FoundationPoseBackend
from tianji_foundationpose.registry import Registry

from .backend import CachedFoundationPoseBackend
from .tracking_service import make_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8025)
    parser.add_argument('--mesh-directory', default=os.getenv('LUNA_REGISTRATION_MESH_DIRECTORY'))
    args = parser.parse_args()
    registry = Registry(os.getenv('FOUNDATIONPOSE_REGISTRY', '/data/registry'))
    base = FoundationPoseBackend(os.getenv('FOUNDATIONPOSE_SOURCE', '/opt/foundationpose'), registry,
                                 search_profile=os.getenv('FOUNDATIONPOSE_SEARCH_PROFILE', 'default'))
    backend = CachedFoundationPoseBackend(base, mesh_directory=args.mesh_directory)
    logging.basicConfig(level=logging.INFO)
    server = make_server(args.host, args.port, backend, registry)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        backend.close()


if __name__ == '__main__':
    main()

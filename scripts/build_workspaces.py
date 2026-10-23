"""Generate fixed single-arm workspaces in the existing cuRobo GPU environment."""
import argparse
from pathlib import Path

from luna_mujoco.workspace_builder import build_workspace

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--arm', choices=['A', 'B'], required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--resolution', type=float, default=.01)
parser.add_argument('--batch-size', type=int, default=1024)
parser.add_argument('--num-seeds', type=int, default=16)
parser.add_argument('--max-batches', type=int, help='Benchmark only; incomplete output must not be deployed')
args = parser.parse_args()
build_workspace(args.root, args.arm, args.output, resolution=args.resolution,
                batch_size=args.batch_size, num_seeds=args.num_seeds, max_batches=args.max_batches)

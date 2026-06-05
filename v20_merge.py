"""
V20 Checkpoint Merger

Copies .npy checkpoint files from a remote/worker directory into 
the main colossus_checkpoints/ directory.

Usage:
  # After downloading worker checkpoints from another machine:
  python v20_merge.py --source worker_checkpoints

  # Or from a custom path:
  python v20_merge.py --source /path/to/downloaded/checkpoints

  # Dry run (see what would be merged without actually copying):
  python v20_merge.py --source worker_checkpoints --dry-run
"""

import argparse
import os
import shutil
import glob

parser = argparse.ArgumentParser(description='Merge V20 worker checkpoints')
parser.add_argument('--source',      required=True,          help='Source checkpoint directory')
parser.add_argument('--destination', default='colossus_checkpoints', help='Main checkpoint directory (default: colossus_checkpoints)')
parser.add_argument('--dry-run',     action='store_true',    help='Preview changes without copying')
args = parser.parse_args()

SRC  = args.source
DEST = args.destination
DRY  = args.dry_run

os.makedirs(DEST, exist_ok=True)

src_files = glob.glob(os.path.join(SRC, "*.npy"))
# Exclude feature cache — only merge model predictions
src_files = [f for f in src_files if "features_cached" not in f]

if not src_files:
    print(f"No .npy checkpoint files found in: {SRC}")
    exit(1)

copied, skipped, overwritten = 0, 0, 0

print(f"{'[DRY RUN] ' if DRY else ''}Merging {len(src_files)} checkpoint files")
print(f"  Source      : {SRC}/")
print(f"  Destination : {DEST}/")
print()

for src_path in sorted(src_files):
    fname     = os.path.basename(src_path)
    dest_path = os.path.join(DEST, fname)
    
    if os.path.exists(dest_path):
        print(f"  [EXISTS]  {fname} — skipping (already in destination)")
        skipped += 1
    else:
        if not DRY:
            shutil.copy2(src_path, dest_path)
        print(f"  [COPY]    {fname}")
        copied += 1

print(f"\n{'─'*50}")
print(f"  Copied   : {copied} files {'(dry run — not actually copied)' if DRY else ''}")
print(f"  Skipped  : {skipped} files (already existed)")
print(f"  Total    : {copied + skipped} files now in {DEST}/")
if not DRY:
    print(f"\nNow run: python v20_stacker.py")

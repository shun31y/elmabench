"""Locating and importing the user's own clone of upstream FlowMDM.

FlowMDM is not redistributed with this repository. Its modules use paths relative to the
repository root (./dataset, ./glove, ./t2m), so we work from inside the clone.
"""
import os
import subprocess
import sys

FLOWMDM_COMMIT = "e81c67f"


def enter_flowmdm(flowmdm_dir, check_commit=True):
    flowmdm_dir = os.path.abspath(flowmdm_dir)
    if not os.path.isfile(os.path.join(flowmdm_dir, "model", "FlowMDM.py")):
        raise FileNotFoundError(
            f"No FlowMDM checkout at {flowmdm_dir}. Run scripts/setup_flowmdm.sh first.")
    if check_commit:
        head = subprocess.run(["git", "-C", flowmdm_dir, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        if not head.startswith(FLOWMDM_COMMIT):
            raise RuntimeError(f"FlowMDM checkout is at {head[:7]}, expected {FLOWMDM_COMMIT}")
        dirty = subprocess.run(["git", "-C", flowmdm_dir, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        if dirty:
            raise RuntimeError(f"FlowMDM checkout has local modifications to tracked files:\n{dirty}")
    os.chdir(flowmdm_dir)
    if flowmdm_dir not in sys.path:
        sys.path.insert(0, flowmdm_dir)
    return flowmdm_dir

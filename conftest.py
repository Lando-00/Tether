# Ensure project root is in sys.path for test imports
import os
import sys

from dotenv import load_dotenv

project_root = os.path.dirname(os.path.abspath(__file__))
# Insert ``src/`` at sys.path[0] so this worktree's tether package takes
# precedence over any sibling worktree registered via the same conda env's
# editable .pth file (two worktrees sharing mlc-venv2 would otherwise race).
_src = os.path.join(project_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load .env file for tests
load_dotenv(os.path.join(project_root, ".env"))

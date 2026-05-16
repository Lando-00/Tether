# Ensure project root is in sys.path for test imports
import sys
import os
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.abspath(__file__))
# Insert src/ at position 0 so this worktree's tether package takes precedence
# over any other editable install of tether that may be registered via .pth
# (e.g. a sibling worktree on the same conda env). The project root itself
# is also kept on the path for test-local imports (conftest, etc.).
src_root = os.path.join(project_root, "src")
if src_root not in sys.path:
    sys.path.insert(0, src_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load .env file for tests
load_dotenv(os.path.join(project_root, ".env"))

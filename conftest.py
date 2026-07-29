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

# The packaged default.yml selects MLCProvider, whose module imports mlc_llm.
# Those are the Qualcomm CodeLinaro Adreno wheels, installed out-of-band and
# only present on the Snapdragon target. Without them, every test that builds
# the app or an Engine from default settings dies in ConfigError long before
# it reaches what it actually asserts (lifespan wiring, middleware, protocol
# endpoints, content-type negotiation) — none of which are MLC-specific.
#
# So off-device, point the default provider at DummyProvider. On the
# Snapdragon box the import succeeds and nothing is overridden, so the real
# provider is exercised exactly as before. setdefault means an explicit
# override from the environment still wins.
try:  # noqa: SIM105
    import mlc_llm  # noqa: F401
except ModuleNotFoundError:
    os.environ.setdefault(
        "TETHER__PROVIDERS__MODEL__IMPL",
        "tether.providers.dummy.provider.DummyProvider",
    )

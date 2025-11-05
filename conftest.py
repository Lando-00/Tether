# Ensure project root is in sys.path for test imports
import sys
import os
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load .env file for tests
load_dotenv(os.path.join(project_root, ".env"))

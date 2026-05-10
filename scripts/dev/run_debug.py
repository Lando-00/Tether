"""
Debug runner for the MLC service with enhanced logging.
"""
import os
import sys

if __name__ == "__main__":
    # Set debug flag for the environment
    os.environ["MLC_DEBUG"] = "1"
    
    # Import and run the app
    # __file__ is now scripts/dev/run_debug.py — go up two levels to repo root
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    
    from tether.app.__main__ import main
    main()
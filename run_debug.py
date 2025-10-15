"""
Debug runner for the MLC service with enhanced logging.
"""
import os
import sys

if __name__ == "__main__":
    # Set debug flag for the environment
    os.environ["MLC_DEBUG"] = "1"
    
    # Import and run the app
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    from llm_service.app import main
    main()
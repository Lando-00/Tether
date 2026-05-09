#!/usr/bin/env python
"""Display auto-generated tool schemas.

Note: this script does NOT load .env. Per _synthesis.md §4 Phase 2 step 26,
.env loading is centralized in ``tether_service.app.__main__`` (the HTTP
entry point) and ``conftest.py`` (test runs). If you need environment
variables (e.g. ``BRAVE_API_KEY``) when running this utility directly,
either invoke after ``set BRAVE_API_KEY=...`` or wrap the call yourself
with ``from dotenv import load_dotenv; load_dotenv()``.
"""
import sys
sys.path.insert(0, '.')

from tether_service.core.tool_registry import ToolRegistry
from tether_service.core.config import load_settings_legacy

# Load config and create registry
config = load_settings_legacy()
registry = ToolRegistry(
    registry_cfg=config['tools']['registry'],
    enabled=config['tools']['enabled']
)

# Get all tool schemas
print('=' * 80)
print('AVAILABLE TOOLS - Auto-Generated Schemas')
print('=' * 80)
print()

for tool_name, tool_instance in registry.all().items():
    schema = tool_instance.auto_schema
    func_schema = schema.get('function', schema)  # Handle nested structure
    
    print(f'🔧 TOOL: {func_schema["name"]}')
    print('-' * 80)
    print(f'📝 Description:')
    print(f'   {func_schema["description"].strip()}')
    print()
    print(f'📋 Parameters:')
    
    params = func_schema.get('parameters', {})
    props = params.get('properties', {})
    required = params.get('required', [])
    
    if not props:
        print('   (No parameters)')
    else:
        for param_name, param_info in props.items():
            req_marker = '✓ REQUIRED' if param_name in required else '  optional'
            param_type = param_info.get('type', 'unknown')
            param_desc = param_info.get('description', 'No description')
            
            print(f'   • {param_name} ({param_type}) [{req_marker}]')
            print(f'     {param_desc}')
    
    print()
    print('=' * 80)
    print()

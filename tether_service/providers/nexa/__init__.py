"""Nexa provider package.

STUB ONLY — every concrete method raises NotImplementedError. The class
exists to validate the Provider v2 contract supports Snapdragon NPU
providers via Seam A (briefing §12.6) without further contract changes.

Future sessions will:
    - Implement NexaProvider.stream_typed against the Nexa Python SDK
    - Decide whether NexaProvider implements HardwareLifecycle (the NPU
      teardown pattern differs from MLC's OpenCL destructor sequence)
    - Add NEXA_TOKEN handling via SecretsProvider (Phase 6.5)

Per ratified plan: NO Nexa runtime impl ships in this refactor; the stub
verifies forward compatibility only.
"""

"""GitHub Copilot SDK provider package.

Import-safe by design: the optional ``github-copilot-sdk`` dependency is
loaded lazily by ``CopilotProvider`` only when the provider is selected.
"""

__all__ = ["CopilotProvider"]


def __getattr__(name: str):
    if name == "CopilotProvider":
        from tether.providers.copilot.provider import CopilotProvider

        return CopilotProvider
    raise AttributeError(name)


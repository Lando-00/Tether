"""Connector framework package marker.

Per connector spec §3 (locked design at
``C:/Users/lovan/.copilot/session-state/5c8a15fc-11c0-4eef-98e1-cf5cd5f6a520/plan.md``)
and synthesis §4 Phase 4.5 (steps 47a-47c), §10 connector spec integration.

A connector is a long-lived plugin that bridges Tether to an external system
(WhatsApp, Gmail, future calendar/filesystem). It owns its auth lifecycle,
exposes one or more :class:`tether_service.core.interfaces.Tool` instances
the model can call, and may publish :class:`InboundEvent` values onto the
inbox stream.

Phase 4.5 ships the *contracts only*:

- :class:`Connector` ABC (``base.py``)
- :class:`ConnectorState` enum + dataclasses (``types.py``)
- :class:`SecretsProvider` interface (``tether_service.core.secrets``)
- :class:`ConnectorsSettings` config schema (``tether_service.config.settings``)

Concrete implementations land later:

- ConnectorRegistry: ``p4_5-connector-registry``
- HTTP routes (``/api/v1/connectors/*``): ``p4_5-engine-wiring-routes``
- Echo connector (smoke test): ``p4_5-echo-connector-tests``
- WhatsApp / Gmail connectors: Phase 2a / 2b

This package is library-first (per synthesis §4 Phase 2 step 22): importing
``tether_service`` does NOT import this module. ``Engine.from_settings``
(when ConnectorRegistry lands) is the sole entry point that imports it.
"""

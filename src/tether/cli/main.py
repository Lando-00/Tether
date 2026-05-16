"""
A modern CLI for interacting with the Tether service.
"""
import json

# --- Configuration ---
# Default API base URL. Reassigned by main() if --api-url is passed.
# Reads from TETHER_API_URL env var if set (allows shell-level override
# without a flag, useful for development).
import os as _os
import time
from pathlib import Path
from typing import Optional

import requests
import typer
from prompt_toolkit import prompt as ptk_prompt
from prompt_toolkit.formatted_text import FormattedText
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

API_BASE_URL = _os.environ.get("TETHER_API_URL", "http://127.0.0.1:8080/api/v1")

# CSRF header name must match Settings.security.csrf_token.header_name.
_CSRF_HEADER = "X-Tether-CSRF"


def _read_csrf_token() -> Optional[str]:
    """Read the persisted CSRF token from platformdirs, if present.

    P0-B3 / ADR-0012: the server writes its generated CSRF token to
    ``platformdirs.user_config_dir('Tether', appauthor=False)/csrf_token``
    at mode 0600. CLI clients read the same path to bootstrap the
    ``X-Tether-CSRF`` header without scraping stderr. Returns ``None`` if
    the file is absent (CSRF disabled) or unreadable.
    """
    try:
        import platformdirs
    except ImportError:
        return None
    path = Path(platformdirs.user_config_dir("Tether", appauthor=False)) / "csrf_token"
    try:
        token = path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return None
    return token or None


def _mutating_headers(extra: Optional[dict] = None) -> dict:
    """Return headers for state-changing requests.

    Always sets ``Content-Type: application/json`` so calls land cleanly
    past the Phase-9 P0-B2 ``RequireJsonContentTypeMiddleware`` even when
    the request has no body (``POST /sessions``, ``DELETE /sessions/{id}``,
    ``POST /connectors/{id}/login/begin``, etc.). Callers passing ``json=``
    to ``requests`` get the same header for free, but no-body mutating
    calls need it explicitly. Per-call ``extra`` wins so callers can
    override (e.g. the chat stream uses ``Accept: application/x-ndjson``).
    Also injects the CSRF token if one is configured.
    """
    headers: dict = {"Content-Type": "application/json"}
    if extra:
        headers.update(extra)
    token = _read_csrf_token()
    if token is not None and _CSRF_HEADER not in headers:
        headers[_CSRF_HEADER] = token
    return headers



# --- Rich Console Initialization ---
console = Console()
app = typer.Typer(
    name="tether-cli",
    help="A modern CLI for interacting with the Tether service.",
    add_completion=False,
    invoke_without_command=True,
)


def _render_connect_error(exc: Exception, action: str) -> None:
    """Pretty-print a connection error with remediation guidance.

    Used by the entry-point fetches (models, sessions, tools) so users
    get a clear pointer when the server isn't running rather than a
    bare requests.RequestException repr.
    """
    exc_name = type(exc).__name__
    msg = str(exc).strip() or "(no detail)"
    panel_body = (
        f"[bold red]Could not {action}.[/bold red]\n\n"
        f"[bold]URL:[/bold] {API_BASE_URL}\n"
        f"[bold]Error:[/bold] {exc_name}: {msg}\n\n"
        f"[dim]The Tether server may not be running. Start it with:[/dim]\n"
        f"  [bold cyan]python -m tether.app[/bold cyan]   (or [bold cyan]tether-server[/bold cyan])\n\n"
        f"[dim]Or pass a different URL with --api-url:[/dim]\n"
        f"  [bold cyan]tether-cli --api-url http://host:port/api/v1[/bold cyan]\n\n"
        f"[dim]Troubleshooting: docs/runbooks/fresh-env-setup.md[/dim]"
    )
    console.print(Panel(panel_body, title="Connection error", border_style="red"))


def _normalise_api_url(api_url: str) -> str:
    """Accept either the API root or the server root and return /api/v1."""
    base = api_url.rstrip("/")
    if base.endswith("/api/v1"):
        return base
    if base.endswith("/api"):
        return f"{base}/v1"
    return f"{base}/api/v1"


def _set_api_base_url(api_url: Optional[str]) -> str:
    """Update API_BASE_URL for command-scoped --api-url overrides."""
    global API_BASE_URL
    if api_url:
        API_BASE_URL = _normalise_api_url(api_url)
    return API_BASE_URL


def _connector_label(connector: str) -> str:
    if connector.lower() == "whatsapp":
        return "WhatsApp"
    return connector.replace("_", " ").replace("-", " ").title()


def _response_error_detail(response) -> str:
    try:
        body = response.json()
    except ValueError:
        body = None

    if isinstance(body, dict):
        detail = body.get("detail") or body.get("error") or body.get("message")
        if detail is not None:
            return str(detail)
    text = getattr(response, "text", "")
    return text.strip() or "No error detail returned."


def _render_http_error(exc: requests.HTTPError, action: str) -> None:
    response = getattr(exc, "response", None)
    if response is None:
        _render_connect_error(exc, action)
        return

    status = getattr(response, "status_code", "unknown")
    detail = _response_error_detail(response)
    # Show the actual request URL (sub-route + query) rather than the
    # base ``API_BASE_URL`` — the base alone is uninformative when a
    # specific endpoint 4xx's. Falls back to ``API_BASE_URL`` if the
    # ``requests`` response object lacks a ``url`` (legacy / mock).
    request_url = getattr(response, "url", None) or API_BASE_URL
    panel_body = (
        f"[bold red]Could not {action}.[/bold red]\n\n"
        f"[bold]URL:[/bold] {request_url}\n"
        f"[bold]Status:[/bold] {status}\n"
        f"[bold]Detail:[/bold] {detail}"
    )
    console.print(Panel(panel_body, title="HTTP error", border_style="red"))


def _response_json(response, action: str):
    try:
        return response.json()
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] Invalid JSON while trying to {action}: {exc}")
        raise typer.Exit(1) from exc


def _render_raw_qr(payload: str, *, fallback: bool = False) -> None:
    console.print(Panel(payload, title="Raw QR payload", border_style="yellow"))
    if fallback:
        console.print(
            "[yellow]Terminal QR rendering needs the optional 'qrcode' package. "
            "Install it with 'pip install qrcode', or copy this string into a QR "
            "generator and scan that code with WhatsApp.[/yellow]"
        )


def _render_qr_ascii(payload: str) -> bool:
    try:
        import qrcode
    except ImportError:
        return False

    qr = qrcode.QRCode(border=1)
    qr.add_data(payload)
    qr.make(fit=True)
    matrix = qr.get_matrix()

    for row_idx in range(0, len(matrix), 2):
        chars = []
        top = matrix[row_idx]
        bottom = matrix[row_idx + 1] if row_idx + 1 < len(matrix) else [False] * len(top)
        for top_dot, bottom_dot in zip(top, bottom):
            if top_dot and bottom_dot:
                chars.append("█")
            elif top_dot:
                chars.append("▀")
            elif bottom_dot:
                chars.append("▄")
            else:
                chars.append(" ")
        console.print("".join(chars))
    return True


def _render_login_prompt(connector: str, prompt: dict, qr_format: str) -> None:
    label = _connector_label(connector)
    kind = prompt.get("kind")
    payload = str(prompt.get("payload") or "")

    if kind == "qr_code":
        console.print(f"\n[bold]QR code for {label}:[/bold]")
        if qr_format == "raw":
            _render_raw_qr(payload)
        elif qr_format == "png":
            console.print("[yellow]PNG output is not implemented yet; showing raw QR payload.[/yellow]")
            _render_raw_qr(payload)
        elif not _render_qr_ascii(payload):
            _render_raw_qr(payload, fallback=True)
        return

    if kind == "url":
        console.print(Panel(payload, title=f"{label} login URL", border_style="cyan"))
        return

    console.print(Panel(payload, title=f"{label} login prompt", border_style="cyan"))


def _print_scan_instructions() -> None:
    console.print(
        "Scan the QR with your phone:\n"
        "  1. Open WhatsApp\n"
        "  2. Settings → Linked Devices → Link a Device\n"
        "  3. Scan this QR code\n"
        "Waiting for pair (will refresh QR if it rotates)..."
    )


def _print_connector_health(connector: str) -> None:
    try:
        response = requests.get(f"{API_BASE_URL}/connectors", timeout=5)
        response.raise_for_status()
        connectors = response.json()
    except requests.RequestException as exc:
        console.print(f"[dim]Logged in, but connector health could not be fetched: {exc}[/dim]")
        return
    except ValueError:
        console.print("[dim]Logged in, but connector health response was not JSON.[/dim]")
        return

    if not isinstance(connectors, list):
        return

    for item in connectors:
        if not isinstance(item, dict) or item.get("id") != connector:
            continue
        health = item.get("health") if isinstance(item.get("health"), dict) else {}
        state = health.get("state") or "unknown"
        detail = health.get("detail")
        suffix = f" ({detail})" if detail else ""
        console.print(f"[dim]Health: {state}{suffix}[/dim]")
        return


# --- API Interaction Functions ---

def get_available_models() -> list:
    """Fetches the list of available models from the service."""
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        response.raise_for_status()
        # The new API returns a list of strings directly
        return response.json()
    except requests.RequestException as e:
        _render_connect_error(e, "fetch model list")
        raise typer.Exit(1)


def get_available_model_details() -> list:
    """Fetch per-model capability metadata from ``/models/details``.

    Returns ``[]`` on connection error so callers can fall back to the
    legacy plain-string list. Each entry is a ``ModelDetails`` dict:
    ``{id, provider_kind, source, context_window, supports_thinking,
    supports_reasoning_effort, reasoning_efforts, is_default}``.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/models/details", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return []


def _model_details_for(
    model_name: str, *, provider_id: Optional[str] = None
) -> Optional[dict]:
    """Look up a single model's :class:`ModelDetails` dict, or ``None``.

    When ``provider_id`` is given (and is not the wrap sentinel), require
    a row that matches BOTH ``id`` and ``provider_id``. ADR-0021 P2 follow-up:
    duplicate model names across providers are valid; the legacy "first
    matching id" lookup silently picked the wrong row when reasoning
    capabilities differed across providers.
    """
    for info in get_available_model_details():
        if info.get("id") != model_name:
            continue
        if provider_id is None or info.get("provider_id") == provider_id:
            return info
    return None


# Sentinel value returned by /models/details for pre-registry servers that do
# not have a provider registry.  When every row carries this value the
# per-provider "Provider" column is not shown (ADR-0021 Phase 2.B).
_PROVIDER_ID_SENTINEL = "_unwrapped_"


def get_provider_health() -> tuple[Optional[dict], Optional[str]]:
    """Return ({pid: {healthy, kind, source, error}}, default_provider_id)
    from /readyz, or (None, None) on connection error.

    Phase 2.A adds the ``providers:`` block + ``default_provider_id`` to
    /readyz; older servers return None for both so the slash command can
    render a degraded fallback.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/readyz", timeout=5)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            return None, None
        providers = data.get("providers")
        default_pid = data.get("default_provider_id")
        if not isinstance(providers, dict):
            return None, None
        return providers, default_pid
    except requests.RequestException:
        return None, None


def get_providers_table() -> Optional[Table]:
    """Build the Rich Table for the ``\\providers`` slash command.

    Returns ``None`` when the server lacks the multi-provider block (older
    server) so the caller can render a degraded fallback message instead.
    """
    providers, default_pid = get_provider_health()
    if providers is None:
        return None

    table = Table(title="Registered Providers", border_style="cyan")
    table.add_column("id", style="bold")
    table.add_column("kind")
    table.add_column("source")
    table.add_column("default", justify="center")
    table.add_column("health")
    table.add_column("error", style="dim")

    for pid, info in providers.items():
        is_default = pid == default_pid
        healthy = info.get("healthy", False)
        table.add_row(
            pid,
            info.get("kind", "—"),
            info.get("source", "—"),
            "✓" if is_default else "",
            "[green]healthy[/green]" if healthy else "[red]DOWN[/red]",
            info.get("error") or "",
        )

    return table


def get_available_tools() -> list:
    """Fetches the list of registered tools (name/description/parameters).

    Returns [] on connection error so the ``\\tools`` slash command can
    fall back to "no tools" rendering rather than tearing down the chat.
    Synthesis §4 Phase 4 step 42 (auto_schema) — exposed via
    /api/v1/tools (added in cli-polish branch).
    """
    try:
        response = requests.get(f"{API_BASE_URL}/tools", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        console.print(f"[red]Could not fetch tools:[/red] {e}")
        return []

def get_sessions() -> list:
    """Fetches the list of active sessions."""
    try:
        response = requests.get(f"{API_BASE_URL}/sessions")
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return []

def create_session() -> Optional[str]:
    """Creates a new session and returns its ID."""
    try:
        response = requests.post(f"{API_BASE_URL}/sessions", headers=_mutating_headers())
        response.raise_for_status()
        session_id = response.json().get("session_id")
        console.print(f"✅ New session created: [yellow]{session_id}[/yellow]")
        return session_id
    except requests.RequestException as e:
        console.print(f"[bold red]Error:[/bold red] Could not create session: {e}")
        return None

def delete_session(session_id: str):
    """Deletes a specific session."""
    try:
        response = requests.delete(f"{API_BASE_URL}/sessions/{session_id}", headers=_mutating_headers())
        response.raise_for_status()
        console.print(f"✅ {response.json().get('detail', 'Session deleted.')}")
    except requests.RequestException as e:
        console.print(f"[bold red]Error deleting session {session_id}:[/bold red] {e}")

def delete_all_sessions():
    """Deletes all sessions on the server."""
    try:
        response = requests.delete(f"{API_BASE_URL}/sessions", headers=_mutating_headers())
        response.raise_for_status()
        console.print(f"✅ {response.json().get('detail', 'All sessions deleted.')}")
    except requests.RequestException as e:
        console.print(f"[bold red]Error deleting all sessions:[/bold red] {e}")

def get_session_history(session_id: str) -> list:
    """Fetches the message history for a given session."""
    try:
        response = requests.get(f"{API_BASE_URL}/sessions/{session_id}/messages")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        console.print(f"[bold red]Error fetching history for session {session_id}:[/bold red] {e}")
        return []

def unload_all_models():
    """Calls the endpoint to unload all models from the cache."""
    console.print("Attempting to unload all models from memory...")
    available_models = get_available_models()
    if not available_models:
        console.print(
            "[yellow]Warning:[/yellow] No available models found to specify "
            "for unload request. The cache might be empty already."
        )
        return

    # The new API unloads all models, but still requires a model name in the path
    model_to_specify = available_models[0]

    try:
        response = requests.post(f"{API_BASE_URL}/models/{model_to_specify}/unload", headers=_mutating_headers())
        response.raise_for_status()
        console.print(f"✅ {response.json().get('detail', 'Unload command sent successfully.')}")
    except requests.RequestException as e:
        console.print(f"[bold red]Error unloading models:[/bold red] {e}")


def display_history(messages: list):
    """Renders the chat history using a more structured format."""
    if not messages:
        return

    console.print(Panel("Chat History", style="bold blue", expand=False))

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if role == "user":
            panel_content = Text(content, style="cyan")
            console.print(Panel(panel_content, title="You", title_align="left", border_style="cyan"))
        elif role == "assistant":
            # Assistant messages can be complex (text or tool calls)
            if isinstance(content, list):  # Tool call
                for part in content:
                    if part.get("type") == "tool_use":
                        tool_name = part.get("tool_name")
                        tool_args = part.get("tool_args")
                        panel_content = Text(f"Tool: {tool_name}\nArgs: {tool_args}", style="green")
                        console.print(
                            Panel(
                                panel_content,
                                title="Assistant (Tool Call)",
                                title_align="left",
                                border_style="green",
                            )
                        )
            else:
                panel_content = Text(content, style="green")
                console.print(Panel(panel_content, title="Assistant", title_align="left", border_style="green"))
        elif role == "tool":
            tool_name = msg.get("tool_name", "unknown_tool")
            tool_content = msg.get("content", "")
            panel_content = Text(str(tool_content), style="yellow")
            console.print(
                Panel(
                    panel_content,
                    title=f"Tool Output ({tool_name})",
                    title_align="left",
                    border_style="yellow",
                )
            )
    console.print()


def manage_sessions() -> tuple[Optional[str], str]:
    """Display and manage chat sessions. Returns (session_id, action)."""
    sessions = get_sessions()

    table = Table(title="Chat Session Management", border_style="blue", show_header=False)
    table.add_column("Key", style="bold cyan")
    table.add_column("Action")

    table.add_row("c", "Create a new chat")

    choices = {"c": "Create a new chat"}
    if sessions:
        table.add_section()
        for i, s in enumerate(sessions):
            session_id = s.get('session_id', 'N/A')
            created_at = s.get('created_at', 'N/A')
            table.add_row(str(i+1), f"Resume session from {created_at} ([yellow]{session_id[:8]}...[/yellow])")
            choices[str(i+1)] = f"Resume session {session_id}"
        table.add_section()
        table.add_row("d", "Delete a session")
        table.add_row("da", "Delete ALL sessions")
        choices["d"] = "Delete a session"
        choices["da"] = "Delete ALL sessions"

    table.add_section()
    table.add_row("u", "Unload all models from memory")
    table.add_row("q", "Quit")
    choices["u"] = "Unload all models from memory"
    choices["q"] = "Quit"

    console.print(table)

    prompt_text = "\nChoose an action"
    action = Prompt.ask(prompt_text, choices=list(choices.keys()), default="c").lower()

    if action == "c":
        return None, "new"
    elif action == "u":
        unload_all_models()
        return None, "manage"
    elif action == "da":
        if not sessions:
            console.print("[red]No sessions to delete.[/red]")
            return None, "manage"
        confirm = Prompt.ask(
            "[bold yellow]Are you sure you want to delete all sessions? (y/n)[/bold yellow]",
            choices=["y", "n"],
            default="n"
        )
        if confirm.lower() == 'y':
            delete_all_sessions()
        return None, "manage"
    elif action == "d":
        if not sessions:
            console.print("[red]No sessions to delete.[/red]")
            return None, "manage"
        del_choice = IntPrompt.ask(
            "Enter the number of the session to DELETE",
            choices=[str(i + 1) for i in range(len(sessions))],
            show_choices=False,
        )
        session_to_delete = sessions[del_choice - 1]['session_id']
        delete_session(session_to_delete)
        return None, "manage"
    elif action.isdigit() and sessions and 0 < int(action) <= len(sessions):
        session_id = sessions[int(action) - 1]['session_id']
        return session_id, "resume"
    elif action == "q":
        return None, "quit"
    else:
        console.print("[red]Invalid choice.[/red]")
        return None, "manage"


def _interactive_model_select(
    rows: list,
    *,
    title: str = "Available Models",
) -> tuple[str, Optional[str]]:
    """Render a Rich table of *rows* from ``/models/details`` and prompt.

    Returns ``(model_id, provider_id_or_None)``.

    Default-row markers:
      ★ — globally-default model (``is_default=True`` AND
          ``provider_id == default_provider_id`` from /readyz).
      ☆ — provider-local default only (``is_default=True`` under a
          non-global-default provider).
    """
    # Show provider_id column only when the registry is active — i.e. any
    # row carries a non-sentinel value.  Single-provider legacy servers send
    # _PROVIDER_ID_SENTINEL for every row, so the column is suppressed there.
    show_provider_col = any(
        r.get("provider_id") and r.get("provider_id") != _PROVIDER_ID_SENTINEL
        for r in rows
    )

    # Sort by (provider_id, id) so each provider's models group together.
    sorted_rows = sorted(
        rows,
        key=lambda r: (r.get("provider_id") or "", r.get("id") or ""),
    )

    # Fetch default_provider_id for the global-default marker (★ vs ☆).
    _, default_pid = get_provider_health()

    models_table = Table(title=title, border_style="cyan")
    models_table.add_column("#", style="bold cyan", justify="right")
    models_table.add_column("Model", style="bold")
    models_table.add_column("Provider")
    models_table.add_column("Source")
    models_table.add_column("Ctx", justify="right")
    models_table.add_column("Reasoning")
    models_table.add_column("Default", justify="center")

    for i, info in enumerate(sorted_rows):
        reasoning = (
            ",".join(info.get("reasoning_efforts") or [])
            if info.get("supports_reasoning_effort")
            else "—"
        )
        ctx = info.get("context_window")
        ctx_str = f"{ctx:,}" if isinstance(ctx, int) else "—"

        pid = info.get("provider_id")
        if pid == _PROVIDER_ID_SENTINEL:
            pid = None

        is_default = info.get("is_default", False)
        if is_default:
            # ★ = server-wide default; ☆ = default within a non-primary provider.
            default_marker = "★" if (default_pid is None or pid == default_pid) else "☆"
        else:
            default_marker = ""

        # Provider column: show provider_id when registry is active,
        # otherwise fall back to provider_kind (single-provider legacy).
        provider_cell = (pid or "—") if show_provider_col else info.get("provider_kind", "—")

        models_table.add_row(
            str(i + 1),
            info.get("id", "?"),
            provider_cell,
            info.get("source", "—"),
            ctx_str,
            reasoning,
            default_marker,
        )

    console.print(models_table)

    while True:
        try:
            choice_str = Prompt.ask(
                "\nPlease enter the number of the model you want to use",
                default="1",
            )
            if not choice_str.strip():
                choice_str = "1"
            choice = int(choice_str)
            if 1 <= choice <= len(sorted_rows):
                selected = sorted_rows[choice - 1]
                sel_id = selected.get("id", "")
                sel_pid = selected.get("provider_id")
                if sel_pid == _PROVIDER_ID_SENTINEL:
                    sel_pid = None
                return sel_id, sel_pid
            console.print(
                f"[red]Invalid choice. Please enter a number between "
                f"1 and {len(sorted_rows)}.[/red]"
            )
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/red]")


def select_model(
    model_name: Optional[str],
    provider: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """Validate or interactively select a model; return ``(model_id, provider_id)``.

    When *model_name* is given:

    - Exactly one ``/models/details`` row matches → return its id and
      provider_id (``None`` when the sentinel ``_unwrapped_`` is present).
    - Multiple rows share the same id (ambiguous; multi-provider):
      - *provider* is supplied → filter to the matching row.
      - *provider* is absent → drop into the interactive selector
        pre-filtered to those rows so the user can pick a provider.
    - No details row matches → fall back to the legacy ``/models`` plain
      list.  If also absent, print an error and exit.

    When *model_name* is ``None``, open the interactive selector over all
    available models (using ``/models/details`` when possible, otherwise
    the legacy plain list).

    Returns ``(model_id, provider_id_or_None)``.
    """
    details = get_available_model_details()

    if model_name:
        if not details:
            # /models/details unavailable; fall back to legacy /models list.
            models = get_available_models()
            if model_name in models:
                return model_name, None
            console.print(f"[bold red]Error:[/bold red] Model '{model_name}' not found.")
            raise typer.Exit(1)

        matching_rows = [
            d for d in details
            if isinstance(d, dict) and d.get("id") == model_name
        ]

        if len(matching_rows) == 0:
            console.print(f"[bold red]Error:[/bold red] Model '{model_name}' not found.")
            raise typer.Exit(1)

        if len(matching_rows) == 1:
            pid = matching_rows[0].get("provider_id")
            if pid == _PROVIDER_ID_SENTINEL:
                pid = None
            return model_name, pid

        # Multiple rows share the same model id across different providers.
        if provider is not None:
            filtered = [r for r in matching_rows if r.get("provider_id") == provider]
            if len(filtered) == 1:
                pid = filtered[0].get("provider_id")
                if pid == _PROVIDER_ID_SENTINEL:
                    pid = None
                return model_name, pid
            if len(filtered) == 0:
                console.print(
                    f"[bold red]Error:[/bold red] Model '{model_name}' not found "
                    f"under provider '{provider}'."
                )
                raise typer.Exit(1)
            # len > 1 is a server configuration error.
            console.print(
                f"[bold red]Error:[/bold red] Multiple entries for model "
                f"'{model_name}' under provider '{provider}'. "
                "Check server configuration."
            )
            raise typer.Exit(1)

        # No --provider given; drop into selector pre-filtered to the
        # ambiguous rows so the user can pick which provider to use.
        console.print(
            f"[yellow]Model '{model_name}' is ambiguous across providers; "
            "pick a provider:[/yellow]"
        )
        return _interactive_model_select(
            matching_rows,
            title=f"Providers for model '{model_name}'",
        )

    # No model_name given: open the interactive selector over all models.
    console.print("🔍 Searching for available models...")
    if details:
        return _interactive_model_select(details)

    # Legacy fallback: /models/details unavailable.
    available_models = get_available_models()
    if not available_models:
        console.print("[bold red]Error:[/bold red] No models found.")
        console.print(
            "Please make sure your compiled models are correctly placed "
            "and the service is running."
        )
        raise typer.Exit(1)

    console.print("\nAvailable Models:")
    for i, name in enumerate(available_models):
        console.print(f"  [bold cyan][{i+1}][/bold cyan] {name}")

    while True:
        try:
            choice_str = Prompt.ask(
                "\nPlease enter the number of the model you want to use",
                default="1",
            )
            if not choice_str.strip():
                choice_str = "1"
            choice = int(choice_str)
            if 1 <= choice <= len(available_models):
                return available_models[choice - 1], None
            console.print(
                "[red]Invalid choice. Please enter a number between "
                f"1 and {len(available_models)}.[/red]"
            )
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/red]")


def select_reasoning_effort(
    model_name: str,
    current: Optional[str] = None,
    *,
    provider_id: Optional[str] = None,
) -> Optional[str]:
    """Prompt the user for a reasoning effort, scoped to the chosen model.

    Returns the new value, or ``None`` when the user clears the override.
    If the model does not advertise ``supports_reasoning_effort``,
    prints a notice and returns ``current`` unchanged (no-op).

    ``provider_id`` scopes the lookup so duplicate model names across
    providers resolve to the right reasoning whitelist (ADR-0021 P2 follow-up).
    """
    info = _model_details_for(model_name, provider_id=provider_id)
    if info is None:
        console.print(
            f"[yellow]No metadata for model '{model_name}' — cannot check "
            "reasoning support. Send the request anyway with the previous "
            "value.[/yellow]"
        )
        return current
    if not info.get("supports_reasoning_effort"):
        console.print(
            f"[yellow]Model '{model_name}' does not support "
            "reasoning_effort. Use [bold]\\models[/bold] to switch to a "
            "model that does.[/yellow]"
        )
        return None
    accepted = list(info.get("reasoning_efforts") or [])
    if not accepted:
        console.print(
            f"[yellow]Model '{model_name}' advertised reasoning support "
            "but exposed an empty efforts list.[/yellow]"
        )
        return current
    options = [*accepted, "off"]
    console.print(
        f"\nReasoning efforts for [bold]{model_name}[/bold]: "
        + ", ".join(f"[cyan]{o}[/cyan]" for o in options)
    )
    default = current if current in accepted else accepted[0]
    while True:
        choice = Prompt.ask(
            "Choose effort ('off' to clear)",
            default=default,
        ).strip().lower()
        if choice == "off":
            return None
        if choice in accepted:
            return choice
        console.print(
            f"[red]Invalid choice. Pick one of: {', '.join(options)}[/red]"
        )


@app.callback(invoke_without_command=True)
def cli(
    ctx: typer.Context,
    model_name: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="The name of the model to use. If not provided, a list will be shown.",
    ),
    api_url: str = typer.Option(
        API_BASE_URL,
        "--api-url",
        help="Base URL of the Tether HTTP API (default: http://127.0.0.1:8080/api/v1, or TETHER_API_URL env var).",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug mode to show detailed event information.",
    ),
    show_thinking: bool = typer.Option(
        True,
        "--show-thinking",
        help="Enable to show the model's thinking process.",
    ),
    reasoning_effort: Optional[str] = typer.Option(
        None,
        "--reasoning-effort",
        help=(
            "Initial reasoning effort hint for the chosen model "
            "(e.g. 'minimal', 'low', 'medium', 'high' for GitHub Copilot SDK "
            "models). Use the in-chat '\\reasoning' command to change it. "
            "Server returns 422 if the model does not support the value."
        ),
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-P",
        help=(
            "Provider id to route the request to. When omitted, the server "
            "uses its configured default. Use `\\providers` in the REPL to "
            "see available ids and health."
        ),
    ),
):
    """Run chat by default, or choose a subcommand."""
    global API_BASE_URL
    API_BASE_URL = _normalise_api_url(api_url)
    if ctx.invoked_subcommand is None:
        main(
            model_name=model_name,
            api_url=API_BASE_URL,
            debug=debug,
            show_thinking=show_thinking,
            reasoning_effort=reasoning_effort,
            provider=provider,
        )


@app.command()
def connect(
    connector: str = typer.Argument(..., help="Connector id to authenticate, e.g. whatsapp."),
    api_url: Optional[str] = typer.Option(
        None,
        "--api-url",
        help="Base URL of the Tether HTTP API or server root.",
    ),
    timeout: float = typer.Option(
        180.0,
        "--timeout",
        min=1.0,
        help="Maximum seconds to wait for pairing.",
    ),
    qr_format: str = typer.Option(
        "ascii",
        "--qr-format",
        help="QR rendering format: ascii, raw, or png.",
    ),
) -> None:
    """Authenticate a connector; WhatsApp uses the QR pairing flow."""
    _set_api_base_url(api_url)
    qr_format = qr_format.lower()
    if qr_format not in {"ascii", "raw", "png"}:
        console.print("[bold red]Error:[/bold red] --qr-format must be ascii, raw, or png.")
        raise typer.Exit(1)

    label = _connector_label(connector)
    begin_url = f"{API_BASE_URL}/connectors/{connector}/login/begin"
    complete_url = f"{API_BASE_URL}/connectors/{connector}/login/complete"

    try:
        response = requests.post(begin_url, headers=_mutating_headers(), timeout=10)
        response.raise_for_status()
    except requests.HTTPError as exc:
        _render_http_error(exc, f"begin {label} login")
        raise typer.Exit(1) from exc
    except requests.RequestException as exc:
        _render_connect_error(exc, f"begin {label} login")
        raise typer.Exit(1) from exc

    prompt = _response_json(response, f"begin {label} login")
    if not isinstance(prompt, dict):
        console.print(f"[bold red]Error:[/bold red] Unexpected {label} login prompt.")
        raise typer.Exit(1)

    _render_login_prompt(connector, prompt, qr_format)
    if prompt.get("kind") == "qr_code":
        _print_scan_instructions()

    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            console.print("[bold red]Timeout — try again.[/bold red]")
            raise typer.Exit(4)

        poll_timeout = min(30.0, remaining)
        try:
            response = requests.post(
                complete_url,
                json={"payload": {"timeout_sec": poll_timeout}},
                headers=_mutating_headers(),
                timeout=poll_timeout + 5,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            _render_http_error(exc, f"complete {label} login")
            raise typer.Exit(1) from exc
        except requests.RequestException as exc:
            _render_connect_error(exc, f"complete {label} login")
            raise typer.Exit(1) from exc

        result = _response_json(response, f"complete {label} login")
        if not isinstance(result, dict):
            console.print(f"[bold red]Error:[/bold red] Unexpected {label} login result.")
            raise typer.Exit(1)

        state = result.get("state")
        detail = result.get("detail") or ""

        if state == "ready":
            console.print(f"[bold green]Logged in to {label}.[/bold green]")
            _print_connector_health(connector)
            return

        if state == "logged_out":
            console.print("[bold red]Unpaired during scan.[/bold red]")
            raise typer.Exit(2)

        if state == "error":
            console.print(f"[bold red]{detail or f'{label} login failed.'}[/bold red]")
            raise typer.Exit(3)

        if state == "authenticating":
            if detail == "qr_scan_timeout":
                console.print("[bold yellow]QR expired without scan; retry?[/bold yellow]")
                raise typer.Exit(4)
            next_prompt = result.get("next_prompt")
            if isinstance(next_prompt, dict):
                console.print("[yellow]QR refreshed; scan the new code.[/yellow]")
                _render_login_prompt(connector, next_prompt, qr_format)
                continue
            console.print(f"[dim]Still waiting for {label} pairing...[/dim]")
            continue

        console.print(f"[bold red]Unexpected {label} login state:[/bold red] {state!r}")
        raise typer.Exit(3)


@app.command()
def logout(
    connector: str = typer.Argument(..., help="Connector id to log out, e.g. whatsapp."),
    api_url: Optional[str] = typer.Option(
        None,
        "--api-url",
        help="Base URL of the Tether HTTP API or server root.",
    ),
) -> None:
    """Log out a connector and delete its persisted credentials."""
    _set_api_base_url(api_url)
    label = _connector_label(connector)
    logout_url = f"{API_BASE_URL}/connectors/{connector}/logout"

    try:
        response = requests.post(logout_url, headers=_mutating_headers(), timeout=10)
        response.raise_for_status()
    except requests.HTTPError as exc:
        _render_http_error(exc, f"log out from {label}")
        raise typer.Exit(1) from exc
    except requests.RequestException as exc:
        _render_connect_error(exc, f"log out from {label}")
        raise typer.Exit(1) from exc

    data = _response_json(response, f"log out from {label}")
    if not isinstance(data, dict) or data.get("ok") is not True:
        console.print(f"[bold red]Could not log out from {label}:[/bold red] {data}")
        raise typer.Exit(1)

    console.print(
        "Unlinked Tether's WhatsApp device session and deleted local credentials. "
        f"Your WhatsApp account on your phone is unaffected. (Logged out from {label}.)"
    )


@app.command("chat")
def main(
    model_name: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="The name of the model to use. If not provided, a list will be shown.",
    ),
    api_url: str = typer.Option(
        API_BASE_URL,
        "--api-url",
        help="Base URL of the Tether HTTP API (default: http://127.0.0.1:8080/api/v1, or TETHER_API_URL env var).",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug mode to show detailed event information.",
    ),
    show_thinking: bool = typer.Option(
        True,
        "--show-thinking",
        help="Enable to show the model's thinking process.",
    ),
    reasoning_effort: Optional[str] = typer.Option(
        None,
        "--reasoning-effort",
        help=(
            "Initial reasoning effort hint for the chosen model. Use "
            "'\\reasoning' mid-chat to change. Server returns 422 if "
            "the model does not advertise support."
        ),
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-P",
        help=(
            "Provider id to route the request to. When omitted, the server "
            "uses its configured default. Use `\\providers` in the REPL to "
            "see available ids and health."
        ),
    ),
):
    """
    Main entry point for the Tether CLI.
    """
    # Rebind the module-level API_BASE_URL so helper functions pick up the
    # flag value. Idempotent — reassigning the global is safe across calls
    # (e.g., when `\menu` recurses into main()).
    global API_BASE_URL
    API_BASE_URL = api_url

    console.print(Panel.fit(
        "[bold blue]Welcome to the Tether CLI![/bold blue]\n"
        f"[dim]API: {API_BASE_URL}[/dim]",
        style="bold blue"
    ))

    model_name_arg = model_name
    model_name, provider_id = select_model(model_name_arg, provider)

    # Sanity-check the startup --reasoning-effort against the chosen
    # model. We don't error here — the server is the authoritative gate —
    # but we surface a warning so the user can fix it before the first
    # 422 turn. Skipped when --reasoning-effort isn't passed.
    if reasoning_effort is not None:
        info = _model_details_for(model_name, provider_id=provider_id)
        if info is None:
            console.print(
                "[dim]No /models/details available — sending "
                f"--reasoning-effort={reasoning_effort} anyway; server "
                "will validate.[/dim]"
            )
        elif not info.get("supports_reasoning_effort"):
            console.print(
                f"[yellow]Warning:[/yellow] model [bold]{model_name}[/bold] "
                "does not advertise reasoning_effort. Clearing the value; "
                "use [bold]\\reasoning[/bold] after switching models."
            )
            reasoning_effort = None
        else:
            accepted = info.get("reasoning_efforts") or []
            if reasoning_effort not in accepted:
                console.print(
                    f"[yellow]Warning:[/yellow] '{reasoning_effort}' is not "
                    f"in this model's accepted efforts: {accepted}. "
                    "Server will return 422 unless you change it via "
                    "[bold]\\reasoning[/bold]."
                )

    # --- Session Management Loop ---
    session_id = None
    while session_id is None:
        selected_session_id, action = manage_sessions()

        if action == "new":
            session_id = create_session()
            if not session_id:
                # Creation failed, loop back to menu
                continue
        elif action == "resume":
            if selected_session_id:
                session_id = selected_session_id
                console.print(f"✅ Resuming session: [yellow]{session_id}[/yellow]")
                history = get_session_history(session_id)
                display_history(history)
            else:
                console.print("[red]Error: Tried to resume a session without an ID.[/red]")
                continue
        elif action == "manage":
            continue # Loop back to the management screen
        elif action == "quit":
            raise typer.Exit()

    console.print(f"🤖 Starting chat with [bold green]{model_name}[/bold green]...")

    info_table = Table.grid(padding=1, expand=True)
    info_table.add_column()
    info_table.add_column(justify="right")
    info_table.add_row(
        f"Debug mode: {'[bold green]enabled[/bold green]' if debug else '[dim]disabled[/dim]'}",
        "Type [bold cyan]\\menu[/bold cyan] for session management"
    )
    info_table.add_row(
        f"Show thinking: {'[bold green]enabled[/bold green]' if show_thinking else '[dim]disabled[/dim]'}",
        "Type [bold cyan]\\thinking[/bold cyan] to toggle thinking"
    )
    info_table.add_row(
        "",
        "Type [bold cyan]\\tools[/bold cyan] to list available tools"
    )
    info_table.add_row(
        "",
        "Type [bold cyan]\\models[/bold cyan] to switch models mid-chat"
    )
    info_table.add_row(
        "",
        "Type [bold cyan]\\reasoning[/bold cyan] to change reasoning effort"
    )
    info_table.add_row(
        "",
        "Type [bold cyan]\\providers[/bold cyan] to list providers"
    )
    info_table.add_row(
        "",
        "Type [bold cyan]\\exit[/bold cyan] or [bold cyan]\\quit[/bold cyan] to end"
    )
    if reasoning_effort is not None:
        info_table.add_row(
            f"Reasoning effort: [bold green]{reasoning_effort}[/bold green]",
            "",
        )
    provider_display = (
        f"[bold green]{provider_id}[/bold green]" if provider_id else "[dim]default[/dim]"
    )
    info_table.add_row(
        f"Provider: {provider_display}",
        "",
    )
    console.print(Panel(info_table, title="Chat Info", border_style="dim"))


    # --- Main chat loop ---
    while True:
        try:
            prompt_message = [
                ('bold cyan', 'You '),
                ('', '(Alt+Enter for newline)\n')
            ]
            user_prompt = ptk_prompt(FormattedText(prompt_message), multiline=True)

            stripped_prompt = user_prompt.strip().lower()
            if stripped_prompt in ["\\exit", "\\quit"]:
                console.print("👋 Goodbye!")
                break
            if stripped_prompt == "\\menu":
                # We need to pass the original arguments to main to restart it correctly
                main(
                    model_name=model_name,
                    api_url=API_BASE_URL,
                    debug=debug,
                    show_thinking=show_thinking,
                    reasoning_effort=reasoning_effort,
                    provider=provider_id,
                )
                break # Exit current chat loop to prevent it from continuing after menu
            if stripped_prompt == "\\thinking":
                show_thinking = not show_thinking
                thinking_status = "[bold green]enabled[/bold green]" if show_thinking else "[dim]disabled[/dim]"
                console.print(f"Show thinking is now {thinking_status}.")
                console.rule()
                continue # Go to next prompt
            if stripped_prompt == "\\tools":
                tools_info = get_available_tools()
                if not tools_info:
                    console.print("[yellow]No tools available (registry empty or server unreachable).[/yellow]")
                else:
                    tools_table = Table(title=f"Available tools ({len(tools_info)})", border_style="cyan")
                    tools_table.add_column("Name", style="bold cyan")
                    tools_table.add_column("Description")
                    tools_table.add_column("Params", style="dim", justify="right")
                    for tool in tools_info:
                        name = tool.get("name", "?")
                        desc = (tool.get("description") or "(no description)").strip().splitlines()[0]
                        params = tool.get("parameters", {})
                        prop_names = list((params.get("properties") or {}).keys())
                        param_summary = ", ".join(prop_names) if prop_names else "—"
                        tools_table.add_row(name, desc, param_summary)
                    console.print(tools_table)
                console.rule()
                continue
            if stripped_prompt == "\\models":
                new_model, new_pid = select_model(None)
                # ADR-0021 follow-up: switch when EITHER model_name or
                # provider_id changed. The previous gating only checked
                # model_name, so picking the same-name model from a
                # different provider silently dropped the new provider_id.
                if new_model and (new_model != model_name or new_pid != provider_id):
                    if new_model != model_name:
                        console.print(
                            f"🔄 Switching from [yellow]{model_name}[/yellow] "
                            f"to [bold green]{new_model}[/bold green]"
                        )
                    if new_pid != provider_id:
                        from_label = provider_id or "default"
                        to_label = new_pid or "default"
                        console.print(
                            f"🔀 Provider: [yellow]{from_label}[/yellow] → "
                            f"[bold green]{to_label}[/bold green]"
                        )
                    model_name = new_model
                    provider_id = new_pid
                    # Clear stale reasoning_effort if the new (provider, model)
                    # pair doesn't accept the current value. Rubber-duck
                    # follow-up: a left-over high-effort hint would
                    # 422 the next turn and force a reset.
                    if reasoning_effort is not None:
                        new_info = _model_details_for(new_model, provider_id=new_pid)
                        accepted = (new_info or {}).get("reasoning_efforts") or []
                        supports = (new_info or {}).get("supports_reasoning_effort", False)
                        if not supports or reasoning_effort not in accepted:
                            console.print(
                                f"[yellow]Cleared reasoning effort '{reasoning_effort}' — "
                                f"not supported by {new_model}. Use "
                                "[bold]\\reasoning[/bold] to set a new value.[/yellow]"
                            )
                            reasoning_effort = None
                else:
                    console.print(f"[dim]Keeping current model: {model_name}[/dim]")
                console.rule()
                continue
            if stripped_prompt == "\\reasoning":
                new_effort = select_reasoning_effort(
                    model_name,
                    current=reasoning_effort,
                    provider_id=provider_id,
                )
                if new_effort != reasoning_effort:
                    if new_effort is None:
                        console.print(
                            "[dim]Reasoning effort cleared.[/dim]"
                        )
                    else:
                        console.print(
                            f"🧠 Reasoning effort set to "
                            f"[bold green]{new_effort}[/bold green]"
                        )
                    reasoning_effort = new_effort
                console.rule()
                continue
            if stripped_prompt == "\\providers":
                table = get_providers_table()
                if table is None:
                    console.print(
                        "[yellow]Server does not expose multi-provider metadata; "
                        "using legacy single-provider routing.[/yellow]"
                    )
                else:
                    console.print(table)
                console.rule()
                continue

            # --- Call the streaming generate endpoint and process events ---
            # Accept header opts into v2 NDJSON vocabulary (p5-cutover-b).
            # Default endpoint still emits v0; version=1.0 routes the request
            # through transport_ndjson. Synthesis §11.3 R18; §4 Phase 5 step 54.
            _post_body = {
                "session_id": session_id,
                "prompt": user_prompt,
                "model_name": model_name,
            }
            if reasoning_effort is not None:
                _post_body["reasoning_effort"] = reasoning_effort
            if provider_id is not None:
                _post_body["provider_id"] = provider_id
            with requests.post(
                f"{API_BASE_URL}/chat/stream",
                json=_post_body,
                headers=_mutating_headers({
                    "Accept": "application/x-ndjson; version=1.0",
                }),
                stream=True,
            ) as response:
                response.raise_for_status()

                assistant_response = ""
                thinking_response = ""
                text_started = False
                thinking_panel_active = False
                spinner_active = True

                # Start spinner while waiting for first response
                with Live(
                    Spinner("dots", text="[dim]Waiting for response...[/dim]"),
                    console=console,
                    refresh_per_second=10,
                ) as live:
                    for line in response.iter_lines():
                        # Stop spinner on first event
                        if spinner_active:
                            live.stop()
                            spinner_active = False

                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            if debug:
                                decoded_line = line.decode("utf-8", errors="replace")
                                console.print(f"[red]Error parsing JSON: {decoded_line}[/red]")
                            continue

                        # v2 vocabulary — top-level fields only (no data wrapper).
                        # Synthesis §4 Phase 5 step 54; §11.3 R18 (opt-in via Accept).
                        evt_type = event.get("type")

                        if debug:
                            console.print(f"[dim]Received event: {event}[/dim]")

                        if evt_type == "message_start":
                            # Header event: turn is opening. No visible UI action.
                            if debug:
                                console.print(f"[dim]Stream started: turn_id={event.get('turn_id')}[/dim]")

                        elif evt_type == "text_delta":
                            delta = event.get("text", "")
                            assistant_response += delta

                            if not text_started:
                                console.print("\n[bold green]Assistant:[/bold green]")
                                text_started = True

                            console.print(delta, end="", style="green")

                        elif evt_type == "thinking_delta":
                            if show_thinking:
                                delta = event.get("text", "")
                                thinking_response += delta

                                if not thinking_panel_active:
                                    console.print("\n[dim italic]💭 Thinking:[/dim italic]")
                                    thinking_panel_active = True

                                console.print(delta, end="", style="dim italic")

                        elif evt_type == "tool_call":
                            if text_started or thinking_panel_active:
                                console.print()
                            tool_name = event.get("name")
                            console.print(
                                Panel(
                                    f"Calling tool: [bold yellow]{tool_name}[/bold yellow]",
                                    expand=False,
                                    border_style="yellow",
                                )
                            )

                        elif evt_type == "tool_result":
                            tool_name = event.get("name")
                            status = event.get("status", "ok")
                            if status == "ok":
                                result = event.get("result", {})
                                output_str = str(result)
                                console.print(
                                    Panel(
                                        f"Tool [bold yellow]{tool_name}[/bold yellow] "
                                        f"output: {output_str[:150]}...",
                                        title="Tool Output",
                                        expand=False,
                                        border_style="dim yellow",
                                    )
                                )
                            else:
                                error = event.get("error", "unknown")
                                error_kind = event.get("error_kind", "")
                                kind_str = f" ({error_kind})" if error_kind else ""
                                if text_started or thinking_panel_active:
                                    console.print()
                                console.print(
                                    Panel(
                                        f"Tool [bold red]{tool_name}[/bold red] "
                                        f"error{kind_str}: {error}",
                                        title="Tool Error",
                                        border_style="red",
                                    )
                                )

                        elif evt_type == "error":
                            if text_started or thinking_panel_active:
                                console.print()
                            message = event.get("message", "")
                            console.print(Panel(f"API Error: {message}", title="Error", border_style="bold red"))

                        elif evt_type == "loop_limit_reached":
                            loops = event.get("loops", "?")
                            console.print(Panel(f"Tool loop limit reached ({loops} iterations)", border_style="yellow"))

                        elif evt_type == "hw_reset":
                            model = event.get("model_name", "")
                            console.print(Panel(f"Hardware reset: model '{model}' was reset", border_style="dim"))

                        elif evt_type == "message_stop":
                            stop_reason = event.get("stop_reason", "complete")
                            if text_started or thinking_panel_active:
                                console.print()
                            if debug or stop_reason != "complete":
                                label = stop_reason.replace("_", " ")
                                console.print(f"[dim][Stream ended: {label}][/dim]")

        except requests.RequestException as e:
            console.print(f"\n[bold red]Error:[/bold red] Could not get response from server. {e}")
            continue
        except Exception as e:
            console.print(f"\n[bold red]Unexpected error:[/bold red] {str(e)}")
            continue
        finally:
            console.rule()


if __name__ == "__main__":
    app()

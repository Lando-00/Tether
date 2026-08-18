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
from enum import Enum
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

# Sentinel from ModelDetails.provider_id for pre-registry single-provider servers.
_PROVIDER_ID_SENTINEL = "_unwrapped_"


class ChatMode(str, Enum):
    """Available server-side orchestrator modes."""

    chat = "chat"
    research = "research"


def _chat_payload(
    *,
    session_id: str,
    prompt: str,
    model_name: str,
    mode: ChatMode,
    provider_id: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> dict[str, str]:
    """Build the `/chat/stream` request body."""
    body: dict[str, str] = {
        "session_id": session_id,
        "prompt": prompt,
        "model_name": model_name,
        "mode": mode.value,
    }
    if provider_id is not None:
        body["provider_id"] = provider_id
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    return body


def _parse_chat_mode(value: str) -> ChatMode:
    """Parse a chat mode entered interactively."""
    normalized = value.strip().lower()
    for mode in ChatMode:
        if normalized == mode.value:
            return mode
    allowed = ", ".join(mode.value for mode in ChatMode)
    raise ValueError(f"mode must be one of: {allowed}")


def _mode_label(mode: ChatMode) -> str:
    if mode is ChatMode.research:
        return "[bold magenta]research[/bold magenta]"
    return "[bold green]chat[/bold green]"


def _mode_hint(mode: ChatMode) -> str:
    if mode is ChatMode.research:
        return "Research mode uses web_search + BRAVE_API_KEY"
    return "Chat mode uses the normal tool-calling orchestrator"


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
        health = item.get("health")
        if not isinstance(health, dict):
            health = {}
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


def get_available_model_details() -> list:
    """Fetch ``GET /models/details`` and return the list of ModelDetails dicts."""
    try:
        response = requests.get(f"{API_BASE_URL}/models/details", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return []


def _reasoning_efforts_for_model(
    model_name: str,
    provider_id: Optional[str],
) -> Optional[list[str]]:
    """Return accepted reasoning efforts, or ``None`` when metadata is ambiguous."""
    matching = [
        detail
        for detail in get_available_model_details()
        if detail.get("id") == model_name
        and (
            provider_id is None
            or detail.get("provider_id") in (_PROVIDER_ID_SENTINEL, provider_id)
        )
    ]
    if provider_id is None and len(matching) != 1:
        return None
    if provider_id is not None and len(matching) != 1:
        return None
    if not matching or not matching[0].get("supports_reasoning_effort", False):
        return []
    return list(matching[0].get("reasoning_efforts") or [])


def get_provider_health() -> tuple[Optional[dict], Optional[str]]:
    """Return ({pid: {healthy, kind, source, error}}, default_provider_id)
    from /readyz, or (None, None) on connection error.
    """
    try:
        resp = requests.get(f"{API_BASE_URL}/readyz", timeout=5)
        data = resp.json()
        providers = data.get("providers")
        default_pid = data.get("default_provider_id")
        return providers, default_pid
    except Exception:
        return None, None


def get_providers_table() -> Optional[Table]:
    """Build the Rich Table for the ``\\providers`` slash command.

    Returns ``None`` when the server lacks the multi-provider block (older
    build or single-provider legacy config).
    """
    providers, default_pid = get_provider_health()
    if providers is None:
        return None

    table = Table(title="Providers", border_style="cyan")
    table.add_column("ID", style="bold cyan")
    table.add_column("Kind")
    table.add_column("Source")
    table.add_column("Default", justify="center")
    table.add_column("Health")
    table.add_column("Error", style="dim")

    for pid, info in sorted(providers.items()):
        is_default = "★" if pid == default_pid else ""
        healthy = info.get("healthy", False)
        health_str = "[green]healthy[/green]" if healthy else "[red]unhealthy[/red]"
        error = info.get("error") or ""
        table.add_row(
            pid,
            info.get("kind", "?"),
            info.get("source", "?"),
            is_default,
            health_str,
            error,
        )
    return table


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
            session_id = s.get("session_id", "N/A")
            created_at = s.get("created_at", "N/A")
            table.add_row(str(i + 1), f"Resume session from {created_at} ([yellow]{session_id[:8]}...[/yellow])")
            choices[str(i + 1)] = f"Resume session {session_id}"
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
            default="n",
        )
        if confirm.lower() == "y":
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
        session_to_delete = sessions[del_choice - 1]["session_id"]
        delete_session(session_to_delete)
        return None, "manage"
    elif action.isdigit() and sessions and 0 < int(action) <= len(sessions):
        session_id = sessions[int(action) - 1]["session_id"]
        return session_id, "resume"
    elif action == "q":
        return None, "quit"
    else:
        console.print("[red]Invalid choice.[/red]")
        return None, "manage"


def select_model(
    model_name: Optional[str],
    provider: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """Validate or interactively select a model; return ``(model_id, provider_id)``.

    When *model_name* is given:
    - When *provider* is supplied, the model must be advertised by that
      provider; no other provider may be selected implicitly.
    - Exactly one ``/models/details`` row matches → return its id and provider_id.
    - Multiple rows share the same id (ambiguous; multi-provider):
      - *provider* is supplied → filter to the matching row.
      - *provider* is absent → drop into the interactive selector.
    - No details row matches → fall back to the legacy ``/models`` plain list.

    When *model_name* is ``None``, open the interactive selector.
    Returns ``(model_id, provider_id_or_None)``.
    """
    details = get_available_model_details()
    if model_name and details:
        filtered = [d for d in details if d.get("id") == model_name]
        if provider is not None:
            selected = [detail for detail in filtered if detail.get("provider_id") == provider]
            if len(selected) == 1:
                return model_name, provider
            console.print(
                f"[bold red]Error:[/bold red] Model '{model_name}' is not available on provider '{provider}'."
            )
            raise typer.Exit(1)
        if len(filtered) == 1:
            pid = filtered[0].get("provider_id")
            if pid == _PROVIDER_ID_SENTINEL:
                pid = None
            return model_name, pid
        if len(filtered) > 1:
            console.print(
                f"[yellow]Model '{model_name}' is ambiguous — available on "
                f"{len(filtered)} providers. Pick one:[/yellow]"
            )
            return _interactive_model_select(details=filtered)
    if model_name:
        if provider is not None:
            console.print(
                "[bold red]Error:[/bold red] Cannot validate "
                f"model '{model_name}' for provider '{provider}' because "
                "/models/details is unavailable."
            )
            raise typer.Exit(1)
        models = get_available_models()
        if model_name in models:
            return model_name, None
        console.print(f"[bold red]Error:[/bold red] Model '{model_name}' not found.")
        raise typer.Exit(1)

    # Interactive selection over all models
    if details:
        if provider is not None:
            details = [detail for detail in details if detail.get("provider_id") == provider]
            if not details:
                console.print(f"[bold red]Error:[/bold red] Provider '{provider}' has no available models.")
                health, _ = get_provider_health()
                error = ((health or {}).get(provider) or {}).get("error")
                if error:
                    console.print(f"[dim]{provider}: {error}[/dim]")
                raise typer.Exit(1)
        return _interactive_model_select(details=details)

    console.print("🔍 Searching for available models...")
    available_models = get_available_models()
    if not available_models:
        console.print("[bold red]Error:[/bold red] No models found.")
        # Name each registered provider and why it offered nothing, instead
        # of a generic "check your models directory" that is wrong for every
        # provider that isn't MLC.
        gaps = _provider_gaps([])
        if gaps:
            console.print("Registered providers and why they offered no models:")
            _render_provider_gaps([])
        else:
            console.print(
                "Please make sure your models are correctly configured and the service is running."
            )
        raise typer.Exit(1)

    console.print("\nAvailable Models:")
    for i, name in enumerate(available_models):
        console.print(f"  [bold cyan][{i + 1}][/bold cyan] {name}")

    while True:
        try:
            choice_str = Prompt.ask("\nPlease enter the number of the model you want to use", default="1")
            if not choice_str.strip():
                choice_str = "1"
            choice = int(choice_str)
            if 1 <= choice <= len(available_models):
                return available_models[choice - 1], None
            else:
                console.print(
                    f"[red]Invalid choice. Please enter a number between 1 and {len(available_models)}.[/red]"
                )
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/red]")


def _provider_gaps(details: list[dict]) -> list[tuple[str, Optional[str]]]:
    """Registered providers that contributed no models, with any error.

    A provider can be registered and reachable yet still return an empty
    catalog (nothing pulled, wheels missing, stub). Without this the picker
    would simply not mention it, which reads as "that provider does not
    exist" rather than "that provider has nothing to offer".
    """
    health, _ = get_provider_health()
    if not health:
        return []
    represented = {
        row.get("provider_id")
        for row in details
        if row.get("provider_id") and row.get("provider_id") != _PROVIDER_ID_SENTINEL
    }
    return [
        (pid, (info or {}).get("error"))
        for pid, info in health.items()
        if pid not in represented
    ]


def _render_provider_gaps(details: list[dict]) -> None:
    """Print one dim line per provider that offers no models."""
    for pid, error in _provider_gaps(details):
        reason = f" — {error}" if error else " — no models available"
        console.print(f"[dim]![/dim] [yellow]{pid}[/yellow][dim]{reason}[/dim]")


def _interactive_model_select(
    *,
    details: list[dict],
) -> tuple[str, Optional[str]]:
    """Render a numbered model selector from /models/details rows."""
    _, default_pid = get_provider_health()

    show_provider_col = any(r.get("provider_id") and r.get("provider_id") != _PROVIDER_ID_SENTINEL for r in details)

    table = Table(title="Models", border_style="cyan")
    table.add_column("#", style="bold cyan", justify="right")
    table.add_column("Model")
    if show_provider_col:
        table.add_column("Provider")
    table.add_column("Source", style="dim")
    table.add_column("Context", justify="right")

    sorted_details = sorted(details, key=lambda d: (d.get("provider_id", ""), d.get("id", "")))
    default_choice = next(
        (
            index
            for index, info in enumerate(sorted_details, 1)
            if (
                info.get("provider_id") == default_pid
                and info.get("is_default", False)
            )
        ),
        next(
            (
                index
                for index, info in enumerate(sorted_details, 1)
                if info.get("provider_id") == default_pid
            ),
            1,
        ),
    )
    previous_pid: Optional[str] = None
    for i, info in enumerate(sorted_details, 1):
        pid = info.get("provider_id", "")
        is_default = info.get("is_default", False)
        marker = " ★" if is_default else ""
        row = [str(i), info.get("id", "?") + marker]
        if show_provider_col:
            # Only label the first row of each provider block so the table
            # reads as grouped sections instead of a repeated column.
            row.append(
                (pid if pid != _PROVIDER_ID_SENTINEL else "—")
                if pid != previous_pid
                else ""
            )
        row.append(info.get("source", "?"))
        row.append(str(info.get("context_window", "?")))
        table.add_row(*row, end_section=(pid != previous_pid and i > 1))
        previous_pid = pid
    console.print(table)
    _render_provider_gaps(details)

    while True:
        choice_str = Prompt.ask(
            "Select model #",
            default=str(default_choice),
        )
        try:
            choice = int(choice_str.strip() or "1")
            if 1 <= choice <= len(sorted_details):
                sel = sorted_details[choice - 1]
                sel_id = sel.get("id", "?")
                sel_pid = sel.get("provider_id")
                if sel_pid == _PROVIDER_ID_SENTINEL:
                    sel_pid = None
                return sel_id, sel_pid
            console.print(f"[red]Pick 1–{len(sorted_details)}.[/red]")
        except ValueError:
            console.print("[red]Invalid input.[/red]")


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
    mode: ChatMode = typer.Option(
        ChatMode.chat,
        "--mode",
        help="Orchestrator mode to use: chat or research.",
        case_sensitive=False,
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
            "Provider id to route the request to. The selected model must "
            "belong to this provider. Omit only to use the server's unique "
            "model-name routing. Use `\\providers` in the REPL to see ids "
            "and health."
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
            mode=mode,
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
    mode: ChatMode = typer.Option(
        ChatMode.chat,
        "--mode",
        help="Orchestrator mode to use: chat or research.",
        case_sensitive=False,
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
            "Provider id to route the request to. The selected model must "
            "belong to this provider. Omit only to use the server's unique "
            "model-name routing. Use `\\providers` in the REPL to see ids "
            "and health."
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

    console.print(
        Panel.fit(
            f"[bold blue]Welcome to the Tether CLI![/bold blue]\n[dim]API: {API_BASE_URL}[/dim]", style="bold blue"
        )
    )

    model_name_arg = model_name
    model_name, provider_id = select_model(model_name_arg, provider)
    # Keep the user's command-line choice distinct from the provider inferred
    # for a uniquely owned model. The latter routes the current request, but
    # must not prevent `\models` from offering other providers later.
    provider_constraint = provider

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
            continue  # Loop back to the management screen
        elif action == "quit":
            raise typer.Exit()

    current_mode = mode
    console.print(f"🤖 Starting chat with [bold green]{model_name}[/bold green] in {_mode_label(current_mode)} mode...")

    info_table = Table.grid(padding=1, expand=True)
    info_table.add_column()
    info_table.add_column(justify="right")
    info_table.add_row(
        f"Debug mode: {'[bold green]enabled[/bold green]' if debug else '[dim]disabled[/dim]'}",
        "Type [bold cyan]\\menu[/bold cyan] for session management",
    )
    info_table.add_row(
        f"Show thinking: {'[bold green]enabled[/bold green]' if show_thinking else '[dim]disabled[/dim]'}",
        "Type [bold cyan]\\thinking[/bold cyan] to toggle thinking",
    )
    info_table.add_row(
        f"Mode: {_mode_label(current_mode)}", "Type [bold cyan]\\mode[/bold cyan] to switch chat/research"
    )
    info_table.add_row("", "Type [bold cyan]\\tools[/bold cyan] to list available tools")
    info_table.add_row("", "Type [bold cyan]\\models[/bold cyan] to switch models mid-chat")
    info_table.add_row(
        f"Provider: {provider_id or 'default'}", "Type [bold cyan]\\providers[/bold cyan] to list providers"
    )
    info_table.add_row(
        f"Reasoning effort: {reasoning_effort or 'provider default'}",
        "Type [bold cyan]\\reasoning[/bold cyan] to change",
    )
    info_table.add_row("", "Type [bold cyan]\\exit[/bold cyan] or [bold cyan]\\quit[/bold cyan] to end")
    console.print(Panel(info_table, title="Chat Info", border_style="dim"))

    # --- Main chat loop ---
    while True:
        try:
            prompt_message = [("bold cyan", "You "), ("", "(Alt+Enter for newline)\n")]
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
                    mode=current_mode,
                    reasoning_effort=reasoning_effort,
                    provider=provider_constraint,
                )
                break  # Exit current chat loop to prevent it from continuing after menu
            if stripped_prompt == "\\thinking":
                show_thinking = not show_thinking
                thinking_status = "[bold green]enabled[/bold green]" if show_thinking else "[dim]disabled[/dim]"
                console.print(f"Show thinking is now {thinking_status}.")
                console.rule()
                continue  # Go to next prompt
            if stripped_prompt == "\\reasoning" or stripped_prompt.startswith("\\reasoning "):
                parts = user_prompt.strip().split(maxsplit=1)
                selected_from_command = parts[1].strip() if len(parts) == 2 else None
                if selected_from_command and selected_from_command.lower() in {
                    "default",
                    "none",
                    "off",
                }:
                    reasoning_effort = None
                    console.print("[green]Reasoning effort reset to the provider default.[/green]")
                else:
                    options = _reasoning_efforts_for_model(model_name, provider_id)
                    if options is None:
                        console.print(
                            "[yellow]Reasoning-effort metadata is unavailable or "
                            "ambiguous for the current model.[/yellow]"
                        )
                    elif not options:
                        console.print(
                            f"[yellow]Model '{model_name}' does not support reasoning effort.[/yellow]"
                        )
                    else:
                        selected = selected_from_command or str(
                            Prompt.ask(
                                "Reasoning effort",
                                choices=[*options, "default"],
                                default=reasoning_effort or "default",
                            )
                        )
                        normalized = selected.lower()
                        if normalized in {"default", "none", "off"}:
                            reasoning_effort = None
                            console.print("[green]Reasoning effort reset to the provider default.[/green]")
                        else:
                            option_by_lower = {
                                option.lower(): option for option in options
                            }
                            if normalized not in option_by_lower:
                                console.print(
                                    f"[red]Choose one of: {', '.join(options)}, default.[/red]"
                                )
                            else:
                                reasoning_effort = option_by_lower[normalized]
                                console.print(
                                    f"[green]Reasoning effort set to {reasoning_effort}.[/green]"
                                )
                console.rule()
                continue
            if stripped_prompt in {"\\chat", "\\research", "\\mode"} or stripped_prompt.startswith("\\mode "):
                try:
                    if stripped_prompt == "\\chat":
                        new_mode = ChatMode.chat
                    elif stripped_prompt == "\\research":
                        new_mode = ChatMode.research
                    else:
                        parts = stripped_prompt.split(maxsplit=1)
                        new_mode = (
                            ChatMode.research
                            if len(parts) == 1 and current_mode is ChatMode.chat
                            else ChatMode.chat
                            if len(parts) == 1
                            else _parse_chat_mode(parts[1])
                        )
                except ValueError as exc:
                    console.print(f"[bold red]Error:[/bold red] {exc}")
                    console.rule()
                    continue
                if new_mode is current_mode:
                    console.print(f"[dim]Already in {_mode_label(current_mode)} mode.[/dim]")
                else:
                    current_mode = new_mode
                    console.print(f"Mode switched to {_mode_label(current_mode)}.")
                    console.print(f"[dim]{_mode_hint(current_mode)}[/dim]")
                console.rule()
                continue
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
                new_model, new_pid = select_model(None, provider=provider_constraint)
                if new_model and (new_model != model_name or new_pid != provider_id):
                    console.print(
                        f"🔄 Switching from [yellow]{model_name}[/yellow] to [bold green]{new_model}[/bold green]"
                    )
                    model_name = new_model
                    provider_id = new_pid
                    new_efforts = _reasoning_efforts_for_model(model_name, provider_id)
                    if (
                        reasoning_effort is not None
                        and new_efforts is not None
                        and reasoning_effort not in new_efforts
                    ):
                        reasoning_effort = None
                        console.print(
                            "[yellow]Reasoning effort reset: the selected model does not accept it.[/yellow]"
                        )
                else:
                    console.print(f"[dim]Keeping current model: {model_name}[/dim]")
                console.rule()
                continue
            if stripped_prompt == "\\providers":
                table = get_providers_table()
                if table is None:
                    console.print(
                        "[yellow]Server does not expose multi-provider metadata; single-provider mode.[/yellow]"
                    )
                else:
                    console.print(table)
                console.rule()
                continue

            # --- Call the streaming generate endpoint and process events ---
            # Accept header opts into v2 NDJSON vocabulary (p5-cutover-b).
            # Default endpoint still emits v0; version=1.0 routes the request
            # through transport_ndjson. Synthesis §11.3 R18; §4 Phase 5 step 54.
            with requests.post(
                f"{API_BASE_URL}/chat/stream",
                json=_chat_payload(
                    session_id=session_id,
                    prompt=user_prompt,
                    model_name=model_name,
                    mode=current_mode,
                    provider_id=provider_id,
                    reasoning_effort=reasoning_effort,
                ),
                headers=_mutating_headers(
                    {
                        "Accept": "application/x-ndjson; version=1.0",
                    }
                ),
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
                                        f"Tool [bold yellow]{tool_name}[/bold yellow] output: {output_str[:150]}...",
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
                                        f"Tool [bold red]{tool_name}[/bold red] error{kind_str}: {error}",
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

                        elif evt_type == "notebook_phase_start":
                            phase = event.get("phase", "?")
                            iteration = event.get("iteration", 0)
                            iter_suffix = f" #{iteration}" if iteration else ""
                            console.print(f"[dim]Research: {phase}{iter_suffix}[/dim]")

                        elif evt_type == "notebook_phase_progress":
                            phase = event.get("phase", "?")
                            elapsed_ms = int(event.get("elapsed_ms") or 0)
                            console.print(f"[dim]Research: {phase} still running ({elapsed_ms / 1000:.0f}s)...[/dim]")

                        elif evt_type == "notebook_fact_added":
                            if debug:
                                fact = str(event.get("fact_text", ""))
                                total = event.get("total_facts", "?")
                                source_kind = event.get("source_kind", "web_search")
                                console.print(
                                    f"[dim]Notebook fact {total} ({source_kind}): {fact[:120]}[/dim]"
                                )

                        elif evt_type == "notebook_clarification_requested":
                            message = str(
                                event.get(
                                    "message",
                                    "Please clarify your question.",
                                )
                            )
                            candidates = event.get("candidates", [])
                            candidate_text = "\n".join(
                                f"• {item}"
                                for item in candidates
                                if isinstance(item, str)
                            )
                            content = message
                            if candidate_text:
                                content += f"\n\nCandidates:\n{candidate_text}"
                            console.print(
                                Panel(
                                    content,
                                    title="Research clarification",
                                    border_style="yellow",
                                )
                            )

                        elif evt_type == "notebook_query_added":
                            if debug:
                                query = str(event.get("query", ""))
                                depth = event.get("queue_depth", "?")
                                console.print(f"[dim]Research query queued ({depth}): {query}[/dim]")

                        elif evt_type == "notebook_limit_reached":
                            kind = event.get("limit_kind", "limit")
                            count = event.get("count", "?")
                            console.print(
                                Panel(
                                    f"Research stopped at {kind}={count}; synthesizing partial notebook.",
                                    border_style="yellow",
                                )
                            )

                        elif evt_type == "notebook_no_facts":
                            queries = event.get("queries_attempted", 0)
                            iterations = event.get("iterations", 0)
                            console.print(
                                Panel(
                                    "Research gathered no facts "
                                    f"({queries} queries, {iterations} iterations); synthesizing anyway.",
                                    border_style="yellow",
                                )
                            )

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

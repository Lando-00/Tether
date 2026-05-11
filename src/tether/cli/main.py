"""
A modern CLI for interacting with the Tether service.
"""
import json
from pathlib import Path
from typing import Optional

import requests
import typer
from rich.console import Console
from rich.prompt import Prompt, IntPrompt
from prompt_toolkit import prompt as ptk_prompt
from prompt_toolkit.formatted_text import FormattedText
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.spinner import Spinner
from rich.live import Live

# --- Configuration ---
# Default API base URL. Reassigned by main() if --api-url is passed.
# Reads from TETHER_API_URL env var if set (allows shell-level override
# without a flag, useful for development).
import os as _os
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
    """Return headers for state-changing requests, injecting CSRF if known."""
    headers: dict = dict(extra) if extra else {}
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
        console.print("[yellow]Warning:[/yellow] No available models found to specify for unload request. The cache might be empty already.")
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
                        console.print(Panel(panel_content, title="Assistant (Tool Call)", title_align="left", border_style="green"))
            else:
                panel_content = Text(content, style="green")
                console.print(Panel(panel_content, title="Assistant", title_align="left", border_style="green"))
        elif role == "tool":
            tool_name = msg.get("tool_name", "unknown_tool")
            tool_content = msg.get("content", "")
            panel_content = Text(str(tool_content), style="yellow")
            console.print(Panel(panel_content, title=f"Tool Output ({tool_name})", title_align="left", border_style="yellow"))
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


def select_model(model_name: Optional[str]) -> str:
    """Guides the user to select a model if one isn't provided."""
    if model_name:
        # The new API just returns a list of strings, so we just check for existence
        models = get_available_models()
        if model_name in models:
            return model_name
        console.print(f"[bold red]Error:[/bold red] Model '{model_name}' not found.")
        raise typer.Exit(1)

    console.print("🔍 Searching for available models...")
    available_models = get_available_models()
    if not available_models:
        console.print("[bold red]Error:[/bold red] No models found.")
        console.print("Please make sure your compiled models are correctly placed and the service is running.")
        raise typer.Exit(1)

    console.print("\nAvailable Models:")
    for i, name in enumerate(available_models):
        console.print(f"  [bold cyan][{i+1}][/bold cyan] {name}")

    while True:
        try:
            choice_str = Prompt.ask(
                "\nPlease enter the number of the model you want to use",
                default="1"
            )
            if not choice_str.strip():
                choice_str = "1"
            choice = int(choice_str)
            if 1 <= choice <= len(available_models):
                return available_models[choice - 1]
            else:
                console.print(f"[red]Invalid choice. Please enter a number between 1 and {len(available_models)}.[/red]")
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/red]")


@app.command()
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
    model_name = select_model(model_name_arg)

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
        "Type [bold cyan]\\exit[/bold cyan] or [bold cyan]\\quit[/bold cyan] to end"
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
                main(model_name=model_name, api_url=API_BASE_URL, debug=debug, show_thinking=show_thinking)
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
                new_model = select_model(None)
                if new_model and new_model != model_name:
                    console.print(f"🔄 Switching from [yellow]{model_name}[/yellow] to [bold green]{new_model}[/bold green]")
                    model_name = new_model
                else:
                    console.print(f"[dim]Keeping current model: {model_name}[/dim]")
                console.rule()
                continue

            # --- Call the streaming generate endpoint and process events ---
            # Accept header opts into v2 NDJSON vocabulary (p5-cutover-b).
            # Default endpoint still emits v0; version=1.0 routes the request
            # through transport_ndjson. Synthesis §11.3 R18; §4 Phase 5 step 54.
            with requests.post(
                f"{API_BASE_URL}/chat/stream",
                json={
                    "session_id": session_id,
                    "prompt": user_prompt,
                    "model_name": model_name,
                },
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
                with Live(Spinner("dots", text="[dim]Waiting for response...[/dim]"), console=console, refresh_per_second=10) as live:
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
                                console.print(f"[red]Error parsing JSON: {line.decode('utf-8', errors='replace')}[/red]")
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
                            console.print(Panel(f"Calling tool: [bold yellow]{tool_name}[/bold yellow]", expand=False, border_style="yellow"))

                        elif evt_type == "tool_result":
                            tool_name = event.get("name")
                            status = event.get("status", "ok")
                            if status == "ok":
                                result = event.get("result", {})
                                output_str = str(result)
                                console.print(Panel(f"Tool [bold yellow]{tool_name}[/bold yellow] output: {output_str[:150]}...", title="Tool Output", expand=False, border_style="dim yellow"))
                            else:
                                error = event.get("error", "unknown")
                                error_kind = event.get("error_kind", "")
                                kind_str = f" ({error_kind})" if error_kind else ""
                                if text_started or thinking_panel_active:
                                    console.print()
                                console.print(Panel(f"Tool [bold red]{tool_name}[/bold red] error{kind_str}: {error}", title="Tool Error", border_style="red"))

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

# Built-in Tools

Auto-generated from `tether.tools` via `scripts/docs/generate.py`. Do not edit by hand.

## `forecast`

Get a weather forecast for a location.

### Parameters

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `days` | `integer` | no | `3` | Number of days to forecast (1-16). |
| `location` | `string` | yes | — | The city name. |
| `unit` | `enum("celsius" \| "fahrenheit")` | no | `"celsius"` | The temperature unit (celsius or fahrenheit). |

## `time`

Get the current time for a timezone in various formats (ISO, RFC2822, or human-readable).

### Parameters

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `format` | `enum("iso" \| "rfc2822" \| "human")` | no | `"human"` | Output format - 'iso' (ISO 8601), 'rfc2822' (RFC 2822), or 'human' (readable). Defaults to 'human'. |
| `timezone` | `string` | no | `"UTC"` | IANA timezone (e.g., 'Europe/Dublin', 'America/New_York', 'UTC'). Defaults to UTC. |

## `weather`

Get the current weather conditions for a location.

### Parameters

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `location` | `string` | yes | — | The city name. |
| `unit` | `enum("celsius" \| "fahrenheit")` | no | `"celsius"` | The temperature unit (celsius or fahrenheit). |

## `web_search`

Search the web using the Brave Search API.

### Parameters

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `count` | `integer` | no | `5` | Number of results to return (1-20, default 5). |
| `country` | `string` | no | `"us"` | 2-letter ISO country code (default 'us'). Maps to Brave's 'cc' param. |
| `freshness` | `enum("pd" \| "pw" \| "pm" \| "py") \| null` | no | `null` | Freshness filter — 'pd' (past day), 'pw' (past week), 'pm' (past month), 'py' (past year), or None (no filter). |
| `query` | `string` | yes | — | Search query (required, non-empty after stripping). |
| `search_lang` | `string` | no | `"en"` | Language code (default 'en'). Maps to Brave's 'hl' param. |

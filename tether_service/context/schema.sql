-- placeholder schema
CREATE TABLE sessions (session_id TEXT PRIMARY KEY, data JSON);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    thinking_text TEXT,
    tool_name TEXT, -- Added for tool calls
    args TEXT,
    result TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
);
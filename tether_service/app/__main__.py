import uvicorn
from tether_service.core.config import load_settings


def main():
    cfg = load_settings()
    # support nested override under app.api or top-level
    api_cfg = cfg.get('app', {}).get('api', {})
    host = api_cfg.get('host', cfg.get("host", "127.0.0.1"))
    port = api_cfg.get('port', cfg.get("port", 8080))
    uvicorn.run("tether_service.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
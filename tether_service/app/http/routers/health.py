import asyncio
from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz():
    return {"ok": True}


@router.get("/readyz")
async def readyz(request: Request):
    """
    Provider: can produce at least one token quickly.
    Store: can read history for a dummy session.
    """
    svc = request.app.state.gen_svc
    # Store check
    try:
        _ = await svc.store.get_history("_readiness")
    except Exception as e:
        return {"ready": False, "store": False, "provider": None, "error": str(e)}

    # Provider check (timeout for safety)
    try:
        async def _probe():
            agen = svc.provider.stream([{"role": "user", "content": "ping"}], tools=None)
            return await agen.__anext__()  # get first chunk

        _ = await asyncio.wait_for(_probe(), timeout=1.0)
        return {"ready": True, "store": True, "provider": True}
    except Exception as e:
        return {"ready": False, "store": True, "provider": False, "error": str(e)}

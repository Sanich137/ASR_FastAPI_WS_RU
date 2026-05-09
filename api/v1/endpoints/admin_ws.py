"""WebSocket endpoint для real-time метрик админ-панели."""

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from sqlalchemy import select

from core.security import decode_token  # type: ignore[import-untyped]
from db.enums import UserRole
from db.models import User
from db.session import AsyncSessionLocal

router = APIRouter()


@router.websocket("/api/v1/admin/ws")
async def admin_websocket(websocket: WebSocket):
    """Push метрик для админов. Auth через первый фрейм."""
    await websocket.accept()

    # Ожидаем auth фрейм в течение 5 секунд
    try:
        auth_msg = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
    except asyncio.TimeoutError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    if auth_msg.get("type") != "auth":
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    token = auth_msg.get("access_token", "")
    payload = decode_token(token)
    if not payload or not getattr(payload, "sub", None):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.id == payload.sub))
        user: User | None = result.scalar_one_or_none()
        if not user or user.role not in (UserRole.admin, UserRole.superadmin):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

    metrics_collector = websocket.app.state.metrics_collector
    ws_manager = websocket.app.state.ws_manager

    try:
        while True:
            if metrics_collector:
                metrics = metrics_collector.collect(
                    active_connections=ws_manager.active_connections_count if ws_manager else 0,
                    max_connections=getattr(ws_manager, "max_connections", 100),
                )
                await websocket.send_json({"type": "metrics", "data": metrics})
            else:
                await websocket.send_json(
                    {"type": "metrics", "data": {"detail": "Collector unavailable"}}
                )
            await asyncio.sleep(5.0)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logging.getLogger(__name__).error(f"Admin WS error: {exc}")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)

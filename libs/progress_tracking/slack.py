"""Slack Web API helper using http.client (no SDK dependency).

Posts messages via chat.postMessage, supporting thread replies.

Usage:
    from libs.progress_tracking.slack import post_message

    ts = post_message("Hello!")
    if ts:
        post_message("Reply!", thread_ts=ts)
"""

from __future__ import annotations

import contextlib
import http.client
import json
import os
import ssl
import sys

_SLACK_HOST = "slack.com"
_SLACK_PATH = "/api/chat.postMessage"
_REQUEST_TIMEOUT_SEC = 10


def post_message(
    text: str,
    *,
    channel: str | None = None,
    token: str | None = None,
    thread_ts: str | None = None,
) -> str | None:
    """Post a message to Slack via Web API.

    Args:
        text: Slack mrkdwn message body.
        channel: Channel name or ID. Defaults to CLAUDE_SLACK_CHANNEL env.
        token: Bot token. Defaults to CLAUDE_SLACK_BOT_TOKEN env.
        thread_ts: If set, reply to this thread. Otherwise post new message.

    Returns:
        The message ts string (for threading), or None if skipped/failed.
    """
    if channel is None:
        channel = os.environ.get("CLAUDE_SLACK_CHANNEL")
    if token is None:
        token = os.environ.get("CLAUDE_SLACK_BOT_TOKEN")

    if not channel or not token:
        return None

    payload: dict[str, str] = {
        "channel": channel,
        "text": text,
    }
    if thread_ts is not None:
        payload["thread_ts"] = thread_ts

    return _send(payload, token)


def _send(payload: dict[str, str], token: str) -> str | None:
    """Execute the HTTPS POST and parse the response.

    Args:
        payload: JSON body for chat.postMessage.
        token: Bearer token for authorization.

    Returns:
        Message ts on success, None on failure.
    """
    data = json.dumps(payload).encode()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }
    ctx = ssl.create_default_context()
    with contextlib.suppress(Exception):
        conn = http.client.HTTPSConnection(
            _SLACK_HOST,
            timeout=_REQUEST_TIMEOUT_SEC,
            context=ctx,
        )
        try:
            conn.request(
                "POST",
                _SLACK_PATH,
                body=data,
                headers=headers,
            )
            resp = conn.getresponse()
            return _parse_response(resp.read())
        finally:
            conn.close()
    return None


def _parse_response(body: bytes) -> str | None:
    """Parse Slack API response for the message timestamp.

    Args:
        body: Raw bytes from HTTP response body.

    Returns:
        The ts string if ok, None otherwise.
    """
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None

    if parsed.get("ok"):
        ts = parsed.get("ts")
        return str(ts) if ts is not None else None

    # Log error to stderr for debugging, but never raise
    error = parsed.get("error", "unknown")
    sys.stderr.write(f"[slack] API error: {error}\n")
    return None

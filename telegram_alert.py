"""Send the CSE outlook to Telegram with a bot.

The hourly job reads cse_output/telegram_config.json. That file holds the
bot token from @BotFather and the chat id of the person who pressed Start
on the bot. Both stay on this computer.

    python telegram_alert.py
"""

from __future__ import annotations

import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "cse_output" / "telegram_config.json"
MESSAGE_PATH = ROOT / "cse_output" / "hourly_message.txt"
API = "https://api.telegram.org/bot{token}/{method}"


def load_config() -> dict[str, str]:
    if not CONFIG_PATH.exists():
        return {"bot_token": "", "chat_id": ""}
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"bot_token": "", "chat_id": ""}
    if not isinstance(data, dict):
        return {"bot_token": "", "chat_id": ""}
    return {
        "bot_token": str(data.get("bot_token") or "").strip(),
        "chat_id": str(data.get("chat_id") or "").strip(),
    }


def save_config(bot_token: str, chat_id: str) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"bot_token": bot_token.strip(), "chat_id": str(chat_id).strip()}
    CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _api(token: str, method: str, **payload) -> tuple[bool, dict | str]:
    try:
        response = requests.post(API.format(token=token, method=method), json=payload, timeout=30)
    except requests.RequestException as exc:
        return False, f"Could not reach Telegram. {exc}"
    try:
        body = response.json()
    except ValueError:
        return False, f"Telegram returned HTTP {response.status_code}."
    if response.ok and body.get("ok"):
        return True, body
    description = body.get("description") if isinstance(body, dict) else ""
    return False, description or f"Telegram returned HTTP {response.status_code}."


def chat_id_from_updates(token: str) -> str:
    ok, body = _api(token, "getUpdates", timeout=0)
    if not ok or not isinstance(body, dict):
        return ""
    for update in reversed(body.get("result") or []):
        message = update.get("message") or update.get("edited_message") or {}
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if chat_id is not None:
            return str(chat_id)
    return ""


def send_message(text: str, bot_token: str = "", chat_id: str = "") -> tuple[bool, str]:
    config = load_config()
    token = (bot_token or config["bot_token"]).strip()
    chat = (chat_id or config["chat_id"]).strip()
    if not token:
        return False, "Telegram is not connected. Add the bot token from @BotFather."
    if not chat:
        chat = chat_id_from_updates(token)
    if not chat:
        return False, "Open the CSE bot in Telegram and press Start, then run the alert again."
    if token != config["bot_token"] or chat != config["chat_id"]:
        save_config(token, chat)
    if len(text) > 4000:
        text = text[:3990].rstrip() + "\n..."
    ok, body = _api(token, "sendMessage", chat_id=chat, text=text)
    if ok:
        return True, "Telegram message sent."
    return False, f"Telegram did not send the message. {body}"


def send_saved_outlook() -> tuple[bool, str]:
    if not MESSAGE_PATH.exists():
        return False, "No outlook message to send."
    return send_message(MESSAGE_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    sent, detail = send_saved_outlook()
    print(detail)
    raise SystemExit(0 if sent else 1)

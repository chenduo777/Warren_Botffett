from __future__ import annotations

import logging
from typing import Iterable

from telegram.ext import Application, CommandHandler, MessageHandler, filters

from src.bot.handlers import (
    deny_handler,
    help_command,
    make_message_handler,
    make_new_command,
    start_command,
)

logger = logging.getLogger(__name__)


def run_bot(graph, token: str, allowed_user_ids: Iterable[int]) -> None:
    """Build the python-telegram-bot Application and start long polling.

    `allowed_user_ids` is a hard whitelist — the bot answers nobody else.
    Refuses to start if empty, because a public Telegram bot with no auth
    can be discovered and used by strangers to drain LLM API credits.
    """
    allowed = set(allowed_user_ids)
    if not allowed:
        raise ValueError(
            "ALLOWED_TG_USER_IDS is empty. Refusing to start without a whitelist — "
            "set at least your own Telegram user_id in .env."
        )

    user_filter = filters.User(user_id=allowed)
    app = Application.builder().token(token).build()

    msg_handler = make_message_handler(graph)

    app.add_handler(CommandHandler("start", start_command, filters=user_filter))
    app.add_handler(CommandHandler("new", make_new_command(graph), filters=user_filter))
    app.add_handler(CommandHandler("help", help_command, filters=user_filter))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND & user_filter, msg_handler)
    )

    # Catch-all for everyone outside the whitelist. Must be added LAST.
    app.add_handler(MessageHandler(~user_filter, deny_handler))

    logger.info("Bot starting with %d whitelisted user(s)…", len(allowed))
    app.run_polling(drop_pending_updates=True)

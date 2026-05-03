from __future__ import annotations

import asyncio
import logging

from langchain_core.messages import HumanMessage
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)

WELCOME = (
    "嗨，我是巴菲特。\n"
    "問我關於 1977–2021 年股東信的任何事，我會根據書信原文回答。\n\n"
    "/new — 開新對話（清除上下文）\n"
    "/help — 顯示說明"
)

# Telegram hard-caps a single text message at 4096 chars. Leave headroom
# for the citations footer we append.
MAX_MESSAGE_CHARS = 3800


def _default_thread_id(chat_id: int) -> str:
    return f"tg-{chat_id}"


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.chat_data["thread_id"] = _default_thread_id(update.effective_chat.id)
    await update.message.reply_text(WELCOME)


def make_new_command(graph):
    """Build a /new handler that wipes this chat's checkpoint history.

    Thread id is deterministic (`tg-<chat_id>`) so it survives bot restarts
    without any extra Telegram-side persistence. /new simply deletes the
    thread's rows from Postgres — the next message starts a fresh state.
    """

    async def new_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        thread_id = _default_thread_id(chat_id)
        await asyncio.to_thread(graph.checkpointer.delete_thread, thread_id)
        context.chat_data["thread_id"] = thread_id
        await update.message.reply_text("已開始新對話，先前的上下文已清除。")

    return new_command


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME)


async def deny_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    user_id = user.id if user else None
    logger.warning("Unauthorized access attempt: user_id=%s", user_id)
    if update.message:
        await update.message.reply_text(
            f"抱歉，這個 bot 是私人使用。\n你的 user_id 是 {user_id}，"
            "若需存取請聯繫擁有者加入白名單。"
        )


def make_message_handler(graph):
    """Build the main RAG message handler bound to a compiled LangGraph."""

    async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message is None or not update.message.text:
            return

        chat_id = update.effective_chat.id
        thread_id = context.chat_data.setdefault(
            "thread_id", _default_thread_id(chat_id)
        )

        await context.bot.send_chat_action(chat_id, ChatAction.TYPING)

        try:
            result = await asyncio.to_thread(
                graph.invoke,
                {"messages": [HumanMessage(content=update.message.text)]},
                {"configurable": {"thread_id": thread_id}},
            )
        except Exception as exc:
            logger.exception("Graph invocation failed for chat_id=%s", chat_id)
            await update.message.reply_text(f"出錯了：{exc}")
            return

        answer = result["messages"][-1].content
        years = result.get("retrieved_years", [])

        if len(answer) > MAX_MESSAGE_CHARS:
            answer = answer[:MAX_MESSAGE_CHARS] + "…（截斷）"

        if years:
            answer = f"{answer}\n\n引用年份：{', '.join(str(y) for y in years)}"

        await update.message.reply_text(answer)

    return message_handler

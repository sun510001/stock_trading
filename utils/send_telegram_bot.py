# -*- coding: utf-8 -*-
"""
Author: sun510001 sqf121@gmail.com
Date: 2025-05-09 18:05:03
LastEditors: sun510001 sqf121@gmail.com
LastEditTime: 2025-05-09 18:05:03
FilePath: /stock_trading/tg_bot.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

from telegram import Bot


class TelegramNotifier:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id

    @staticmethod
    def format_message(stock_id, current_info, indicators, suggestions):
        lines = [f"💡 Suggestion: {suggestions}"]
        if suggestions.get('mfi+kdj', "NO DATA!") == "NO DATA!":
            lines.append("No data available.")
            return "\n".join(lines)
        
        lines.append(f"\n📈 Market Snapshot -> {stock_id}")
        lines += [f"├ {k.capitalize():<6}: {v}" for k, v in current_info.items()]

        lines.append("\n📊 Technical Indicators")
        lines.append(f"├ RSI  : {indicators.get('RSI'):.2f}")
        lines.append(f"├ MFI  : {indicators.get('MFI'):.2f}")
        lines.append("├ ADX  :")
        lines += [f"   ├ {k.capitalize():<6}: {v}" for k, v in indicators.get('ADX', {}).items()]

        kdj = indicators.get("KDJ", {})
        lines.append("├ KDJ")
        lines.append(f"   ├ K: {kdj.get('K', 0):.2f}")
        lines.append(f"   ├ D: {kdj.get('D', 0):.2f}")
        lines.append(f"   └ J: {kdj.get('J', 0):.2f}")

        return "\n".join(lines)

    async def send_message(self, stock_id, current_info, indicators, suggestions):
        bot = Bot(token=self.token)
        text = self.format_message(stock_id, current_info, indicators, suggestions)
        await bot.send_message(chat_id=self.chat_id, text=text)


def main():
    import asyncio
    
    
    stock_id = "AAPL.US"

    token = "xxxx"
    chat_id = "xxxx"

    current_info = {"close": 29.265, "high": 29.27, "low": 29.22, "volume": 1743.0}
    indicators = {
        "RSI": 47.8806,
        "MFI": 66.0124,
        "KDJ": {"K": 14.408, "D": 13.629, "J": 15.965},
    }
    suggestions = {"mfi":"hold"}

    notifier = TelegramNotifier(token, chat_id)
    asyncio.run(notifier.send_message(stock_id, current_info, indicators, suggestions))


if __name__ == "__main__":
    main()

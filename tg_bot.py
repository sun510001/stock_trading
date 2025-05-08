from telegram import Bot


class TelegramNotifier:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id

    @staticmethod
    def format_message(stock_id, current_info, indicators, suggestion):
        lines = [f"📈 Market Snapshot -> {stock_id}"]
        lines += [f"├ {k.capitalize():<6}: {v}" for k,
                  v in current_info.items()]

        lines.append("\n📊 Technical Indicators")
        lines.append(f"├ RSI  : {indicators.get('RSI'):.2f}")
        lines.append(f"├ MFI  : {indicators.get('MFI'):.2f}")

        kdj = indicators.get("KDJ", {})
        lines.append("├ KDJ")
        lines.append(f"   ├ K: {kdj.get('K', 0):.2f}")
        lines.append(f"   ├ D: {kdj.get('D', 0):.2f}")
        lines.append(f"   └ J: {kdj.get('J', 0):.2f}")

        lines.append(f"\n💡 Suggestion: {suggestion}")
        return "\n".join(lines)

    async def send_message(self, stock_id, current_info, indicators, suggestion):
        bot = Bot(token=self.token)
        text = self.format_message(stock_id, current_info, indicators, suggestion)
        await bot.send_message(chat_id=self.chat_id, text=text)


def main():
    import asyncio
    token = "7658035273:AAFUn2BmO_LD07u0VGlFaUh__BHjXgIuLKw"
    chat_id = "1531273636"

    current_info = {
        'close': 29.265,
        'high': 29.27,
        'low': 29.22,
        'volume': 1743.0
    }
    indicators = {
        'RSI': 47.8806,
        'MFI': 66.0124,
        'KDJ': {
            'K': 14.408,
            'D': 13.629,
            'J': 15.965
        }
    }
    suggestion = "hold"

    notifier = TelegramNotifier(token, chat_id)
    asyncio.run(notifier.send_message(current_info, indicators, suggestion))


if __name__ == '__main__':
    main()

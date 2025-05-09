'''
Author: sun510001 sqf121@gmail.com
Date: 2025-05-08 01:29:51
LastEditors: sun510001 sqf121@gmail.com
LastEditTime: 2025-05-09 18:06:13
FilePath: /home_process/stock_trading/send_email.py
Description: send email
'''

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from logger import logger


class GmailStockNotifier:
    def __init__(
        self, sender_email, app_password, receiver_email, smtp_server="smtp.gmail.com", smtp_port=465
    ):
        """
        Initialize the notifier with Gmail credentials and receiver address.
        """
        self.sender_email = sender_email
        self.app_password = app_password
        self.receiver_email = receiver_email
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port

    def send_stock_report(
        self, stock_name, stock_code, current_info, indicators, suggestion
    ):
        subject = f"Stock Report: {stock_name} ({stock_code})"

        # Format KDJ
        kdj = indicators.get("KDJ", {})
        k_value = kdj.get("K", "N/A")
        d_value = kdj.get("D", "N/A")
        j_value = kdj.get("J", "N/A")

        html_content = f"""
        <html>
        <body>
            <h2>📈 Stock Report - {stock_name} ({stock_code})</h2>

            <h3>💰 Current Info</h3>
            <ul>
                <li><strong>Close:</strong> {current_info.get('close', 'N/A')}</li>
                <li><strong>High:</strong> {current_info.get('high', 'N/A')}</li>
                <li><strong>Low:</strong> {current_info.get('low', 'N/A')}</li>
                <li><strong>Volume:</strong> {current_info.get('volume', 'N/A')}</li>
            </ul>

            <h3>📊 Indicators</h3>
            <ul>
                <li><strong>RSI:</strong> {indicators.get('RSI', 'N/A'):.2f}</li>
                <li><strong>MFI:</strong> {indicators.get('MFI', 'N/A'):.2f}</li>
                <li><strong>KDJ:</strong> K={k_value:.2f}, D={d_value:.2f}, J={j_value:.2f}</li>
            </ul>

            <h3>📌 Suggestion</h3>
            <p style="font-size: 16px; color: blue;"><strong>{suggestion.upper()}</strong></p>
        </body>
        </html>
        """

        message = MIMEMultipart()
        message["From"] = self.sender_email
        message["To"] = self.receiver_email
        message["Subject"] = subject
        message.attach(MIMEText(html_content, "html"))

        try:
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                server.login(self.sender_email, self.app_password)
                server.send_message(message)
                logger.info("E-mail sent successfully!")
        except BaseException as e:
            logger.warning(f"Failed to send E-mail: {e}")
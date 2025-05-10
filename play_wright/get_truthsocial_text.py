"""
Author: sun510001 sqf121@gmail.com
Date: 2025-05-07 20:03:13
LastEditors: sun510001 sqf121@gmail.com
LastEditTime: 2025-05-07 20:03:13
FilePath: /home_process/play_wright/get_truthsocial_text.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

import time
import random

from logger import logging
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

from user_agent_gen import UserAgent


def process_html(html):
    result = {}
    soup = BeautifulSoup(html, "html.parser")

    username_tag = soup.find("p", string=lambda s: s and s.startswith("@"))
    result["username"] = username_tag.get_text(strip=True)

    time_tag = soup.find("time")
    # result['time_pass'] = time_tag.get_text(strip=True)
    result["time_title"] = time_tag.get("title")

    main_text = username_tag.find_next("p")
    raw_text = main_text.get_text(strip=True)
    result["text"] = raw_text
    return result


def get_latest_trump_post(url, user_agent, div):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=user_agent)
        page = context.new_page()
        page.goto(url, timeout=60000)

        page.wait_for_selector(div)
        post = page.query_selector(div)
        # posts = page.query_selector_all(div)

        # for post in posts:
        if post:
            html = post.inner_html()
            text = process_html(html)
            # logging.info(f"Get post: \n{[print(x) for x in text]}")
        else:
            text = {}
            logging.error(f"Failed to get post.")

        browser.close()
        return text


if __name__ == "__main__":
    url = "https://truthsocial.com/@realDonaldTrump"
    # user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    div = "div[class*='status__wrapper space-y-4 status-public p-4']"
    prev_text = {}
    count = 0

    user_agent = UserAgent()

    while True:
        count += 1
        wait_time = random.uniform(10.0, 20.0)
        ua = user_agent()
        logging.info(f"User agent: {ua}.")
        time.sleep(wait_time)
        logging.info(f"Sleep {wait_time}, try: {count}.")
        text = get_latest_trump_post(url, ua, div)
        if prev_text != text:
            prev_text = text
            logging.info(f"Get post: {text}")

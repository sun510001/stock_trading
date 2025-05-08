import random

from logger import logging


class UserAgent:
    def __init__(self):
        self.os_list = [
            # Windows
            "Windows NT 10.0; Win64; x64",
            "Windows NT 6.1; Win64; x64",
            "Windows NT 6.1; WOW64",
            "Windows NT 6.3; Win64; x64",
            "Windows NT 10.0; WOW64",
            # macOS
            "Macintosh; Intel Mac OS X 13_6",
            "Macintosh; Intel Mac OS X 12_5",
            "Macintosh; Intel Mac OS X 11_4",
            # Linux
            "X11; Linux x86_64",
            "X11; Ubuntu; Linux x86_64",
            "X11; Fedora; Linux x86_64",
            # iOS (iPhone, iPad)
            "iPhone; CPU iPhone OS 17_1 like Mac OS X",
            "iPhone; CPU iPhone OS 16_6 like Mac OS X",
            "iPad; CPU OS 16_6 like Mac OS X",
            # Android
            "Linux; Android 13; Pixel 7 Pro",
            "Linux; Android 12; Pixel 6",
            "Linux; Android 11; SM-G991B",  # Galaxy S21
            "Linux; Android 10; Mi 9",
            # ChromeOS
            "X11; CrOS x86_64 14526.83.0",
        ]
        self.browser_dict = {
            "Chrome": self.get_chrome_params(),
            "Firefox": self.get_firefox_params(),
            "Safari": self.get_safari_params(),
            "Edge": self.get_edge_params(),
            "Opera": self.get_opera_params(),
        }

    def get_chrome_params(self):
        major = random.randint(118, 122)
        minor = 0
        build = random.randint(5000, 6500)
        patch = random.randint(50, 200)
        return ("Chrome", f"{major}.{minor}.{build}.{patch}")

    def get_firefox_params(self):
        major = random.randint(115, 125)
        return ("Firefox", f"{major}.0")

    def get_safari_params(self):
        return ("Safari", random.choice(["604.1", "605.1.15"]))

    def get_edge_params(self):
        major = random.randint(118, 122)
        minor = 0
        build = random.randint(2000, 3000)
        patch = random.randint(50, 150)
        return ("Edge", f"{major}.{minor}.{build}.{patch}")

    def get_opera_params(self):
        major = random.randint(90, 110)
        minor = 0
        build = random.randint(4000, 6000)
        patch = random.randint(10, 100)
        return ("Opera", f"{major}.{minor}.{build}.{patch}")

    def __call__(self):
        while True:
            os_str = random.choice(self.os_list)
            browser_func_key = random.choice(list(self.browser_dict.keys()))
            browser, version = self.browser_dict[browser_func_key]
            if browser == "Chrome":
                ua = f"Mozilla/5.0 ({os_str}) AppleWebKit/537.36 (KHTML, like Gecko) {browser}/{version} Safari/537.36"
                break
            elif browser == "Firefox":
                ua = f"Mozilla/5.0 ({os_str}) Gecko/20100101 {browser}/{version}"
                break
            elif browser == "Safari":
                if "iPhone" in os_str or "Mac" in os_str:
                    ua = f"Mozilla/5.0 ({os_str}) AppleWebKit/{version} (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/{version}"
                    break
                else:
                    continue
            else:
                continue
        return ua


def main():
    user_agent = UserAgent()
    ua = user_agent()
    logging.info(ua)


if __name__ == "__main__":
    main()

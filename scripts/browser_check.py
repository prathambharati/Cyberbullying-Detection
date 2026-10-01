"""Click through the real app in Chrome and check that everything interactive works.

    pip install playwright
    python scripts/browser_check.py                     # just the checks
    python scripts/browser_check.py --screenshots docs  # and retake the README pictures

It seeds a throwaway database, starts the app, then signs up, types, posts, likes,
comments, switches theme and so on, like a person would. It uses the Chrome that's
already installed, so Playwright doesn't need to download a browser. The news page
needs an internet connection.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
PORT = 5091
BASE = f"http://127.0.0.1:{PORT}"


class Run:
    def __init__(self):
        self.results = []
        self.console = []

    def check(self, name, ok):
        self.results.append((name, bool(ok)))
        print(("  ok    " if ok else "  FAIL  ") + name, flush=True)

    def watch(self, page):
        def on_console(message):
            # News thumbnails come from another site and some fail to load. The page removes those.
            if message.type == "error" and "Failed to load resource" not in message.text:
                self.console.append(message.text)

        page.on("console", on_console)
        page.on("pageerror", lambda error: self.console.append(str(error)))
        page.on("dialog", lambda dialog: dialog.accept())


def start_server(folder: Path):
    env = {**os.environ, "CB_DATABASE": str(folder / "browser-check.db"), "CB_SECRET_KEY": "browser-check",
           "PYTHONIOENCODING": "utf-8"}
    flask = [sys.executable, "-m", "flask", "--app", "cyberbullying.web"]
    subprocess.run(flask + ["seed"], cwd=ROOT, env=env, check=True, capture_output=True)
    log = open(folder / "server.log", "w")
    server = subprocess.Popen(flask + ["run", "--port", str(PORT)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            requests.get(BASE, timeout=2)
            return server, log
        except requests.RequestException:
            time.sleep(1)
    raise SystemExit("The app didn't start. See " + str(folder / "server.log"))


def likes_of(post):
    return int(post.locator("[data-like-count]").inner_text())


def walk_through(run: Run, browser, shots: Path | None):
    def snap(page, name):
        if shots:
            page.screenshot(path=str(shots / name))

    page = browser.new_context(viewport={"width": 1400, "height": 980}, color_scheme="light").new_page()
    run.watch(page)

    print("Signing up")
    page.goto(BASE + "/register")
    page.fill("input[name=username]", "priya")
    page.fill("input[name=password]", "a strong password")
    page.fill("input[name=confirm]", "a strong password")
    page.click("form.stack button")
    page.wait_for_url(BASE + "/")
    run.check("signing up lands on the feed with a welcome toast", page.locator(".toast").first.is_visible())
    run.check("the stories row shows the demo people", page.locator(".stories .story").count() >= 6)
    run.check("the feed shows the demo posts", page.locator("[data-post]").count() >= 5)

    print("The live meter")
    box = page.locator("[data-composer] textarea")
    box.fill("Congrats on the new job, you totally deserve it!")
    page.wait_for_function("document.querySelector('[data-live]').dataset.state === 'good'")
    run.check("a kind draft reads as friendly", page.inner_text("[data-live-label]") == "Looks friendly")
    box.fill("nobody likes you, you pathetic idiot. just leave")
    page.wait_for_function("document.querySelector('[data-live]').dataset.state === 'bad'")
    run.check("a nasty draft is flagged while you type", "won't go up" in page.inner_text("[data-live-label]"))
    run.check("the character count keeps up", page.inner_text("[data-count]").startswith("48/"))

    print("A blocked post")
    page.click("[data-composer] button.primary")
    page.wait_for_selector(".notice.bad")
    run.check("the post comes back with an explanation", "Hold on" in page.inner_text(".notice.bad"))
    run.check("the words that set it off are highlighted", page.locator(".notice.bad mark.word").count() >= 5)
    run.check("the draft is still there to edit", "pathetic idiot" in box.input_value())
    page.wait_for_function("document.querySelector('[data-live]').dataset.state === 'bad'")
    snap(page, "feed.png")

    print("Likes")
    first = page.locator("[data-post]").first
    before = likes_of(first)
    first.locator("[data-like]").click()
    expect(first.locator("[data-like]")).to_have_attribute("aria-pressed", "true")
    run.check("the heart likes a post without reloading", likes_of(first) == before + 1)
    first.locator("[data-like]").click()
    expect(first.locator("[data-like]")).to_have_attribute("aria-pressed", "false")
    run.check("a second click takes the like back", likes_of(first) == before)
    second = page.locator("[data-post]").nth(1)
    was_liked = second.locator("[data-like]").get_attribute("aria-pressed") == "true"
    second.locator("[data-double-tap]").dblclick()
    run.check("double-clicking a post shows the heart burst", page.locator(".burst").count() >= 1)
    expect(second.locator("[data-like]")).to_have_attribute("aria-pressed", "true")
    run.check("double-clicking a post likes it", not was_liked)

    print("The theme")
    page.click(".sidebar [data-theme-toggle]")
    run.check("the theme button switches to dark", page.evaluate("document.documentElement.dataset.theme") == "dark")
    page.goto(BASE + "/")
    run.check("the choice sticks on the next page", page.evaluate("document.documentElement.dataset.theme") == "dark")
    page.click(".sidebar [data-theme-toggle]")
    run.check("and it switches back", page.evaluate("document.documentElement.dataset.theme") == "light")

    print("Posting, commenting and profiles")
    page.fill("[data-composer] textarea", "Just finished a great book about octopuses. Highly recommend it!")
    page.click("[data-composer] button.primary")
    page.wait_for_url(BASE + "/")
    run.check("a kind post goes up at the top", "octopuses" in page.locator("[data-post]").first.inner_text())
    page.locator("[data-post]").first.locator("a[aria-label=Comments]").click()
    page.fill("[data-composer] textarea", "Adding it to my list, thanks!")
    page.click("[data-composer] button.primary")
    page.wait_for_selector(".comment .bubble")
    run.check("a comment shows up under the post", "Adding it to my list" in page.inner_text(".comments"))
    page.goto(BASE + "/u/priya")
    run.check("the profile shows the new post", "octopuses" in page.inner_text("main"))
    page.locator("[data-post] button[aria-label='Delete post']").first.click()
    page.wait_for_url(BASE + "/")
    run.check("you can delete your own post after confirming", "octopuses" not in page.inner_text("main"))

    print("The playground")
    page.goto(BASE + "/check")
    page.click("[data-example='shut up you dumb idiot']")
    page.wait_for_selector(".verdict")
    run.check("an example chip checks itself", "sent back" in page.inner_text(".verdict h2"))
    page.fill("#text", "Girls like you should stay in the kitchen and shut up")
    page.click("form[data-check] button.primary")
    page.wait_for_selector(".verdict.bad")
    run.check("targeted abuse gets its category", "gender" in page.inner_text(".verdict"))
    snap(page, "check.png")

    print("News")
    page.goto(BASE + "/news")
    page.wait_for_load_state("networkidle")
    run.check("today's headlines load", page.locator(".headline").count() > 0)
    run.check("they're recent, not years old", "days ago" not in page.locator(".headline .source").first.inner_text())

    print("On a phone")
    phone = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True,
                                has_touch=True, color_scheme="light").new_page()
    run.watch(phone)
    phone.goto(BASE + "/login")
    phone.fill("input[name=username]", "maya")
    phone.fill("input[name=password]", "demo-password")
    phone.click("form.stack button")
    phone.wait_for_url(BASE + "/")
    run.check("the tab bar shows on a phone", phone.locator(".tabbar").is_visible())
    run.check("the sidebar hides on a phone", not phone.locator(".sidebar").is_visible())
    if shots:
        phone.wait_for_timeout(5200)  # let the welcome toast slide away first
        snap(phone, "mobile.png")

    print("Logging out")
    page.goto(BASE + "/")
    page.click(".side-foot form button")
    page.wait_for_url(BASE + "/")
    run.check("logging out brings back the welcome card", page.locator(".welcome").is_visible())


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the app in a real browser.")
    parser.add_argument("--screenshots", type=Path, help="save the README screenshots to this folder")
    args = parser.parse_args()
    if args.screenshots:
        args.screenshots.mkdir(parents=True, exist_ok=True)

    run = Run()
    folder = Path(tempfile.mkdtemp())
    server, log = start_server(folder)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True)
            walk_through(run, browser, args.screenshots)
            browser.close()
    finally:
        server.terminate()
        server.wait()
        log.close()

    server_errors = [line for line in (folder / "server.log").read_text(errors="replace").splitlines()
                     if "Traceback" in line or "Error" in line]
    run.check("no JavaScript errors in the browser console", not run.console)
    run.check("no errors in the server log", not server_errors)
    for line in run.console + server_errors:
        print("    " + line)
    failed = [name for name, ok in run.results if not ok]
    print(f"\n{len(run.results) - len(failed)} of {len(run.results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

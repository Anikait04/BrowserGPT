import asyncio
from playwright.async_api import async_playwright

async def fetch_dom():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://langchain.com", wait_until="networkidle")
        dom = await page.content()

        with open("doms.txt", "w", encoding="utf-8") as f:
            f.write(dom)

        await browser.close()

if __name__ == "__main__":
    asyncio.run(fetch_dom())

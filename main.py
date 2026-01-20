import asyncio
import os

from test_agent import run_agent
from logs import logger, log_separator


async def main():
    """Main entry point with example tasks"""

    log_separator("APPLICATION START")

    # Check for OpenRouter API Key
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY not found in .env file")
        return

    logger.info("OPENROUTER_API_KEY detected successfully")

    logger.info("Starting Browser Automation Agent")

    # TASK 1: Simple Google search
    logger.info("Task started: Google search for langchain.com")

    try:
        await run_agent(
            goal="Search for langchain.com and click the first result.",
            max_steps=30
        )
        logger.info("Task completed successfully")

    except Exception as e:
        logger.exception("Agent execution failed")

    log_separator("APPLICATION END")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Application interrupted by user")
    except Exception:
        logger.exception("Unhandled application error")

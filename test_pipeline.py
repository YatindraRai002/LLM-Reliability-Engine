import asyncio
from core.aggregator import async_run_full_pipeline
import logging

logging.basicConfig(level=logging.INFO)

async def test():
    print("Testing async_run_full_pipeline...")
    try:
        res = await async_run_full_pipeline("what is a super weird random query 12345")
        # Write to a file instead of print to avoid cp1252 encoding errors
        with open("test_output.txt", "w", encoding="utf-8") as f:
            f.write(str(res))
        print("Success, wrote to test_output.txt")
    except Exception as e:
        print("Pipeline failed:", e)

if __name__ == "__main__":
    asyncio.run(test())

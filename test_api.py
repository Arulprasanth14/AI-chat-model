import httpx
import asyncio

async def main():
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            r = await client.post('http://localhost:8000/conversation/message', json={'user_message': '__start__', 'session_id': None, 'vertical': 'restaurant'})
            print(r.status_code)
            print(r.text)
        except Exception as e:
            print("EXCEPTION:", repr(e))

asyncio.run(main())

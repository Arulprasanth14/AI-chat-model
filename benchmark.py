import asyncio
import httpx
import uuid
import re

BASE_URL = "http://127.0.0.1:8000"

async def run_turn(client: httpx.AsyncClient, session_id: str, message: str, vertical: str = None, template_key: str = None):
    payload = {
        "user_message": message,
    }
    if session_id:
        payload["session_id"] = session_id
    if vertical:
        payload["vertical"] = vertical
    if template_key:
        payload["template_key"] = template_key

    print(f"\nSending message: {message}")
    
    async with client.stream("POST", f"{BASE_URL}/conversation/message", json=payload, timeout=60.0) as response:
        response.raise_for_status()
        text = ""
        async for chunk in response.aiter_text():
            text += chunk
        
        # parse final done event for session ID if we didn't have one
        import json
        last_event = text.strip().split("data: ")[-1]
        try:
            data = json.loads(last_event)
            if data.get("done"):
                return data["snapshot"]["session_id"]
        except:
            pass
        return session_id

async def main():
    async with httpx.AsyncClient() as client:
        # Scenario 1: New session (no extraction, pure chat)
        session_id = await run_turn(client, None, "Hi, I just want to chat.", "restaurant", "restaurant_cafe_static_post")
        
        # Scenario 2: Single-field extraction
        await run_turn(client, session_id, "The name of my restaurant is The Golden Spoon.")
        
        # Scenario 3: Multi-field extraction
        await run_turn(client, session_id, "It is located in New York, and we serve Italian food. My target audience is families.")
        
        # Scenario 4: Full RAG retrieval trigger (asking for guidance)
        await run_turn(client, session_id, "Can you give me some examples of good social media posts for this type of restaurant?")
        
        # Scenario 5: Another pure chat
        await run_turn(client, session_id, "That makes sense, thanks.")
        
if __name__ == "__main__":
    asyncio.run(main())

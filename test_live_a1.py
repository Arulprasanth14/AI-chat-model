import httpx
import json

def run_test():
    base_url = "http://localhost:8000"
    
    # 1. Start a new session with ALL required info to force completion immediately
    print("Starting a new session and sending all info...")
    msg1 = "Hi, I need a static post for my restaurant, The Gourmet Kitchen. Target audience is foodies. Objective is to increase visits. Tone is warm. Key message is fresh food. Timeline is next week. Distribution is Instagram. Budget is $500. Deliverables are 1 post. Success metrics is 100 likes. No offer. Use saved brand kit."
    resp1 = httpx.post(
        f"{base_url}/conversation/message", 
        json={"user_message": msg1, "vertical": "restaurant", "template_key": "restaurant_cafe_static_post"},
        timeout=60.0
    )
    
    if resp1.status_code != 200:
        print(f"HTTP Error {resp1.status_code}: {resp1.text}")
        return
    
    # Parse SSE to get session_id
    session_id = None
    snapshot = None
    for line in resp1.iter_lines():
        if line.startswith("data: "):
            try:
                data = json.loads(line[6:])
                if "done" in data and data["done"]:
                    snapshot = data["snapshot"]
                    session_id = snapshot["session_id"]
            except Exception:
                pass
                
    print(f"Session ID: {session_id}")
    if snapshot:
        print(f"Is Complete? {snapshot.get('is_complete')}")
        print(f"All extracted answers: {snapshot.get('extracted_answers')}")
        print(f"Current Target Audience: {snapshot.get('extracted_answers', {}).get('target_audience', {}).get('value')}")
    else:
        print("Error: No snapshot received. The stream might have ended abruptly.")
        return
    
    # 3. Post-completion correction
    msg3 = "Actually, change the target audience to families with kids."
    print(f"\nSending correction: '{msg3}'")
    resp3 = httpx.post(f"{base_url}/conversation/message", json={"session_id": session_id, "user_message": msg3}, timeout=120.0)
    
    reply = ""
    for line in resp3.iter_lines():
        if line.startswith("data: "):
            data = json.loads(line[6:])
            if "chunk" in data:
                reply += data["chunk"]
            if "done" in data and data["done"]:
                snapshot = data["snapshot"]
                
    print(f"AI Reply: {reply}")
    print(f"Updated Target Audience: {snapshot['extracted_answers'].get('target_audience', {}).get('value')}")

if __name__ == "__main__":
    run_test()

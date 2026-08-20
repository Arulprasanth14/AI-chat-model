import httpx
import json

base_url = "http://localhost:8000"

def start_session():
    msg1 = "Hi, I need a static post for my restaurant."
    print("Starting a new session...")
    resp1 = httpx.post(
        f"{base_url}/conversation/message", 
        json={"user_message": msg1, "vertical": "restaurant", "template_key": "restaurant_cafe_static_post"},
        timeout=60.0
    )
    
    session_id = None
    for line in resp1.iter_lines():
        if line.startswith("data: "):
            try:
                data = json.loads(line[6:])
                if "done" in data and data["done"]:
                    snapshot = data["snapshot"]
                    session_id = snapshot["session_id"]
                    print(f"Start Session Intent: {snapshot.get('last_intent')}")
            except Exception:
                pass
    return session_id

def send_message(session_id, message, expected_intent):
    print(f"\nSending: '{message}'")
    resp = httpx.post(
        f"{base_url}/conversation/message", 
        json={"session_id": session_id, "user_message": message}, 
        timeout=120.0
    )
    
    intent = None
    reply = ""
    for line in resp.iter_lines():
        if line.startswith("data: "):
            data = json.loads(line[6:])
            if "chunk" in data:
                reply += data["chunk"]
            if "done" in data and data["done"]:
                snapshot = data["snapshot"]
                intent = snapshot.get("last_intent")
                
    print(f"AI Reply: {reply}")
    print(f"Intent classified: {intent}")
    print(f"Expected: {expected_intent}")
    if intent == expected_intent:
        print("[PASS]")
    else:
        print("[FAIL]")

if __name__ == "__main__":
    sid = start_session()
    if not sid:
        print("Failed to start session")
        exit(1)
        
    test_cases = [
        ("wait that's wrong, it's actually families", "CORRECT_PREVIOUS"),
        ("what do you mean by target audience?", "REQUEST_CLARIFICATION"),
        ("how much does a billboard cost?", "ASK_QUESTION"),
        ("sure sounds good", "CONFIRM"),
        ("what is your favorite color? also my budget is $500", ["PROVIDE_INFO", "ASK_QUESTION"])
    ]
    
    for msg, expected in test_cases:
        if isinstance(expected, list):
            print(f"\nSending: '{msg}'")
            resp = httpx.post(
                f"{base_url}/conversation/message", 
                json={"session_id": sid, "user_message": msg}, 
                timeout=120.0
            )
            intent = None
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if "done" in data and data["done"]:
                        snapshot = data["snapshot"]
                        intent = snapshot.get("last_intent")
            print(f"Intent classified: {intent} (Expected one of: {expected})")
            if intent in expected:
                print("[PASS]")
            else:
                print("[FAIL]")
        else:
            send_message(sid, msg, expected)

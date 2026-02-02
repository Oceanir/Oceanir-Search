#!/usr/bin/env python3
"""
Oceanir Search + Free LLM Integration Test
Uses DuckDuckGo AI Chat (free, no API key needed)
"""

import subprocess
import requests
import json

def oceanir_search(query):
    """Run Oceanir Search"""
    result = subprocess.run(
        ["./zig-out/bin/oceanir-search", "search", query],
        capture_output=True,
        text=True,
        timeout=30
    )
    # Combine stdout and stderr
    output = result.stdout + result.stderr
    return output.strip()

def ask_duckduckgo_ai(prompt):
    """Use DuckDuckGo AI Chat - FREE, no API key!"""
    # DuckDuckGo AI uses a simple POST endpoint
    url = "https://duckduckgo.com/duckchat/v1/chat"
    
    # First get a vqd token
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "x-vqd-accept": "1"
    }
    
    # Get status/token
    status = requests.get("https://duckduckgo.com/duckchat/v1/status", headers={"x-vqd-accept": "1"})
    vqd = status.headers.get("x-vqd-4", "")
    
    if not vqd:
        return "Could not get DuckDuckGo token"
    
    headers["x-vqd-4"] = vqd
    
    payload = {
        "model": "gpt-4o-mini",  # Free via DDG
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        # Parse SSE response
        text = ""
        for line in resp.text.split("\n"):
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if "message" in data:
                        text += data["message"]
                except:
                    pass
        return text if text else resp.text[:300]
    except Exception as e:
        return f"Error: {e}"

def main():
    print("=" * 60)
    print("🔍 OCEANIR SEARCH + FREE LLM TEST")
    print("=" * 60)
    
    # Step 1: Search
    query = "error handling"
    print(f"\n1️⃣  Searching for: '{query}'")
    search_results = oceanir_search(query)
    print(search_results)
    
    # Step 2: Ask LLM to analyze
    print("\n" + "-" * 60)
    print("2️⃣  Asking GPT-4o-mini (via DuckDuckGo, FREE)...")
    
    prompt = f"""Analyze these code search results and tell me which file is most relevant for error handling:

{search_results}

Answer in 2 sentences max."""

    llm_response = ask_duckduckgo_ai(prompt)
    print(f"\n🤖 LLM Analysis:\n{llm_response}")
    
    print("\n" + "=" * 60)
    print("✅ Integration test complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

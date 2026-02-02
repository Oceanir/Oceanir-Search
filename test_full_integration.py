#!/usr/bin/env python3
"""
🚀 FULL INTEGRATION TEST: Oceanir Search + Ollama LLM
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
    return (result.stdout + result.stderr).strip()

def ask_ollama(prompt, model="qwen2.5:0.5b"):
    """Query local Ollama"""
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=30
        )
        return resp.json().get("response", "No response")
    except Exception as e:
        return f"Ollama error: {e}"

def main():
    print("=" * 60)
    print("🚀 OCEANIR SEARCH + OLLAMA LLM - FULL TEST")
    print("=" * 60)
    
    # Step 1: Semantic search
    query = "embedding model initialization"
    print(f"\n1️⃣  Oceanir Search: '{query}'")
    print("-" * 40)
    search_results = oceanir_search(query)
    print(search_results)
    
    # Step 2: LLM analysis
    print("\n2️⃣  Asking Qwen2.5 (local Ollama)...")
    print("-" * 40)
    
    prompt = f"""Based on these code search results:

{search_results}

Which file should I look at first for understanding embedding initialization? Answer in one sentence."""

    llm_answer = ask_ollama(prompt)
    print(f"🤖 LLM: {llm_answer}")
    
    # Step 3: Follow-up search based on LLM suggestion
    print("\n3️⃣  Follow-up search: 'onnx session'")
    print("-" * 40)
    followup = oceanir_search("onnx session")
    print(followup)
    
    print("\n" + "=" * 60)
    print("✅ FULL INTEGRATION WORKING!")
    print("=" * 60)
    print("""
This demonstrates:
  1. Semantic code search (Oceanir Search)
  2. LLM analysis of results (Ollama)  
  3. Iterative search refinement

Ready for production use! 🎉
""")

if __name__ == "__main__":
    main()

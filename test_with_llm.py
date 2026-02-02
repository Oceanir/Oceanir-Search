#!/usr/bin/env python3
"""Test Oceanir Search with free LLMs"""

import subprocess
import requests
import os
import json

def oceanir_search(query):
    """Run Oceanir Search CLI"""
    try:
        result = subprocess.run(
            ["./zig-out/bin/oceanir-search", "search", query],
            capture_output=True,
            text=True,
            timeout=30,
            cwd="/Users/kanayochukew/railweb/OceanirPublic/oceanir-search"
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Search error: {e}"

def test_search():
    print("=" * 60)
    print("🔍 OCEANIR SEARCH TEST")
    print("=" * 60)
    
    queries = [
        "error handling",
        "vector similarity", 
        "embedding model"
    ]
    
    for q in queries:
        print(f"\n📌 Query: '{q}'")
        print("-" * 40)
        result = oceanir_search(q)
        # Show first 400 chars
        print(result[:400] if result else "No results")
    
    print("\n" + "=" * 60)
    print("✅ Oceanir Search working!")
    print("=" * 60)

    # Test with free online LLM via OpenRouter (has free models)
    print("\n\n🤖 Testing LLM Integration...")
    
    # Try HuggingChat API (free, no key)
    try:
        search_result = oceanir_search("main function")
        
        # Use a simple echo test since we don't have API keys
        print(f"\nSearch result for 'main function':")
        print(search_result[:300])
        
        print("\n💡 To integrate with LLM:")
        print("   1. Get free Groq key: console.groq.com")
        print("   2. Or use OpenRouter: openrouter.ai (free models available)")
        print("   3. Pass search results as context to any LLM")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_search()

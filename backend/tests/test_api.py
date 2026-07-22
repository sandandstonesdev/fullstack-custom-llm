#!/usr/bin/env python3
"""
Simple script to test the FastAPI endpoints
"""
import requests
import json

def test_api_endpoints():
    base_url = "http://localhost:8000"
    
    print("Testing Polish LLM API Endpoints")
    print("=" * 40)
    
    # Test 1: Health check
    print("\n1. Testing /health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Model info
    print("\n2. Testing /model-info endpoint...")
    try:
        response = requests.get(f"{base_url}/model-info")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Inference
    print("\n3. Testing /inference endpoint...")
    test_prompt = "Polska to piękny kraj, który"
    payload = {
        "prompt": test_prompt,
        "max_tokens": 30,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            f"{base_url}/inference",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        print(f"Prompt: {test_prompt}")
        print(f"Generated: {response.json()['generated_text']}")
        print(f"Tokens: {response.json()['input_tokens']} input, {response.json()['generated_tokens']} generated")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: Batch inference
    print("\n4. Testing /batch-inference endpoint...")
    batch_payload = [
        {"prompt": "Warszawa jest", "max_tokens": 20, "temperature": 0.5},
        {"prompt": "Największe miasta Polski to", "max_tokens": 25, "temperature": 0.6}
    ]
    
    try:
        response = requests.post(
            f"{base_url}/batch-inference",
            json=batch_payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        results = response.json()["results"]
        for i, result in enumerate(results):
            print(f"\nBatch {i+1}:")
            print(f"Prompt: {result['prompt']}")
            print(f"Generated: {result['generated_text']}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api_endpoints()
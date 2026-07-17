import socket
import urllib.request
import urllib.error
import json
import sys

# The address your host machine must use to talk to the containerized Langfuse
LOCAL_HOST = "http://localhost:3000"
HEALTH_ENDPOINT = f"{LOCAL_HOST}/api/public/health"

print("=" * 60)
print(" LANGFUSE LOCAL CONNECTIVITY DIAGNOSTIC RUN")
print("=" * 60)

# Step 1: Check if port 3000 is open on localhost
print("\n[Step 1] Verifying host-to-container port mapping...")
try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(3)
        s.connect(("127.0.0.1", 3000))
    print("  SUCCESS: Port 3000 is open and accepting connections on localhost.")
except Exception as e:
    print(f"  ❌ FAILURE: Cannot connect to 127.0.0.1:3000. Error: {e}")
    print("  👉 Reason: Your Docker containers might be stopped, or port 3000 is not exposed.")
    print("  👉 Fix: Run 'docker compose up -d' and check 'docker compose ps'.")
    sys.exit(1)

# Step 2: Test API Health Endpoint
print("\n[Step 2] Testing Langfuse health API endpoint...")
try:
    req = urllib.request.Request(HEALTH_ENDPOINT, method="GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
        body = resp.read().decode("utf-8")
        data = json.loads(body)
        status = data.get("status", "")
        
        print(f"  HTTP Response Status: {resp.status}")
        print(f"  API Payload Received: {data}")
        
        if status == "OK":
            print("  SUCCESS: Langfuse server is internally healthy and operational!")
        else:
            print(f"  ⚠️ WARNING: Connected, but status payload is non-OK: {status}")
except urllib.error.URLError as e:
    print(f"  ❌ FAILURE: HTTP request failed to reach {HEALTH_ENDPOINT}.")
    print(f"  Error details: {e}")
    sys.exit(1)
except Exception as e:
    print(f"  ❌ FAILURE: Unexpected structural parsing error: {e}")
    sys.exit(1)

# Step 3: Verify the official SDK initialization
print("\n[Step 3] Testing Langfuse Python SDK authorization handshake...")
try:
    from langfuse import Langfuse
    
    # Hardcoded local variables to bypass config loader issues
    lf = Langfuse(
        public_key="pk-lf-ff6ebcae-7f5f-470a-92b9-cd78ed04a8be",
        secret_key="sk-lf-30b5912b-5882-4fe3-acfd-ecf0e38d1bb1",
        host=LOCAL_HOST,
        debug=False
    )
    
    auth_ok = lf.auth_check()
    if auth_ok:
        print("  SUCCESS: SDK successfully authenticated with the local Langfuse database!")
        print("\n Everything is configured correctly. Your environment variable must be:")
        print(f'   export LANGFUSE_HOST="{LOCAL_HOST}"')
    else:
        print("  ❌ FAILURE: Connected to server, but the API Keys were rejected.")
except ImportError:
    print("  ❌ FAILURE: The 'langfuse' package is missing from your active virtual environment.")
    print("  👉 Fix: Run 'pip install langfuse'.")
except Exception as e:
    print(f"  ❌ FAILURE: SDK authentication loop crashed. Details: {e}")

print("\n" + "=" * 60)
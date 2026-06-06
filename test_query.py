import urllib.request, json, urllib.error, time

BASE = "http://localhost:8000"

print("Waiting for FastAPI server...")
for i in range(20):
    try:
        r = urllib.request.urlopen(f"{BASE}/api/health", timeout=3)
        data = json.loads(r.read().decode())
        print(f"Server UP: {data}")
        break
    except Exception:
        time.sleep(3)
        print(f"  ...waiting ({(i+1)*3}s)")
else:
    print("Server not reachable!"); exit(1)

print("\nRunning RAG query...")
data = json.dumps({"question": "What are the symptoms of diabetes?"}).encode()
req = urllib.request.Request(
    f"{BASE}/api/ask", data=data,
    headers={"Content-Type": "application/json"}
)
try:
    r = urllib.request.urlopen(req, timeout=120)
    resp = json.loads(r.read().decode())
    print("\n=== MODEL ===", resp.get("model"))
    print("\n=== ANSWER (first 500 chars) ===")
    print(resp.get("answer", "")[:500])
    print("\n=== CITATIONS ===")
    for c in resp.get("citations", []):
        print("  -", c["filename"], "p." + str(c["page"]), "| CE:", round(c["rerank_score"], 3))
    print("\n=== TIMING ===")
    t = resp.get("timing", {})
    for k, v in t.items():
        print(f"  {k}: {v}s")
except urllib.error.HTTPError as e:
    print("HTTP ERROR", e.code, e.read().decode()[:500])
except Exception as e:
    print("ERROR:", type(e).__name__, e)

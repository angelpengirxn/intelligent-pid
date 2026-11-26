"""Create a temporary public link to the app running on this Mac (Cloudflare quick tunnel, no account).

1) Start the app first:   uvicorn app.main:app --port 8000
2) Then run:               python src/share_link.py
The link works while this script and the app keep running and the Mac stays awake.
"""
import subprocess
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

try:
    from pycloudflared import try_cloudflare
except ImportError:
    sys.exit("Install it first:  pip install pycloudflared")

print(f"Opening a public link to http://localhost:{PORT} ...")
urls = try_cloudflare(port=PORT, verbose=False)
link = getattr(urls, "tunnel", urls)
print("\n  Share this link:  ", link, "\n")
print("Keep this window open. Press Ctrl + C to stop sharing.")

# keep the Mac awake while sharing (macOS 'caffeinate'); stops when this script stops
try:
    subprocess.run(["caffeinate", "-dimsu"])
except (FileNotFoundError, KeyboardInterrupt):
    try:
        input()
    except KeyboardInterrupt:
        pass

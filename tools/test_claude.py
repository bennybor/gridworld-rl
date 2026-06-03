import os
import json
import urllib.request
import urllib.error
from pathlib import Path


def get_claude_key():
    key = os.environ.get("CLAUDE_API_KEY")
    if key:
        return key
    p = Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml"
    if p.exists():
        for line in p.read_text().splitlines():
            if "CLAUDE_API_KEY" in line:
                parts = line.split("=", 1)[1].strip()
                if parts.startswith('"') and parts.endswith('"'):
                    return parts[1:-1]
                return parts
    return None


def main():
    key = get_claude_key()
    if not key:
        print("No CLAUDE_API_KEY found in env or .streamlit/secrets.toml")
        return
    url = "https://api.anthropic.com/v1/messages"
    prompt_text = "Convert this: return a JSON with scale 5 and fewer walls"
    candidate_models = [
        "claude-opus-4-8",
        "claude-3.5",
        "claude-3",
        "claude-2.1",
        "claude-2",
        "claude-instant",
        "claude-instant-100k",
    ]

    headers = {
        "Content-Type": "application/json",
        "x-api-key": key,
        "Anthropic-Version": "2023-06-01",
    }

    for model in candidate_models:
        print("\nTrying model:", model)
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ],
            "max_tokens": 300,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                print("STATUS:", resp.status)
                print("HEADERS:", dict(resp.getheaders()))
                print("BODY:\n", body)
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = "<failed to read>"
            print("HTTPError:", e.code, getattr(e, 'reason', ''))
            try:
                print("HEADERS:", dict(e.headers))
            except Exception:
                pass
            print("BODY:\n", err_body)
            # If temperature is deprecated, retry without it
            if "temperature" in err_body and "deprecated" in err_body.lower():
                print("Retrying without temperature...")
                payload.pop("temperature", None)
                data = json.dumps(payload).encode("utf-8")
                req2 = urllib.request.Request(url, data=data, headers=headers, method="POST")
                try:
                    with urllib.request.urlopen(req2, timeout=30) as resp2:
                        body2 = resp2.read().decode("utf-8", errors="replace")
                        print("RETRY STATUS:", resp2.status)
                        print("RETRY HEADERS:", dict(resp2.getheaders()))
                        print("RETRY BODY:\n", body2)
                except urllib.error.HTTPError as e2:
                    try:
                        err_body2 = e2.read().decode("utf-8", errors="replace")
                    except Exception:
                        err_body2 = "<failed to read>"
                    print("RETRY HTTPError:", e2.code)
                    print("RETRY BODY:\n", err_body2)
                except Exception as e2:
                    print("Retry Error:", e2)
        except Exception as e:
            print("Error:", e)


if __name__ == '__main__':
    main()

"""Чтение публичных файлов через Chrome на macOS."""

import base64
import json
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import urlsplit

_CHROME_LOCK = threading.Lock()


def applescript(lines):
    command = ["osascript"]
    for line in lines:
        command.extend(["-e", line])
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def browser_download(url):
    with _CHROME_LOCK:
        return _browser_download(url)


def _browser_download(url):
    if sys.platform != "darwin":
        raise RuntimeError("Резервная загрузка Chrome поддерживается на macOS")
    host = urlsplit(url).hostname or ""
    if not any(
        host == domain or host.endswith("." + domain)
        for domain in ("rosstat.gov.ru", "minfin.gov.ru")
    ):
        raise RuntimeError("Загрузка Chrome ограничена публичными файлами Росстата и Минфина")
    origin = f"https://{host}/"
    selected = applescript(
        [
            'tell application "Google Chrome"',
            "repeat with w in windows",
            "repeat with t in tabs of w",
            f'if URL of t starts with {json.dumps(origin)} then return (id of t as text) & " existing"',
            "end repeat",
            "end repeat",
            "if (count of windows) is 0 then make new window",
            "set w to front window",
            "set previousTab to active tab index of w",
            f"set newTab to make new tab at end of tabs of w with properties {{URL:{json.dumps(origin)}}}",
            "set newID to id of newTab",
            "set active tab index of w to previousTab",
            'return (newID as text) & " created"',
            "end tell",
        ]
    )
    tab_id, mode = selected.split()
    key = "__usdRubFetch_" + uuid.uuid4().hex

    def execute(script):
        return applescript(
            [
                'tell application "Google Chrome"',
                "repeat with w in windows",
                "repeat with t in tabs of w",
                f'if (id of t as text) is "{tab_id}" then return execute t javascript {json.dumps(script, ensure_ascii=False)}',
                "end repeat",
                "end repeat",
                "end tell",
            ]
        )

    try:
        deadline = time.monotonic() + 45
        while mode == "created":
            if execute("location.origin") == origin.rstrip("/"):
                break
            if time.monotonic() > deadline:
                raise RuntimeError("Chrome: страница источника не открылась")
            time.sleep(0.25)
        script = f"""window[{json.dumps(key)}] = {{pending:true}};
fetch({json.dumps(url)}, {{credentials:'omit'}}).then(async r => {{
  const bytes = new Uint8Array(await r.arrayBuffer());
  let binary = '';
  for (let offset=0; offset<bytes.length; offset+=32768)
    binary += String.fromCharCode(...bytes.subarray(offset,offset+32768));
  window[{json.dumps(key)}] = {{status:r.status,url:r.url,content:btoa(binary)}};
}}).catch(e => {{window[{json.dumps(key)}] = {{error:String(e)}};}});
'started';"""
        execute(script)
        while time.monotonic() < deadline:
            raw = execute(f"JSON.stringify(window[{json.dumps(key)}] || {{pending:true}})")
            result = json.loads(raw)
            if result.get("error"):
                raise RuntimeError("Chrome: " + result["error"])
            if not result.get("pending"):
                if result["status"] != 200:
                    raise RuntimeError(f"Chrome: HTTP {result['status']}")
                return base64.b64decode(result["content"], validate=True), result["url"]
            time.sleep(0.25)
        raise RuntimeError("Chrome: превышено время загрузки")
    finally:
        if mode == "created":
            applescript(
                [
                    'tell application "Google Chrome"',
                    "repeat with w in windows",
                    "repeat with t in tabs of w",
                    f'if (id of t as text) is "{tab_id}" then close t',
                    "end repeat",
                    "end repeat",
                    "end tell",
                ]
            )
        else:
            execute(f'delete window[{json.dumps(key)}]; "cleared"')

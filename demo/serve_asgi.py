"""
Minimal pure-Python ASGI server fallback using asyncio.
Used when uvicorn is not available in the environment.
"""

import asyncio
import urllib.parse

async def run_asgi(app, host="0.0.0.0", port=8000):
    async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            request_line = await reader.readline()
            if not request_line:
                writer.close()
                await writer.wait_closed()
                return

            line_str = request_line.decode("utf-8", errors="replace").strip()
            parts = line_str.split(" ")
            if len(parts) < 2:
                writer.close()
                await writer.wait_closed()
                return

            method, raw_path = parts[0], parts[1]
            if "?" in raw_path:
                path, query_string = raw_path.split("?", 1)
            else:
                path, query_string = raw_path, ""

            headers = []
            content_length = 0
            while True:
                line = await reader.readline()
                if not line or line == b"\r\n" or line == b"\n":
                    break
                header_line = line.decode("utf-8", errors="replace").strip()
                if ":" in header_line:
                    k, v = header_line.split(":", 1)
                    k, v = k.strip(), v.strip()
                    headers.append((k.lower().encode("latin1"), v.encode("latin1")))
                    if k.lower() == "content-length":
                        content_length = int(v)

            body = b""
            if content_length > 0:
                body = await reader.readexactly(content_length)

            scope = {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": method,
                "path": urllib.parse.unquote(path),
                "raw_path": path.encode("latin1"),
                "query_string": query_string.encode("latin1"),
                "headers": headers,
                "server": (host, port),
            }

            res_status = 200
            res_headers = []
            res_body = bytearray()

            async def receive():
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False,
                }

            async def send(message):
                nonlocal res_status, res_headers, res_body
                msg_type = message.get("type")
                if msg_type == "http.response.start":
                    res_status = message.get("status", 200)
                    res_headers = message.get("headers", [])
                elif msg_type == "http.response.body":
                    res_body.extend(message.get("body", b""))

            await app(scope, receive, send)

            status_text = {200: "OK", 404: "Not Found", 500: "Internal Server Error"}.get(res_status, "OK")
            response_line = f"HTTP/1.1 {res_status} {status_text}\r\n".encode("latin1")
            writer.write(response_line)

            for hk, hv in res_headers:
                writer.write(hk + b": " + hv + b"\r\n")

            if not any(hk.lower() == b"content-length" for hk, _ in res_headers):
                writer.write(f"content-length: {len(res_body)}\r\n".encode("latin1"))

            writer.write(b"\r\n")
            writer.write(res_body)
            await writer.drain()

        except Exception as e:
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, host, port)
    print(f"\n{'='*60}")
    print(f"  ZebraID Federated Demo (Frontend)")
    print(f"  Coordinator: http://{host}:{port}")
    print(f"{'='*60}\n")
    async with server:
        await server.serve_forever()

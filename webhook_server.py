from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers['content-length'])
        body = self.rfile.read(length)
        print("\n=== ALERT RECEIVED ===")
        print(body.decode())
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'OK')

HTTPServer(('', 8080), Handler).serve_forever()

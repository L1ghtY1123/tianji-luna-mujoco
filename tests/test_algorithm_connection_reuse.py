"""Successive cuRobo calls keep a socket without changing request contents."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from luna_mujoco.algorithms import Algorithms


def test_algorithm_requests_reuse_connection_and_close_it(monkeypatch):
    # Windows registry proxies may otherwise rewrite Connection: close.
    monkeypatch.setenv('NO_PROXY', '127.0.0.1')
    received = []
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            received.append((self.client_address, self.path, body))
            encoded = json.dumps({'ok': True, 'value': body['value']}).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    algorithms = Algorithms()
    algorithms.curobo_url = f'http://127.0.0.1:{server.server_port}'
    try:
        assert algorithms.request('/one', {'value': 17}) == {'ok': True, 'value': 17}
        assert algorithms.request('/two', {'value': 29}) == {'ok': True, 'value': 29}
        assert [row[1:] for row in received] == [('/one', {'value': 17}), ('/two', {'value': 29})]
        assert received[0][0] == received[1][0], 'connection was recreated for each algorithm call'
    finally:
        if hasattr(algorithms, 'close'):
            algorithms.close()
        server.shutdown()
        server.server_close()
        worker.join()

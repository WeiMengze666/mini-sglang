import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest


def _runner():
    path = Path(__file__).resolve().parents[2] / "benchmark/online/run_profile.py"
    spec = importlib.util.spec_from_file_location("online_profile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_profile


@pytest.mark.parametrize("code", [0, 4])
def test_profile_preserves_commands_and_cleans_owned_server(tmp_path, code):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    server = tmp_path / "server.py"
    server.write_text("""import http.server,sys
class Handler(http.server.BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200);self.end_headers();self.wfile.write(b'{"ready":true}')
http.server.HTTPServer(('127.0.0.1',int(sys.argv[1])),Handler).serve_forever()
""")
    bench = tmp_path / "bench.py"
    bench.write_text("""import sys,pathlib,urllib.request
assert urllib.request.urlopen(sys.argv[2]+'/v1/models').status==200
pathlib.Path('bench-argv.json').write_text(__import__('json').dumps(sys.argv))
raise SystemExit(int(sys.argv[3]))
""")
    url = f"http://127.0.0.1:{port}"
    profile = {
        "reference": {"argv": [sys.executable, str(server), str(port)]},
        "benchmark": {"argv": [sys.executable, str(bench), "--url", url, str(code)]},
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    output = tmp_path / "output"
    run = _runner()
    if code:
        with pytest.raises(RuntimeError, match="AIPerf exited with 4"):
            run(path, "reference", output, readiness_timeout=10, benchmark_timeout=10)
    else:
        result = run(path, "reference", output, readiness_timeout=10, benchmark_timeout=10)
        assert result["status"] == "completed"
    saved = json.loads((output / "run.json").read_text())
    assert saved["server_argv"] == profile["reference"]["argv"]
    assert saved["benchmark_argv"] == profile["benchmark"]["argv"]
    assert saved["benchmark_returncode"] == code
    assert saved["status"] == ("failed" if code else "completed")
    assert (output / "metrics-before.txt").exists() and (output / "metrics-after.txt").exists()
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", port)) != 0

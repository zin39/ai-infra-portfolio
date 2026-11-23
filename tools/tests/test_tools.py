import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bench_sweep  # noqa: E402
import vram_calc  # noqa: E402


# ---------- vram_calc ----------

def test_weights_8b_fp16_is_16gb():
    assert vram_calc.weights_gb(8, 16) == pytest.approx(16.0)


def test_kv_per_token_llama3_8b():
    # 32 layers, 8 KV heads, head_dim 128, fp16 -> 128 KiB per token
    assert vram_calc.kv_bytes_per_token(32, 8, 128) == 131072


def test_fp8_kv_halves_kv_size():
    assert vram_calc.kv_bytes_per_token(32, 8, 128, 8) == 65536


def test_decode_ceiling_matches_measured_ratio():
    # 8B fp16 on a 1792 GB/s card -> ~112 tok/s ceiling (I measured 82.7 = 74% of it)
    r = vram_calc.plan(8, 16, 32, 8, 128, vram_gb=96, bandwidth_gbs=1792)
    assert r.decode_ceiling_tps == pytest.approx(112, rel=0.01)
    assert 0.7 < 82.7 / r.decode_ceiling_tps < 0.8


def test_moe_ceiling_uses_active_params():
    dense = vram_calc.plan(30, 4, 48, 4, 128, 96, 1792)
    moe = vram_calc.plan(30, 4, 48, 4, 128, 96, 1792, active_params_b=3)
    assert moe.weights_gb == dense.weights_gb
    assert moe.decode_ceiling_tps == pytest.approx(dense.decode_ceiling_tps * 10)


def test_model_that_does_not_fit():
    r = vram_calc.plan(70, 16, 80, 8, 128, vram_gb=48, bandwidth_gbs=864)
    assert r.max_kv_tokens == 0 and r.kv_budget_gb == 0


def test_bad_util_rejected():
    with pytest.raises(ValueError):
        vram_calc.plan(8, 16, 32, 8, 128, 96, 1792, util=1.5)


# ---------- bench_sweep against a fake streaming server ----------

TOKENS = 20
DELAY = 0.005


class FakeOpenAI(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(TOKENS):
            time.sleep(DELAY)
            chunk = {"choices": [{"delta": {"content": f"t{i} "}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):
        pass


@pytest.fixture()
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_single_request_counts_tokens(server):
    r = bench_sweep.one_request(server, "m", "hi", 50)
    assert r.tokens == TOKENS
    assert 0 < r.ttft_s < r.total_s
    assert r.decode_tps > 0


def test_sweep_reports_worst_user(server):
    rows = bench_sweep.sweep(server, "m", [1, 4], 50, "hi")
    assert [row["concurrency"] for row in rows] == [1, 4]
    for row in rows:
        assert row["decode_worst_tps"] <= row["decode_avg_tps"]
        assert row["aggregate_tps"] > 0
        assert row["errors"] == 0
    # 4 users in parallel should push more aggregate tokens/s than 1
    assert rows[1]["aggregate_tps"] > rows[0]["aggregate_tps"]


def test_decode_tps_edge_cases():
    assert bench_sweep.Result(1.0, 1.0, 1).decode_tps == 0.0
    assert bench_sweep.Result(1.0, 2.0, 11).decode_tps == pytest.approx(10.0)


def test_unreachable_server_counts_errors_instead_of_crashing():
    rows = bench_sweep.sweep("http://127.0.0.1:9", "m", [2], 5, "hi")
    assert rows == [{"concurrency": 2, "n": 0, "errors": 2}]


def test_cli_prints_plan(capsys):
    vram_calc.main(["--params", "8", "--layers", "32", "--kv-heads", "8", "--head-dim", "128",
                    "--vram-gb", "96", "--bandwidth-gbs", "1792"])
    out = capsys.readouterr().out
    assert "16.00 GB" in out and "112.0 tok/s" in out

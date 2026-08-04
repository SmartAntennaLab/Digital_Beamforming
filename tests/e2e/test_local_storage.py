"""Verify real browser-local settings persistence across page sessions."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from device_settings import DEVICE_STORAGE_KEY

try:
    from playwright.sync_api import expect, sync_playwright
except ImportError:  # Unit-test environments do not need browser dependencies.
    expect = None
    sync_playwright = None


ROOT = Path(__file__).resolve().parents[2]
RUN_E2E = os.environ.get("RUN_E2E") == "1"


def _unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@unittest.skipUnless(
    RUN_E2E and sync_playwright is not None,
    "Set RUN_E2E=1 and install requirements-e2e.txt to run browser tests.",
)
class DeviceLocalStorageE2ETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.port = _unused_local_port()
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ROOT / "main.py"),
            "--server.headless=true",
            f"--server.port={cls.port}",
            "--browser.gatherUsageStats=false",
        ]
        creation_flags = (
            subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        )
        cls.server = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
        )
        health_url = f"{cls.base_url}/_stcore/health"
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if cls.server.poll() is not None:
                raise RuntimeError("Streamlit exited before the E2E test started.")
            try:
                with urlopen(health_url, timeout=1.0) as response:
                    if response.status == 200:
                        return
            except (OSError, URLError):
                time.sleep(0.2)
        raise TimeoutError("Timed out waiting for the Streamlit health endpoint.")

    @classmethod
    def tearDownClass(cls) -> None:
        server = getattr(cls, "server", None)
        if server is None or server.poll() is not None:
            return
        server.terminate()
        try:
            server.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5.0)

    def _expect_frequency(self, page, value: float) -> None:
        slider = page.get_by_role("slider", name="주파수 (GHz)")
        expect(slider).to_have_value(f"{value:g}", timeout=20_000)

    def test_save_new_session_and_refresh_restore_local_settings(self) -> None:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            context = browser.new_context()
            source_page = context.new_page()
            source_page.goto(
                f"{self.base_url}/?frequency_ghz=35",
                wait_until="networkidle",
            )
            self._expect_frequency(source_page, 35.0)
            source_page.get_by_role(
                "button", name="설정 적용 및 계산"
            ).click()
            source_page.wait_for_function(
                """key => {
                    const value = window.localStorage.getItem(key)
                    if (!value) return false
                    try {
                      return JSON.parse(value)?.settings?.frequency_ghz === 35
                    } catch (_) {
                      return false
                    }
                }""",
                arg=DEVICE_STORAGE_KEY,
                timeout=20_000,
            )

            restored_page = context.new_page()
            restored_page.goto(self.base_url, wait_until="networkidle")
            self._expect_frequency(restored_page, 35.0)
            restored_page.reload(wait_until="networkidle")
            self._expect_frequency(restored_page, 35.0)
            browser.close()

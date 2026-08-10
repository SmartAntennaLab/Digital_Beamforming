import json
import tomllib
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DeploymentConfigurationTests(unittest.TestCase):
    def test_project_dependencies_are_pinned_and_grouped(self):
        project = tomllib.loads(
            (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]
        self.assertEqual(project["requires-python"], ">=3.11,<3.15")
        self.assertEqual(
            project["dependencies"],
            [
                "numpy==2.3.5",
                "pandas==2.3.3",
                "plotly==6.9.0",
                "psutil==7.2.2",
                "streamlit==1.60.0",
            ],
        )
        self.assertEqual(
            project["optional-dependencies"],
            {
                "dev": [
                    "pip==26.2",
                    "pip-audit==2.10.1",
                    "pip-licenses==5.5.5",
                    "setuptools==83.0.0",
                ],
                "e2e": ["playwright==1.61.0"],
                "quality": [
                    "coverage==7.15.4",
                    "mypy==2.3.0",
                    "ruff==0.16.2",
                ],
            },
        )
        self.assertFalse((PROJECT_ROOT / "requirements.txt").exists())
        self.assertFalse((PROJECT_ROOT / "requirements-e2e.txt").exists())

    def test_uv_lock_is_universal_and_hashes_registry_artifacts(self):
        lock = tomllib.loads((PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8"))
        self.assertEqual(lock["requires-python"], ">=3.11, <3.15")
        packages = lock["package"]
        project = next(
            package
            for package in packages
            if package["name"] == "digital-beamforming-simulator"
        )
        self.assertEqual(project["version"], "1.5.0")
        self.assertEqual(
            set(project["optional-dependencies"]), {"dev", "e2e", "quality"}
        )
        for package in packages:
            if "registry" not in package.get("source", {}):
                continue
            artifacts = (
                [package["sdist"]] if "sdist" in package else []
            ) + package.get("wheels", [])
            self.assertTrue(artifacts, package["name"])
            self.assertTrue(
                all(artifact["hash"].startswith("sha256:") for artifact in artifacts),
                package["name"],
            )

    def test_quality_gates_are_pinned_and_enforced(self):
        project_config = tomllib.loads(
            (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        tool_config = project_config["tool"]
        self.assertEqual(
            tool_config["ruff"]["lint"]["select"],
            ["E4", "E7", "E9", "F", "I", "B", "UP"],
        )
        self.assertEqual(tool_config["mypy"]["python_version"], "3.11")
        self.assertIn("directivity.py", tool_config["mypy"]["files"])
        self.assertTrue(tool_config["coverage"]["run"]["branch"])
        self.assertEqual(tool_config["coverage"]["report"]["fail_under"], 80)

        performance_gate = (
            PROJECT_ROOT / "benchmarks" / "check_performance_regression.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"DBF_PERF_EXACT_64_SECONDS", 5.0', performance_gate)
        self.assertIn('"DBF_PERF_FAST_64_SECONDS", 6.0', performance_gate)
        self.assertIn("> 0.5", performance_gate)

    def test_devcontainer_syncs_only_from_the_frozen_lock(self):
        config = json.loads(
            (PROJECT_ROOT / ".devcontainer" / "devcontainer.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertNotIn("updateContentCommand", config)
        install_command = config["postCreateCommand"]
        self.assertEqual(install_command.count("uv sync --frozen"), 1)
        self.assertIn("uv==0.11.29", install_command)
        self.assertIn("--extra dev", install_command)
        self.assertNotIn("requirements.txt", install_command)
        self.assertNotIn("install --user", install_command)

    def test_devcontainer_does_not_disable_web_protection(self):
        config = json.loads(
            (PROJECT_ROOT / ".devcontainer" / "devcontainer.json").read_text(
                encoding="utf-8"
            )
        )
        run_command = config["postAttachCommand"]["server"]
        self.assertEqual(
            run_command,
            "uv run --frozen --no-sync streamlit run main.py --server.headless=true",
        )
        self.assertNotIn("enableCORS", run_command)
        self.assertNotIn("enableXsrfProtection", run_command)

    def test_streamlit_config_is_headless_and_keeps_web_protection(self):
        config = tomllib.loads(
            (PROJECT_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
        )
        server = config["server"]
        self.assertTrue(server["headless"])
        self.assertEqual(server["port"], 8501)
        self.assertTrue(server["enableCORS"])
        self.assertTrue(server["enableXsrfProtection"])

    def test_ci_covers_locked_cross_platform_and_dependency_health(self):
        ci = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        dependency_health = (
            PROJECT_ROOT / ".github" / "workflows" / "dependency-health.yml"
        ).read_text(encoding="utf-8")
        dependabot = (PROJECT_ROOT / ".github" / "dependabot.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("ubuntu-latest", ci)
        self.assertIn("windows-latest", ci)
        self.assertIn('python-version: ["3.11", "3.14"]', ci)
        self.assertIn("uv sync --frozen", ci)
        self.assertIn("uv pip check", ci)
        self.assertIn("dependency-health.yml", ci)
        self.assertIn("quality-gates:", ci)
        self.assertIn("ruff check .", ci)
        self.assertIn("mypy", ci)
        self.assertIn("coverage run", ci)
        self.assertIn("coverage report", ci)
        self.assertIn("check_performance_regression.py", ci)

        self.assertIn("schedule:", dependency_health)
        self.assertIn("--all-extras", dependency_health)
        self.assertIn("python -m pip check", dependency_health)
        self.assertIn("python -m pip_audit", dependency_health)
        self.assertIn("python -m piplicenses", dependency_health)
        self.assertIn("GPL;AGPL;UNKNOWN", dependency_health)
        combined_workflows = ci + dependency_health
        self.assertNotIn("actions/checkout@v", combined_workflows)
        self.assertNotIn("actions/setup-python@v", combined_workflows)

        self.assertIn('package-ecosystem: "uv"', dependabot)
        self.assertIn('package-ecosystem: "github-actions"', dependabot)
        self.assertIn('interval: "weekly"', dependabot)

    def test_public_server_examples_require_auth_rate_limits_and_quotas(self):
        nginx = (PROJECT_ROOT / "deploy" / "nginx.conf.example").read_text(
            encoding="utf-8"
        )
        service = (
            PROJECT_ROOT / "deploy" / "digital-beamforming.service.example"
        ).read_text(encoding="utf-8")
        deployment_guide = (PROJECT_ROOT / "deploy" / "README.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("auth_basic_user_file", nginx)
        self.assertIn("limit_req_zone", nginx)
        self.assertIn("limit_conn_zone", nginx)
        self.assertIn("proxy_set_header Upgrade", nginx)
        self.assertIn("CPUQuota=", service)
        self.assertIn("MemoryMax=", service)
        self.assertIn("compute_health", deployment_guide)
        self.assertIn("DBF_MAX_CONCURRENT_CALCULATIONS", deployment_guide)


if __name__ == "__main__":
    unittest.main()

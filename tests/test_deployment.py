import json
from pathlib import Path
import tomllib
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DeploymentConfigurationTests(unittest.TestCase):
    def test_direct_dependencies_are_pinned_once(self):
        requirements = (
            PROJECT_ROOT / "requirements.txt"
        ).read_text(encoding="utf-8").splitlines()
        self.assertEqual(
            requirements,
            [
                "streamlit==1.60.0",
                "numpy==2.3.5",
                "plotly==6.9.0",
                "pandas==2.3.3",
            ],
        )

    def test_devcontainer_installs_only_from_requirements(self):
        config = json.loads(
            (PROJECT_ROOT / ".devcontainer" / "devcontainer.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertNotIn("updateContentCommand", config)
        install_command = config["postCreateCommand"]
        self.assertEqual(install_command.count("install -r requirements.txt"), 1)
        self.assertNotIn("install --user", install_command)
        self.assertNotIn("install streamlit", install_command)

    def test_devcontainer_does_not_disable_web_protection(self):
        config = json.loads(
            (PROJECT_ROOT / ".devcontainer" / "devcontainer.json").read_text(
                encoding="utf-8"
            )
        )
        run_command = config["postAttachCommand"]["server"]
        self.assertEqual(
            run_command,
            "python -m streamlit run main.py --server.headless=true",
        )
        self.assertNotIn("enableCORS", run_command)
        self.assertNotIn("enableXsrfProtection", run_command)

    def test_streamlit_config_keeps_cors_and_xsrf_enabled(self):
        config = tomllib.loads(
            (PROJECT_ROOT / ".streamlit" / "config.toml").read_text(
                encoding="utf-8"
            )
        )
        server = config["server"]
        self.assertFalse(server["headless"])
        self.assertEqual(server["port"], 8501)
        self.assertTrue(server["enableCORS"])
        self.assertTrue(server["enableXsrfProtection"])


if __name__ == "__main__":
    unittest.main()

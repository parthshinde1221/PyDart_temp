from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"
# BUILTIN_OUTPUTS_DIR = OUTPUTS_DIR / "built_in"
CUSTOM_OUTPUTS_DIR = REPO_ROOT / "outputs_custom"
# LOGS_DIR = OUTPUTS_DIR / "logs"
OUTPUTS_DIR_RUN = REPO_ROOT / "outputs_built_in_run"


# print(REPO_ROOT,OUTPUTS_DIR,CUSTOM_OUTPUTS_DIR)
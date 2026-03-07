# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-demand: Test Runner

# COMMAND ----------

# Install only what is not pre-installed on Databricks.
# Critically: do NOT reinstall scikit-learn or numpy — they conflict with
# Databricks' pre-installed pyarrow when reinstalled via pip in a serverless env.
# MAGIC %pip install catboost doubleml lifelines pytest

# COMMAND ----------

import subprocess
import sys
import os

# Clone the repo to get test files
clone = subprocess.run(
    ["git", "clone", "--depth=1", "https://github.com/burningcost/insurance-demand.git", "/tmp/insurance-demand"],
    capture_output=True, text=True
)
print("Clone:", clone.returncode, clone.stderr[:200] if clone.returncode != 0 else "OK")

# COMMAND ----------

# Install the library itself (editable, uses already-installed sklearn/numpy)
install = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/tmp/insurance-demand",
     "--no-deps"],  # no-deps: don't reinstall scipy/numpy/sklearn from PyPI
    capture_output=True, text=True
)
print("Install:", install.returncode)
print(install.stdout[-1000:] if install.stdout else "")
print(install.stderr[-500:] if install.stderr else "")

# COMMAND ----------

env = {**os.environ, "PYTHONPATH": "/tmp/insurance-demand/src"}

result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/tmp/insurance-demand/tests/",
        "-v", "--tb=short", "--no-header",
    ],
    capture_output=True, text=True,
    cwd="/tmp/insurance-demand",
    env=env,
)

full_output = result.stdout + "\n--- STDERR ---\n" + result.stderr
display_output = full_output[-10000:] if len(full_output) > 10000 else full_output
print(display_output)

# COMMAND ----------

exit_message = f"returncode={result.returncode}\n" + display_output[-4000:]
dbutils.notebook.exit(exit_message)

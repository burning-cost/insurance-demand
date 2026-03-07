# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-demand: Test Runner

# COMMAND ----------

# MAGIC %pip install "git+https://github.com/burningcost/insurance-demand.git" \
# MAGIC   catboost doubleml lifelines matplotlib pytest

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

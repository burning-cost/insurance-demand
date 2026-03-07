# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-demand: Test Runner
# MAGIC
# MAGIC Key constraint: do NOT reinstall scikit-learn, numpy, scipy, or pyarrow.
# MAGIC Databricks serverless pre-installs these against its own pyarrow version.
# MAGIC Reinstalling from PyPI causes an `_ARRAY_API not found` AttributeError.
# MAGIC
# MAGIC Strategy: install polars, catboost, doubleml, lifelines, and pytest explicitly,
# MAGIC then install the library with --no-deps.

# COMMAND ----------

# MAGIC %pip install polars catboost doubleml lifelines pytest statsmodels

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

# Install the library itself without reinstalling sklearn/numpy/scipy/pandas
install = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/tmp/insurance-demand",
     "--no-deps"],
    capture_output=True, text=True
)
print("Install:", install.returncode)
print(install.stdout[-500:] if install.stdout else "")
if install.returncode != 0:
    print(install.stderr[-500:])

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

# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-demand: Test Runner
# MAGIC
# MAGIC Runs tests using the Databricks system Python (/databricks/python) which
# MAGIC has a compatible version of pyarrow and scikit-learn pre-installed.

# COMMAND ----------

import subprocess
import sys
import os

SYSTEM_PIP = "/databricks/python/bin/pip"
SYSTEM_PYTHON = "/databricks/python/bin/python3"

# Install everything into the system Python (which has the correct pyarrow)
packages = ["polars", "catboost", "doubleml", "lifelines", "statsmodels", "pytest"]
install = subprocess.run(
    [SYSTEM_PIP, "install"] + packages + ["--quiet"],
    capture_output=True, text=True
)
print("Install exit code:", install.returncode)
if install.returncode != 0:
    print(install.stderr[-2000:])
else:
    print("OK")

# COMMAND ----------

# Clone repo
clone = subprocess.run(
    ["git", "clone", "--depth=1", "https://github.com/burningcost/insurance-demand.git", "/tmp/insurance-demand"],
    capture_output=True, text=True
)
print("Clone:", clone.returncode, clone.stderr[:200] if clone.returncode != 0 else "OK")

# Install library (no-deps — sklearn/numpy/scipy already in system Python)
lib_install = subprocess.run(
    [SYSTEM_PIP, "install", "-e", "/tmp/insurance-demand", "--no-deps", "--quiet"],
    capture_output=True, text=True
)
print("Library install:", lib_install.returncode)
if lib_install.returncode != 0:
    print(lib_install.stderr[-500:])

# COMMAND ----------

env = {**os.environ, "PYTHONPATH": "/tmp/insurance-demand/src"}

result = subprocess.run(
    [SYSTEM_PYTHON, "-m", "pytest",
     "/tmp/insurance-demand/tests/",
     "-v", "--tb=short", "--no-header"],
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

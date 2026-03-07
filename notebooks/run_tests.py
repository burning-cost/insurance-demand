# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-demand: Test Runner
# MAGIC
# MAGIC Note on the environment: Databricks serverless uses an ephemeral venv for
# MAGIC packages installed via %pip. The venv's scikit-learn conflicts with the
# MAGIC pre-installed pyarrow. Solution: reinstall pyarrow AFTER other packages
# MAGIC to get a compatible version, or exclude doubleml/statsmodels from the
# MAGIC install path used for tests (they're optional for core tests).
# MAGIC
# MAGIC The core tests (datasets, conversion, retention, demand_curve, optimiser,
# MAGIC compliance) only require: polars, catboost, scipy, statsmodels, sklearn, pandas.
# MAGIC They do NOT require doubleml or econml. We run those separately.

# COMMAND ----------

# Install packages. Note: we install pyarrow last (after any sklearn upgrade)
# to ensure a compatible version is in place.
# MAGIC %pip install polars catboost lifelines statsmodels pytest

# COMMAND ----------

# Reinstall pyarrow to get a version compatible with the freshly installed sklearn
# MAGIC %pip install "pyarrow>=14.0" --upgrade --quiet

# COMMAND ----------

import subprocess
import sys
import os

# Clone the repo
clone = subprocess.run(
    ["git", "clone", "--depth=1",
     "https://github.com/burningcost/insurance-demand.git",
     "/tmp/insurance-demand"],
    capture_output=True, text=True
)
print("Clone:", clone.returncode, clone.stderr[:200] if clone.returncode != 0 else "OK")

# COMMAND ----------

# Install library without reinstalling deps (use the venv's sklearn/pandas/scipy)
lib_install = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e",
     "/tmp/insurance-demand", "--no-deps", "--quiet"],
    capture_output=True, text=True
)
print("Library install:", lib_install.returncode)
if lib_install.returncode != 0:
    print(lib_install.stderr[-500:])

# COMMAND ----------

env = {**os.environ, "PYTHONPATH": "/tmp/insurance-demand/src"}

result = subprocess.run(
    [sys.executable, "-m", "pytest",
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

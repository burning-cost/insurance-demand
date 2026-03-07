# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-demand: Test Runner
# MAGIC
# MAGIC Uses the Databricks system Python (not the ephemeral venv) to avoid the
# MAGIC pyarrow/_ARRAY_API conflict that occurs when scikit-learn is reinstalled
# MAGIC via %pip in a serverless environment.

# COMMAND ----------

import subprocess
import sys
import os

# Identify the system Python (not the ephemeral venv pip uses)
# On Databricks serverless, /databricks/python/bin/python3 is the stable interpreter
system_pythons = [
    "/databricks/python/bin/python3",
    "/databricks/python3/bin/python3",
    "/usr/bin/python3",
]
system_python = None
for p in system_pythons:
    check = subprocess.run([p, "--version"], capture_output=True, text=True)
    if check.returncode == 0:
        system_python = p
        print(f"Using system Python: {p} ({check.stdout.strip()})")
        break

if system_python is None:
    system_python = sys.executable
    print(f"Falling back to current Python: {system_python}")

# COMMAND ----------

# Install packages into the system Python (avoids the venv/pyarrow conflict)
packages = ["polars", "catboost", "doubleml", "lifelines", "statsmodels", "pytest"]
install = subprocess.run(
    [system_python, "-m", "pip", "install"] + packages + ["--quiet"],
    capture_output=True, text=True
)
print("Install exit code:", install.returncode)
if install.returncode != 0:
    print(install.stderr[-1000:])
else:
    print("Packages installed OK")

# COMMAND ----------

# Clone the repo to get test files
clone = subprocess.run(
    ["git", "clone", "--depth=1", "https://github.com/burningcost/insurance-demand.git", "/tmp/insurance-demand"],
    capture_output=True, text=True
)
print("Clone:", clone.returncode, clone.stderr[:200] if clone.returncode != 0 else "OK")

# COMMAND ----------

# Install library itself (no-deps: sklearn/numpy/scipy/pandas already present)
lib_install = subprocess.run(
    [system_python, "-m", "pip", "install", "-e", "/tmp/insurance-demand", "--no-deps", "--quiet"],
    capture_output=True, text=True
)
print("Library install:", lib_install.returncode)
if lib_install.returncode != 0:
    print(lib_install.stderr[-500:])

# COMMAND ----------

env = {**os.environ, "PYTHONPATH": "/tmp/insurance-demand/src"}

result = subprocess.run(
    [
        system_python, "-m", "pytest",
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

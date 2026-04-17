## virtualenv

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cu124
make run
```

## uv

```bash
# uv init --python 3.12
# uv sync
uv run python setup.py build_ext --inplace
uv run test.py
```

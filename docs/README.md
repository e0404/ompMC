# Building the documentation locally

No C++ compiler needed: the compiled `_ompmc` extension is mocked out (see
`conf.py`), so this only needs Python, Doxygen, and the packages in
`requirements.txt`.

```sh
# Doxygen: apt install doxygen / brew install doxygen / choco install doxygen.strawberry
python -m venv .venv-docs
.venv-docs/Scripts/activate        # .venv-docs/bin/activate on Linux/macOS
pip install -r docs/requirements.txt

sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`. Rebuild after an edit with the same
command; add `-E` to force a full rebuild (e.g. after editing `conf.py` or
`Doxyfile`) or `-W -n --keep-going` to build the way CI does, which turns
warnings and broken cross-references into a failing build.

`sphinx-autobuild docs docs/_build/html` (from the `sphinx-autobuild`
package, not in `requirements.txt`) rebuilds and reloads a browser tab on
every save, useful while editing prose.

# Installation

`bii` requires Python ≥ 3.10 and is built on [JAX](https://jax.readthedocs.io).

## From source

```bash
pip install git+https://github.com/mascaretti/bii.git
```

## Development install

```bash
git clone https://github.com/mascaretti/bii.git
cd bii
pip install -e ".[dev]"      # adds pytest, hypothesis, ruff, mypy
```

## GPU / accelerators

The default dependency pulls a CPU build of JAX, which is enough for small
problems and for running the test suite. NUTS over many triplets or a
high-dimensional weight vector is much faster on a GPU; install the matching
accelerator build of JAX for your platform, for example:

```bash
pip install -U "jax[cuda12]"   # match your CUDA toolkit
```

Verify the device JAX sees:

```python
import jax; print(jax.devices())
```

## Building the docs locally

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

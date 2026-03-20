# WellConfiguration

Julia code and notebook examples for well-configuration density estimation in stratigraphic modeling.

Paper:

- https://link.springer.com/article/10.1007/s11004-024-10144-7

![Julia](https://img.shields.io/badge/Julia-1.10%2B-9558B2?logo=julia&logoColor=white)
![Notebook](https://img.shields.io/badge/workflow-Julia%20%2B%20Jupyter-F2A65A)

```mermaid
flowchart LR
    A["Prior bedset surfaces"] --> B["Well intersections"]
    B --> C["Configuration likelihood"]
    C --> D["Posterior density estimate"]
```

## Quickstart

Install the Julia dependencies:

```bash
julia setup.jl
```

Then open the notebook:

```bash
jupyter lab
```

and run [`example.ipynb`](./example.ipynb).

## Repository Layout

- `functions.jl`: core Julia implementation for simulating surfaces, extracting well configurations, and computing configuration probabilities.
- `example.ipynb`: worked example showing how to reproduce the workflow.
- `setup.jl`: convenience script for installing the required Julia packages.

## Reproducibility Notes

- The notebook is the recommended entry point.
- `functions.jl` is self-contained and can also be included directly from a Julia REPL or script.
- `setup.jl` installs `IJulia` so the notebook workflow is easy to reproduce on a fresh machine.

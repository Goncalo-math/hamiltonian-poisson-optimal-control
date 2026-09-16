# Hamiltonian Poisson Integrators for Optimal Control

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Numerical experiments comparing a second-order **Hamiltonian Poisson integrator (HPI)** with an explicit second-order **Runge-Kutta method (RK2)** on optimal-control problems derived from Pontryagin's principle.

The repository accompanies the paper **“Symplectic Groupoids in Optimal Control Problems: A Numerical Study of Hamiltonian Poisson Integrators”** by Gonçalo Inocêncio Oliveira.

## Project overview

Pontryagin's principle converts an optimal-control problem into a Hamiltonian boundary-value problem for the state and costate. This project investigates whether a groupoid-derived, structure-preserving integrator offers better Hamiltonian behavior than a standard explicit RK2 scheme.

| Example | Objective | Control structure | Script |
| --- | --- | --- | --- |
| Pendulum swing-up | Minimize control energy | Smooth | [`pendulum.py`](examples/pendulum.py) |
| Lunar soft landing | Minimize fuel use | Coast-then-burn | [`moon_landing.py`](examples/moon_landing.py) |
| Insect allocation | Maximize terminal output | Bang-bang | [`insect_allocation.py`](examples/insect_allocation.py) |
| Ferry navigation | Minimize crossing time | Smooth, free final time | [`ferry_navigation.py`](examples/ferry_navigation.py) |
| Fish harvesting | Maximize profit | Bang-bang | [`harvesting.py`](examples/harvesting.py) |

At the meshes reported in the paper, both methods recover the same qualitative controls and satisfy the endpoint conditions. HPI produces the clearest reduction in Hamiltonian drift for the pendulum and insect-allocation examples, smaller improvements for lunar landing and harvesting, and results comparable to RK2 for the ferry problem.

## Repository contents

```text
.
├── docs/paper.pdf
├── examples/
│   ├── ferry_navigation.py
│   ├── harvesting.py
│   ├── insect_allocation.py
│   ├── moon_landing.py
│   └── pendulum.py
├── notebooks/project_overview.ipynb
├── CITATION.cff
├── LICENSE
├── README.md
└── requirements.txt
```

The [project notebook](notebooks/project_overview.ipynb) gives a guided introduction to the formulation, methods, experiments, results, and limitations. The [full paper](docs/paper.pdf) contains the derivations and detailed discussion.

## Installation

```bash
git clone <your-repository-url>
cd hamiltonian-poisson-optimal-control
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Usage

Run an experiment from the repository root:

```bash
python examples/pendulum.py
```

Or open the explanatory notebook:

```bash
jupyter lab notebooks/project_overview.ipynb
```

Each experiment prints the shooting solution and displays Matplotlib figures comparing HPI and RK2.

## Selected results

| Example | HPI | RK2 |
| --- | ---: | ---: |
| Pendulum control cost | 1.21977 | 1.08924 |
| Lunar fuel used | 1222.86 kg | 1222.86 kg |
| Ferry optimal time | 3.33464 | 3.33441 |
| Harvesting profit | 4796.63 | 4796.65 |

These are fixed-mesh numerical results, not proof that either method is universally more accurate or computationally cheaper.

## Limitations and future work

All five Pontryagin systems evolve on canonical cotangent spaces. The experiments therefore test the canonical symplectic limit of the groupoid construction, not its distinctive ability to preserve noncanonical symplectic leaves and Casimir invariants.

A natural continuation is a mesh-refinement study on a genuinely noncanonical Poisson control problem, comparing objective error, endpoint residuals, execution time, Hamiltonian drift, Casimir drift, and symplectic-leaf error.

## Citation

If this project supports your research, use the metadata in [`CITATION.cff`](CITATION.cff). GitHub will display a **Cite this repository** button automatically.

## License

The source code is released under the [MIT License](LICENSE). The included paper remains attributable to its named author.

# stopro — Elementary Stochastic Processes

**stopro** is a small Python library for generating and simulating common
(multivariate) stochastic processes.

Currently included processes:

1. Wiener process  
2. Ornstein–Uhlenbeck process  
3. Integrated Ornstein–Uhlenbeck process  
4. Exponential Ornstein–Uhlenbeck process  
5. Geometric Brownian Motion  
6. Colored Geometric Brownian Motion  
7. Gillespie Replicator  
8. Kimura Replicator  
9. White Replicator  
10. Colored Replicator  
11. Multispecies Moran process  
    (discrete particle kinetics & diffusion approximation)  
12. Competitive Lotka–Volterra process  
    (discrete particle kinetics & diffusion approximation)

Examples and documentation are provided as Jupyter notebooks.

---

## Quick start (recommended)

This project uses **uv**, a fast Python package manager and virtual environment
tool. You need to install uv on your system first. When that's done:

```bash
git clone https://github.com/dirkbrockmann/stopro.git
cd stopro
make notebook
```

---

## Using stopro in your own code

```python
import stopro
```

---



## Installation via pip

You can now install stopro directly from PyPI:

```bash
pip install stopro
```

To install the latest development version from source:

```bash
pip install git+https://github.com/dirkbrockmann/stopro.git
```


Or clone and work with the code locally:

```bash
git clone https://github.com/dirkbrockmann/stopro.git
cd stopro
pip install -e .
```

---

## Running the examples

To run the Jupyter notebook examples, install stopro with the extra dependencies:

```bash
pip install "stopro[examples]"
```

This will install all required packages for the example notebooks.

## Running the benchmarks

To run the Jupyter notebook examples, install stopro with the extra dependencies:

```bash
pip install "stopro[bench]"
```

This will install all required packages for the example notebooks.

---

## License

MIT

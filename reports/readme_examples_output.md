# README 示例输出

## Materials and VQE example
### Material summary
```json
[
  {
    "name": "Fermi-Hubbard 2x2",
    "tags": [
      "hubbard",
      "strongly-correlated",
      "benchmark"
    ],
    "citation": "Phasecraft demo circuits (private communication)",
    "parameters": {
      "lattice_size": 4,
      "u": 4.0,
      "t": 1.0
    }
  },
  {
    "name": "FeSe monolayer",
    "tags": [
      "superconductor",
      "high-Tc"
    ],
    "citation": "Tencent News (2025)",
    "parameters": {
      "Tc": 65.0,
      "doping": 0.12
    }
  },
  {
    "name": "LiH minimal basis",
    "tags": [
      "molecular",
      "variational",
      "benchmarks"
    ],
    "citation": "Tencent News (2025). Phasecraft announces hardware-agnostic quantum materials platform.",
    "parameters": {
      "electrons": 2,
      "orbitals": 4
    }
  }
]
```
### LowDepthVQE final energy
`-1.000000000000`
### Density expectation
`0.0`
### Registry low_depth_vqe instantiation
`True`

## Built-in algorithm demonstrations
### Grover search
Most likely state: `5`
Probabilities:
```
[0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.9453125,
 0.0078125, 0.0078125]
```
### Quantum Fourier Transform
Transformed amplitudes:
```
[0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j]
```
Inverse-transformed amplitudes:
```
[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]
```
### Quantum Phase Estimation
Estimated binary phase: `010`
Estimated numeric phase: `0.25`

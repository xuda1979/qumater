# README 示例输出

## 材料与 VQE 示例
### 材料摘要
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
### LowDepthVQE 最终能量
`-1.000000000000`
### 密度矩阵期望值
`0.0`
### 注册表 low_depth_vqe 实例化结果
`True`

## 内置算法示例
### Grover 搜索
最可能的计算基态：`5`
概率分布：
```
[0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.9453125,
 0.0078125, 0.0078125]
```
### 量子傅里叶变换
变换后的振幅：
```
[0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j]
```
逆变换后的振幅：
```
[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]
```
### 量子相位估计
估计得到的二进制相位：`010`
对应的数值相位：`0.25`

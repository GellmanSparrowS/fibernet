# FiberNet - 纤维网络结构研究工具包

## 总目标
构建一个完整的、Nature Materials级别的纤维网络结构研究Python工具包，涵盖：
- 生成工具：2D/3D有序/无序/超结构/压纽/手性/纤维束/编织等
- 模拟工具：结构变形、应力应变、动力学、断裂、热力学、电磁等
- 分析工具：结构表征、拓扑分析、性能指标
- 可视化工具：3D渲染、动画、交互式展示
- 最终作为pip包发布，提升学术影响力

## 包名: fibernet
## GitHub: https://github.com (待创建)
## 主页: https://ml-biomat.com/

## 技术栈
- NumPy/SciPy: 核心数学
- Taichi: GPU加速模拟
- NetworkX: 图/拓扑分析
- PyVista: 3D可视化
- Matplotlib: 2D绘图
- scikit-learn: 结构分类/特征提取

## 架构
```
fibernet/
├── core/       # 核心数据结构 (Fiber, Network, Material)
├── gen/        # 网络生成器
│   ├── ordered.py      # 有序结构 (晶格, 周期)
│   ├── disordered.py   # 无序结构 (随机沉积, 交联)
│   ├── chiral.py       # 手性结构 (螺旋, 扭转)
│   ├── bundle.py       # 纤维束/纱线
│   ├── woven.py        # 编织结构
│   ├── hierarchical.py # 层次/超结构
│   └── lattice.py      # 晶格模板
├── sim/        # 模拟引擎
│   ├── mechanical.py   # 力学 (FEM, 梁理论)
│   ├── dynamics.py     # 动力学 (MD, Brownian)
│   ├── fracture.py     # 断裂力学
│   ├── thermal.py      # 热传导
│   ├── electromagnetic.py # 电磁
│   └── coupling.py     # 多物理场耦合
├── analysis/   # 分析与表征
│   ├── topology.py     # 拓扑分析
│   ├── morphology.py   # 形态学
│   └── properties.py   # 性能指标
├── viz/        # 可视化
│   ├── render3d.py     # 3D渲染
│   ├── animate.py      # 动画
│   └── plot2d.py       # 2D绘图
├── utils/      # 工具函数
│   ├── io.py           # 文件I/O
│   ├── geometry.py     # 几何工具
│   └── materials.py    # 材料数据库
└── examples/   # 示例脚本
```

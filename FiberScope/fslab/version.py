"""Single source of truth for app identity.

Used by the UI version chip, the About dialog, the Windows exe resources
(scripts/build_exe.py) and the docs, so the number can never drift apart.
"""
APP_NAME = "FiberScope"
APP_VERSION = "3.0.0"
VERSION_TUPLE = (3, 0, 0, 0)
VERSION_CHIP = "V" + ".".join(APP_VERSION.split(".")[:2])

AUTHOR = "杨云浩"
AUTHOR_SIGNATURE = "复旦大学高分子科学系 杨云浩"
ORG_ZH = "复旦大学高分子科学系 · 聚合物分子工程国家重点实验室"
ORG_EN = ("Department of Macromolecular Science, Fudan University · "
          "State Key Laboratory of Polymer Molecular Engineering")

CONTEST_ZH = "世界人工智能开源大赛（GOAI）"
CONTEST_EN = "Global Open-source AI Competition (GOAI)"
TRACK_ZH = "赛道三 · 前沿探索 AI for Research · 开放探索赛题"
TRACK_EN = "Track 3 · AI for Research · open exploration"

ACK_ZH = ("本作品为「世界人工智能开源大赛（GOAI）」赛道三 · 前沿探索 "
          "AI for Research 开放探索赛题参赛作品。结构生成复用了自有开源项目 "
          "fibernet（MIT）的 gen.pattern 模块，物理引擎、渗流、特征、机器学习、"
          "逆设计适配、探索管线与界面为本项目实现；监督模型使用 scikit-learn，强化学习使用 Stable-Baselines3。")
ACK_EN = ("Built for the Global Open-source AI Competition (GOAI), Track 3: "
          "AI for Research, open exploration. Structure generation reuses the "
          "authors' own MIT-licensed fibernet package (gen.pattern); the "
          "physics engine, percolation, features, ML surrogate, inverse "
          "design adapters, exploration pipeline and UI are project implementations. Supervised estimators use scikit-learn; reinforcement learning uses Stable-Baselines3.")
UPSTREAM = "https://github.com/GellmanSparrowS/fibernet"

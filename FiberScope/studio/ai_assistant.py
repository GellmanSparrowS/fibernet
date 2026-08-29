"""AI assistant dock: DeepSeek (OpenAI-compatible) function-calling loop that
can drive every internal interface of FiberScope (structure editing, sims,
inverse design, export, theme/lang/tab control).

The API key is entered once per user and persisted to
~/.fiberscope/config.json; shared copies of the app start without a key.
"""
import json
import os
import threading
import urllib.error
import urllib.request

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QDialog, QDockWidget, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QTextBrowser,
                               QVBoxLayout, QWidget)

from fslab import StructureFactory
from fslab.exporter import export_json, export_svg
from fslab.inverse import (run_inverse, metrics_of, PTS, TARGETS, SCALARS)
from fslab.structure import UNIT_PRESETS
from .i18n import tr
from .theme import colors

CFG_DIR = os.path.join(os.path.expanduser("~"), ".fiberscope")
CFG_PATH = os.path.join(CFG_DIR, "config.json")

DEFAULT_CFG = {"api_key": "", "base_url": "https://api.deepseek.com",
               "model": "deepseek-v4-flash"}


def load_cfg():
    cfg = dict(DEFAULT_CFG)
    try:
        if os.path.exists(CFG_PATH):
            with open(CFG_PATH, encoding="utf-8") as fh:
                cfg.update(json.load(fh))
    except Exception:
        pass
    return cfg


def save_cfg(cfg):
    os.makedirs(CFG_DIR, exist_ok=True)
    with open(CFG_PATH, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=1)


SYSTEM_PROMPT = """你是 FiberScope（纤维网络超材料探索台）的内置 AI 助手，一位兼具
计算力学与机器学习背景的科研搭档。用户通过你操纵本软件的全部接口；你应
主动、可靠、简洁地完成任务。

软件功能总览（用户不可见，供你决策）：
- 结构生成：单元类型（list_units 可查，如 square/hexagonal/honeycomb/
  reentrant/chiral/missing_rib 等）、网格 grid_x/grid_y、每边内点数 pts、
  随机种子 seed、节点扰动 perturbation(0..1)、单线位移谱 line_displacements
  （按各线方向周期复制到所有纤维线，从而整体改变基元形状）。
- 原位仿真：载荷路径渗流（渗流帧 perc_frame、序参量 P、骨架占比 backbone）
  与拉伸（峰值力/刚度/韧性 metrics、接触事件数 contacts_max）。
- AI 逆设计：曲线目标 J/C/linear/multi 或标量 max_peak/min_peak/
  max_stiffness/min_stiffness/max_toughness；优化单线中间点值；
  load_best 可把最优结构发回结构板块。
- 导出 JSON/SVG；切换 tab/主题/语言；探索回放摘要（新颖性搜索 agent vs
  R0 均匀随机参照系）。

工作原则：
1. 先调用工具获取真实数据再回答，绝不编造数字。
2. 每次任务结束必须给出一段简洁中文总结，包含关键数字（渗流帧、P、峰值力、
   刚度、韧性、目标距离等）；不要只停留在工具调用上。
3. 修改结构后说明改了什么、为什么；并给出下一步建议（逆设计/导出/换单元）。
4. 用户意图模糊时选择最合理默认并说明假设；一次最多做必要的事，不刷屏。"""


def build_tools():
    t = lambda name, desc, props, req=None: {
        "type": "function",
        "function": {"name": name, "description": desc,
                     "parameters": {"type": "object",
                                    "properties": props,
                                    "required": req or []}}}
    return [
        t("get_app_state", "当前 tab、结构规格、主题、语言等状态", {}),
        t("list_units", "列出全部可选单元类型", {}),
        t("set_structure", "设置结构并同步到所有仿真板块",
          {"unit": {"type": "string", "enum": list(UNIT_PRESETS)},
           "grid_x": {"type": "integer", "minimum": 1, "maximum": 8},
           "grid_y": {"type": "integer", "minimum": 1, "maximum": 8},
           "pts": {"type": "integer", "minimum": 0, "maximum": 6},
           "seed": {"type": "integer"},
           "perturbation": {"type": "number", "minimum": 0, "maximum": 1}},
          ["unit"]),
        t("set_line_displacements",
          "设置单线位移谱（列表，每项 [dx,dy] 为线长比例，长度=pts）；"
          "会周期复制到所有纤维线",
          {"displacements": {"type": "array", "items":
                             {"type": "array", "items": {"type": "number"}}}},
          ["displacements"]),
        t("set_perturbation", "设置节点扰动强度 0..1",
          {"value": {"type": "number", "minimum": 0, "maximum": 1}},
          ["value"]),
        t("randomize_seed", "随机一个新的种子", {}),
        t("run_percolation", "运行载荷路径渗流仿真并返回摘要",
          {"stretch": {"type": "number", "minimum": 1.1, "maximum": 3.0},
           "alpha": {"type": "number", "minimum": 0.02, "maximum": 0.15}}),
        t("run_stretch", "运行拉伸仿真并返回力/能量/接触摘要",
          {"stretch": {"type": "number", "minimum": 1.1, "maximum": 3.0},
           "use_bending": {"type": "boolean"},
           "use_contact": {"type": "boolean"}}),
        t("run_inverse_design",
          "运行逆设计（只优化当前单元基元的位移谱/扰动，不更换拓扑）；"
          "target 可为曲线 J/C/linear/multi 或标量 "
          "max_peak/min_peak/max_stiffness/min_stiffness/max_toughness；"
          "load_best=True 时把最优结构发送到结构板块",
          {"target": {"type": "string"},
           "budget": {"type": "integer", "minimum": 16, "maximum": 80},
           "load_best": {"type": "boolean"}}),
        t("export_structure", "导出当前结构为 json 或 svg",
          {"format": {"type": "string", "enum": ["json", "svg"]},
           "path": {"type": "string"}}, ["format"]),
        t("set_tab", "切换主界面 tab",
          {"tab": {"type": "string", "enum": ["structure", "percolation",
                                              "stretch", "design",
                                              "replay"]}}, ["tab"]),
        t("set_theme", "切换深/浅主题",
          {"mode": {"type": "string", "enum": ["dark", "light"]}}, ["mode"]),
        t("set_lang", "切换语言",
          {"lang": {"type": "string", "enum": ["zh", "en"]}}, ["lang"]),
        t("get_replay_summary", "探索回放日志摘要（agent vs R0 + 高新颖发现）", {}),
        t("load_replay_structure",
          "把探索回放第 index 条记录的结构载入结构板块",
          {"index": {"type": "integer", "minimum": 0}}, ["index"]),
    ]


def _j(**kw):
    return kw


class _ToolRequest:
    def __init__(self, name, args):
        self.name, self.args = name, args
        self.result = None
        self.event = threading.Event()


class AIWorker(QThread):
    note = Signal(str)
    aside = Signal(str)
    answer = Signal(str)
    failed = Signal(str)
    finished_ok = Signal()

    def __init__(self, cfg, messages, executor, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.messages = messages
        self.executor = executor

    def _post(self, msgs):
        body = json.dumps({"model": self.cfg["model"], "messages": msgs,
                           "tools": build_tools(), "temperature": 0.2},
                          ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.cfg["base_url"].rstrip("/") + "/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {self.cfg['api_key']}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.load(r)

    def run(self):
        import time
        t0 = time.time()
        self.tool_names = []
        self.last_result = ""
        self.answered = False
        self.elapsed = 0.0
        try:
            msgs = [dict(m) for m in self.messages]
            for _ in range(8):
                data = self._post(msgs)
                msg = data["choices"][0]["message"]
                entry = {"role": "assistant",
                         "content": msg.get("content") or ""}
                if msg.get("tool_calls"):
                    entry["tool_calls"] = msg["tool_calls"]
                    if entry["content"]:
                        self.aside.emit(entry["content"])
                msgs.append(entry)
                if not msg.get("tool_calls"):
                    if entry["content"]:
                        self.answered = True
                        self.answer.emit(entry["content"])
                    break
                for call in msg["tool_calls"]:
                    name = call["function"]["name"]
                    try:
                        args = json.loads(call["function"]["arguments"]
                                          or "{}")
                    except Exception:
                        args = {}
                    self.tool_names.append(name)
                    self.note.emit(name)
                    result = self.executor(name, args)
                    self.last_result = result
                    msgs.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": result})
            self.elapsed = time.time() - t0
            self._history = msgs
            self.finished_ok.emit()
        except urllib.error.HTTPError as e:
            self.failed.emit(f"HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}")
        except Exception as e:
            self.failed.emit(f"{e.__class__.__name__}: {e}")


class KeyDialog(QDialog):
    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("ai_btn"))
        self.setMinimumWidth(420)
        v = QVBoxLayout(self)
        hint = QLabel(tr("ai_key_hint"))
        hint.setWordWrap(True)
        v.addWidget(hint)
        form = QFormLayout()
        self.key = QLineEdit(cfg.get("api_key", ""))
        self.key.setEchoMode(QLineEdit.Password)
        self.url = QLineEdit(cfg.get("base_url", ""))
        self.model = QLineEdit(cfg.get("model", ""))
        form.addRow("API Key", self.key)
        form.addRow("Base URL", self.url)
        form.addRow("Model", self.model)
        v.addLayout(form)
        row = QHBoxLayout()
        ok = QPushButton("OK"); ok.setProperty("primary", True)
        cancel = QPushButton("Cancel")
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        row.addStretch(1); row.addWidget(ok); row.addWidget(cancel)
        v.addLayout(row)

    def result_cfg(self):
        return {"api_key": self.key.text().strip(),
                "base_url": self.url.text().strip(),
                "model": self.model.text().strip()}


class AIPanel(QWidget):
    _req = Signal(object)

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.cfg = load_cfg()
        self.worker = None
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "system", "content": ""}]
        self._req.connect(self._run_tool)
        self.setObjectName("aipanel")

        v = QVBoxLayout(self)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)
        top = QHBoxLayout()
        self.title_lbl = QLabel(tr("ai_title"))
        self.title_lbl.setObjectName("subtitle")
        self.cfg_btn = QPushButton("⚙")
        self.cfg_btn.setFixedWidth(34)
        self.cfg_btn.clicked.connect(self._open_cfg)
        self.hide_btn = QPushButton("×")
        self.hide_btn.setFixedWidth(34)
        self.hide_btn.clicked.connect(self.hide)
        top.addWidget(self.title_lbl, 1)
        top.addWidget(self.cfg_btn)
        top.addWidget(self.hide_btn)
        v.addLayout(top)
        self.status = QLabel()
        self.status.setObjectName("hint")
        v.addWidget(self.status)
        self._log = []
        self.view = QTextBrowser()
        self.view.setObjectName("aichat")
        self.view.setOpenExternalLinks(False)
        v.addWidget(self.view, 1)
        bottom = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText(tr("ai_placeholder"))
        self.input.returnPressed.connect(self.send)
        self.send_btn = QPushButton("➤")
        self.send_btn.setFixedWidth(40)
        self.send_btn.clicked.connect(self.send)
        bottom.addWidget(self.input, 1)
        bottom.addWidget(self.send_btn)
        v.addLayout(bottom)
        self.setMinimumWidth(320)
        self.status.setText(tr("ready"))
        self._build_registry()

    def retranslate(self):
        self.title_lbl.setText(tr("ai_title"))
        self.input.setPlaceholderText(tr("ai_placeholder"))

    def _dynamic_context(self):
        win = self.win
        ctx = {"tab": win.tabs.tabText(win.tabs.currentIndex()),
               "theme": win.mode}
        try:
            f = win.tab_struct.collect_factory()
            ctx["structure"] = dict(unit=f.unit, grid_x=f.grid_x,
                                    grid_y=f.grid_y,
                                    pts=f.n_pts_per_side, seed=f.seed,
                                    perturbation=round(f.perturbation, 3))
        except Exception:
            ctx["structure"] = {}
        t = win.tab_sim
        if getattr(t, "run", None) is not None:
            ctx["last_sim"] = dict(frames=t.run.n_frames,
                                   perc_frame=int(t.perc.perc_frame))
        return json.dumps(ctx, ensure_ascii=False)

    # ---------------- tools ----------------
    def _build_registry(self):
        win = self.win
        r = {}

        def j(**kw):
            return kw

        r["get_app_state"] = lambda: j(
            tab=win.tabs.tabText(win.tabs.currentIndex()),
            structure=win.tab_struct.collect_factory().__dict__,
            theme=win.mode, lang=__import__(
                "studio.i18n", fromlist=["get_lang"]).get_lang())
        r["list_units"] = lambda: j(units=list(UNIT_PRESETS))
        r["set_structure"] = lambda unit, grid_x=3, grid_y=3, pts=5, seed=7, \
            perturbation=0.0: (
                win.tab_struct.load_spec(StructureFactory(
                    unit=unit, grid_x=grid_x, grid_y=grid_y,
                    n_pts_per_side=pts, seed=seed,
                    perturbation=perturbation)),
                j(ok=True, unit=unit))[1]
        r["set_line_displacements"] = lambda displacements: (
            win.tab_struct.editor.set_displacements(displacements),
            win.tab_struct._sync_row(),
            win.tab_struct._update_preview(),
            win.tab_struct.push_spec(),
            j(ok=True, pts=len(displacements)))[4]
        r["set_perturbation"] = lambda value: (
            win.tab_struct.pert.setValue(int(round(value * 100))),
            win.tab_struct.push_spec(), j(ok=True))[2]
        r["randomize_seed"] = lambda: (
            win.tab_struct._random_seed(), j(ok=True,
                                             seed=win.tab_struct.seed.value()))[1]
        r["run_percolation"] = lambda stretch=2.0, alpha=0.05: self._run_perc(
            stretch, alpha)
        r["run_stretch"] = lambda stretch=2.2, use_bending=True, \
            use_contact=True: self._run_stretch(stretch, use_bending,
                                                use_contact)
        r["run_inverse_design"] = lambda target="J", budget=30, \
            load_best=False: self._run_inverse(target, budget, load_best)
        r["export_structure"] = lambda format, path=None: self._export(
            format, path)
        r["set_tab"] = lambda tab: (
            win.tabs.setCurrentIndex(
                {"structure": 0, "percolation": 1,
                 "stretch": 1, "design": 2,
                 "replay": 3}[tab]), j(ok=True))[1]
        r["set_theme"] = lambda mode: (
            win._toggle_theme() if win.mode != mode else None,
            j(ok=True, mode=win.mode))[1]
        r["set_lang"] = lambda lang: (
            win.lang_combo.setCurrentIndex(0 if lang == "zh" else 1),
            j(ok=True))[1]
        r["get_replay_summary"] = lambda: self._replay_summary()
        r["load_replay_structure"] = lambda index: self._load_replay(
            int(index))
        self.registry = r

    def _replay_summary(self):
        recs = self.win.tab_replay.recs
        top = sorted((r for r in recs if r.get("accepted")),
                     key=lambda r: -r["novelty"])[:5]
        return _j(recs=len(recs),
                  clusters_agent=self.win.tab_replay.cluster_curve["agent"][-1],
                  clusters_r0=self.win.tab_replay.cluster_curve["r0"][-1],
                  novelty_max=self.win.tab_replay.novelty_curve[-1],
                  top_discoveries=[
                      dict(id=int(r["id"]), unit=r["unit"],
                           seed=int(r["seed"]),
                           pert=round(float(r["pert"]), 3),
                           novelty=round(float(r["novelty"]), 2))
                      for r in top])

    def _load_replay(self, idx):
        recs = self.win.tab_replay.recs
        if not 0 <= idx < len(recs):
            return _j(error="index out of range", n=len(recs))
        r = recs[idx]
        self.win.tab_struct.load_spec(StructureFactory(
            unit=r["unit"], grid_x=3, grid_y=3, n_pts_per_side=PTS,
            seed=int(r["seed"]), perturbation=float(r["pert"])))
        return _j(ok=True, unit=r["unit"], seed=int(r["seed"]),
                  pert=float(r["pert"]), novelty=float(r["novelty"]))

    def _run_perc(self, stretch, alpha):
        t = self.win.tab_perc
        t.stretch.setValue(float(stretch))
        t.quant.setValue(int(round(alpha * 100)))
        t.run_sync()
        run, perc = t.run, t.perc
        return _j(frames=run.n_frames, N=int(run.frames_xy.shape[1]),
                 E=int(run.n_edges), perc_frame=int(perc.perc_frame),
                 P_end=float(perc.spanning_frac[-1]),
                 backbone_end=float(perc.backbone_frac[-1]))

    def _run_stretch(self, stretch, bend, contact):
        t = self.win.tab_stretch
        t.stretch.setValue(float(stretch))
        t.chk_bend.setChecked(bool(bend))
        t.chk_contact.setChecked(bool(contact))
        t.run_sync()
        run = t.run
        m = metrics_of(run)
        return _j(frames=run.n_frames, contacts_max=int(
            run.contact_counts.max()), metrics=m,
            final_stretch=float(run.strain_levels[-1]))

    def _run_inverse(self, target, budget, load_best):
        if target not in TARGETS and target not in SCALARS:
            return _j(error=f"unknown target {target}; choose from "
                           f"{list(TARGETS) + list(SCALARS)}")
        dt = self.win.tab_design

        def builder(unit, pert, ld):
            return StructureFactory(unit=unit, grid_x=3, grid_y=3,
                                    n_pts_per_side=PTS, seed=7,
                                    perturbation=pert,
                                    line_displacements=ld).build()
        res = run_inverse(builder, target, budget=int(budget), seed=7,
                          fixed_unit=self.win.tab_struct.collect_factory().unit)
        if load_best and res.get("best_spec"):
            sp = res["best_spec"]
            self.win.tab_struct.load_spec(StructureFactory(
                unit=sp["unit"], grid_x=3, grid_y=3, n_pts_per_side=PTS,
                seed=7, perturbation=sp["pert"],
                line_displacements=sp["line_displacements"]))
        try:
            self.win.tab_design.show_external(res, target)
        except Exception:
            pass
        m = metrics_of(res["best_run"]) if res.get("best_run") else None
        return _j(best_label=res["best_label"],
                 best_obj=round(res["best_dist"], 4),
                 best_spec=res["best_spec"], metrics=m,
                 loaded_into_structure=bool(load_best))

    def _export(self, fmt, path):
        f = self.win.tab_struct.collect_factory()
        path = os.path.abspath(path or os.path.join(
            os.path.expanduser("~"), "Documents",
            f"fiberscope_{f.unit}.{fmt}"))
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        if fmt == "json":
            export_json(f, path)
        else:
            export_svg(f, path, dark=self.win.mode == "dark")
        return _j(ok=True, path=path)

    def _run_tool(self, req):
        try:
            fn = self.registry[req.name]
            out = fn(**req.args)
            req.result = json.dumps(out, ensure_ascii=False, default=str)
        except Exception as e:
            req.result = json.dumps(
                {"error": f"{e.__class__.__name__}: {e}"})
        if len(req.result) > 4000:
            req.result = req.result[:4000] + "…(truncated)"
        req.event.set()

    def _executor(self, name, args):
        req = _ToolRequest(name, args)
        self._req.emit(req)
        req.event.wait(600)
        return req.result or json.dumps({"error": "tool timeout"})

    # ---------------- chat ----------------
    def _esc(self, s):
        return (s.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;"))

    def _append(self, who, text, role="ai"):
        c = colors(self.win.mode)
        col = {"ai": c["accent"], "user": c["text"], "note": c["violet"],
               "dim": c["sub"], "ok": c["ok"],
               "err": c["hot"]}.get(role, c["text"])
        self._log.append((who, text, role))
        self.view.append(
            f'<p style="color:{col};margin-top:6px"><b>{who}</b> '
            f'{self._esc(text)}</p>')

    def refresh_theme(self):
        """Re-render chat history with the current theme palette."""
        entries, self._log = self._log, []
        sb = self.view.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        self.view.clear()
        for who, text, role in entries:
            self._append(who, text, role)
        if at_bottom:
            self.view.verticalScrollBar().setValue(
                self.view.verticalScrollBar().maximum())

    def _open_cfg(self):
        d = KeyDialog(self.cfg, self)
        if d.exec():
            self.cfg = d.result_cfg()
            save_cfg(self.cfg)
            self._append("·", "config saved", "ok")

    def send(self):
        text = self.input.text().strip()
        if not text:
            return
        if not self.cfg.get("api_key"):
            self._open_cfg()
            if not self.cfg.get("api_key"):
                return
        self.input.clear()
        self._append("你", text, "user")
        self.messages[1] = {"role": "system",
                            "content": "当前状态（仅供你参考，勿逐字复述）: "
                                       + self._dynamic_context()}
        self.messages.append({"role": "user", "content": text})
        if self.worker is not None and self.worker.isRunning():
            return
        self.worker = AIWorker(self.cfg, self.messages, self._executor)
        self.worker.note.connect(
            lambda n: self._append("·", f"tool → {n}", "note"))
        self.worker.aside.connect(
            lambda t2: self._append("·", t2, "dim"))
        self.worker.answer.connect(self._on_answer)
        self.worker.failed.connect(self._on_fail)
        self.worker.finished_ok.connect(self._on_worker_done)
        self.send_btn.setEnabled(False)
        self.status.setText(tr("ai_thinking"))
        self.worker.start()

    def _on_answer(self, text):
        self.messages.append({"role": "assistant", "content": text})
        self._append("AI", text)

    def _on_fail(self, msg):
        self._append("!", msg, "err")
        self.send_btn.setEnabled(True)
        self.status.setText(tr("ready"))

    def _on_worker_done(self):
        self.send_btn.setEnabled(True)
        w = self.worker
        names = getattr(w, "tool_names", [])
        if not getattr(w, "answered", False):
            uniq = ", ".join(sorted(set(names)))
            extra = ""
            lr = getattr(w, "last_result", "")
            if lr:
                extra = " · " + lr[:160]
            txt = tr("ai_fallback") + uniq + extra
            self._append("AI", txt)
            self.messages.append({"role": "assistant", "content": txt})
        self._append("·", f"{tr('ai_done')} · tools {len(names)} · "
                          f"{getattr(w, 'elapsed', 0.0):.1f}s", "ok")
        self.status.setText(tr("ready"))
        # keep chat memory bounded
        head = [m for m in self.messages if m["role"] == "system"][:2]
        tail = [m for m in self.messages if m["role"] != "system"][-16:]
        self.messages = head + tail


AIDock = AIPanel  # legacy alias

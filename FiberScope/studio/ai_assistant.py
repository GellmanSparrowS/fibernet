'''AI assistant dock: DeepSeek (OpenAI-compatible) function-calling loop that
can drive every internal interface of FiberScope (structure editing, sims,
inverse design, features, 3D surface, ML surrogate, export, theme/lang/tab).

Chat UI is a professional LLM-console style: user/assistant bubbles,
tool-call cards with expandable arguments/result, thinking indicator and
explicit completion feedback.  The API key is entered once per user and
persisted to ~/.fiberscope/config.json; shared copies start without a key.
'''
import json
import os
import threading
import urllib.error
import urllib.request

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (QDialog, QDockWidget, QFormLayout,
                               QFrame, QHBoxLayout, QLabel, QLineEdit,
                               QPlainTextEdit, QPushButton, QScrollArea,
                               QVBoxLayout, QWidget)

from fslab import StructureFactory
from fslab.exporter import export_json, export_svg
from fslab.inverse import (run_inverse, metrics_of, PTS, TARGETS, SCALARS)
from fslab.structure import UNIT_PRESETS, SPECTRUM_PRESETS
from .i18n import tr
from .theme import colors

CFG_DIR = os.path.join(os.path.expanduser('~'), '.fiberscope')
CFG_PATH = os.path.join(CFG_DIR, 'config.json')

DEFAULT_CFG = {'api_key': '', 'base_url': 'https://api.deepseek.com',
               'model': 'deepseek-v4-flash'}


def load_cfg():
    cfg = dict(DEFAULT_CFG)
    try:
        if os.path.exists(CFG_PATH):
            with open(CFG_PATH, encoding='utf-8') as fh:
                cfg.update(json.load(fh))
    except Exception:
        pass
    return cfg


def save_cfg(cfg):
    os.makedirs(CFG_DIR, exist_ok=True)
    with open(CFG_PATH, 'w', encoding='utf-8') as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=1)


SKILL_DOC = '''# FiberScope 操作技能文档（内置）

## 快速定位
- 每次工具调用后界面会自动跳到对应 tab，无需手动切页。
- get_app_state 先看当前状态；list_units 查可用单元。

## 结构生成
- 基础单元（UI）：square / triangle / hexagon / voronoi /
  reentrant / chiral / star / cross / missing_rib / diamond。
- 方形谱预设：square / auxetic_bow(内凹弓) / rhombic_bow(外凸弓) /
  swirl(旋涡) / zigzag(锯齿) / pinwheel(风车)，本质都是方形基元 +
  单线位移谱。
- 编辑方式：set_line_displacements 给 [dx,dy]（线长比例，长度=pts，
  通常 5）；这是“调整中间点 -> 整体同构变形”的核心接口。
- set_structure(unit, grid_x, grid_y, pts, seed, perturbation)。

## 仿真
- run_percolation(stretch, alpha) 看载荷路径是否贯通（perc_frame>=0）。
- run_stretch(stretch, use_bending, use_contact) 看峰值力/刚度/韧性/
  接触事件。

## 逆设计
- 目标曲线：J（延迟硬化）/ C（早期硬化）/ linear。
- 标量：max_peak / min_peak / max_stiffness / min_stiffness / max_toughness。
- 只改当前基元，不换拓扑；budget 推荐 40-60。load_best=True 回填结构。
- 先用大预算粗搜，再小预算精修。

## 特征 / 曲面 / ML
- run_features(groups=all|structure|pore|contact)。
- set_surface(obj, unit, preset, amplitude) 铺到 OBJ 曲面。
- run_ml_training(n, epochs) 返回 R2。

## 导出
- export_structure(format=json|svg, path 可省，默认保存到文档目录)。
'''

SYSTEM_PROMPT = '''你是 FiberScope（纤维网络超材料探索台）的内置 AI 助手，一位兼具
计算力学与机器学习背景的科研搭档。用户通过你操纵本软件的全部接口；你应
主动、可靠、简洁地完成任务。遇到不熟悉的操作可调用 get_skill_doc 读取
内置技能文档，再执行。

软件功能总览（用户不可见，供你决策）：
- 结构生成：方形基元 + 单线位移谱（P1 生成模型，C4 旋转复制+透镜焊接）；
  谱预设 square/auxetic_bow/rhombic_bow/swirl/zigzag/pinwheel；经典基元
  triangle/hexagon/voronoi/reentrant/chiral/star/cross/
  missing_rib/diamond。核心：编辑一条参考线的中间点即整体同构变形。
- 原位仿真：载荷路径渗流（perc_frame、P、backbone）与拉伸（峰值力/刚度/
  韧性、接触事件数 contacts_max）。
- AI 逆设计：曲线目标 J/C/linear 或标量 max_peak/min_peak/max_stiffness/
  min_stiffness/max_toughness；只优化当前基元，不更换拓扑。
- 特征分析：run_features 返回结构/孔隙/接触三组特征（中文名）。
- 三维曲面：set_surface 把纤维网络铺到 OBJ 曲面并实时变形。
- 机器学习：run_ml_training 生成结构-性能数据集并训练代理模型（返回 R2）。
- 导出 JSON/SVG；切换 tab/主题/语言；探索回放摘要（agent vs R0 参照系）。

工作原则：
1. 先调用工具获取真实数据再回答，绝不编造数字。
2. 每次任务结束必须给出一段简洁中文总结，包含关键数字；不要只停留在工具调用。
3. 修改结构后说明改了什么、为什么；并给出下一步建议。
4. 用户意图模糊时选择最合理默认并说明假设；一次最多做必要的事，不刷屏。
5. 要操作某个界面时直接调用对应工具，系统会自动跳转到该界面。'''


def build_tools():
    t = lambda name, desc, props, req=None: {
        'type': 'function',
        'function': {'name': name, 'description': desc,
                     'parameters': {'type': 'object',
                                    'properties': props,
                                    'required': req or []}}}
    return [
        t('get_app_state', '当前 tab、结构规格、主题、语言等状态', {}),
        t('list_units', '列出全部可选单元类型', {}),
        t('set_structure', '设置结构并同步到所有仿真板块',
          {'unit': {'type': 'string', 'enum': list(UNIT_PRESETS)},
           'grid_x': {'type': 'integer', 'minimum': 1, 'maximum': 8},
           'grid_y': {'type': 'integer', 'minimum': 1, 'maximum': 8},
           'pts': {'type': 'integer', 'minimum': 0, 'maximum': 6},
           'seed': {'type': 'integer'},
           'perturbation': {'type': 'number', 'minimum': 0, 'maximum': 1}},
          ['unit']),
        t('set_line_displacements',
          '设置单线位移谱（列表，每项 [dx,dy] 为线长比例，长度=pts）；'
          '会周期复制到所有纤维线',
          {'displacements': {'type': 'array', 'items':
                             {'type': 'array', 'items': {'type': 'number'}}}},
          ['displacements']),
        t('set_perturbation', '设置节点扰动强度 0..1',
          {'value': {'type': 'number', 'minimum': 0, 'maximum': 1}},
          ['value']),
        t('randomize_seed', '随机一个新的种子', {}),
        t('run_percolation', '运行载荷路径渗流仿真并返回摘要',
          {'stretch': {'type': 'number', 'minimum': 1.1, 'maximum': 3.0},
           'alpha': {'type': 'number', 'minimum': 0.02, 'maximum': 0.15}}),
        t('run_stretch', '运行拉伸仿真并返回力/能量/接触摘要',
          {'stretch': {'type': 'number', 'minimum': 1.1, 'maximum': 3.0},
           'use_bending': {'type': 'boolean'},
           'use_contact': {'type': 'boolean'}}),
        t('run_inverse_design',
          '运行逆设计（只优化当前单元基元的位移谱/扰动，不更换拓扑）；'
          'target 可为曲线 J/C/linear/multi 或标量 '
          'max_peak/min_peak/max_stiffness/min_stiffness/max_toughness；'
          'load_best=True 时把最优结构发送到结构板块',
          {'target': {'type': 'string'},
           'budget': {'type': 'integer', 'minimum': 16, 'maximum': 80},
           'load_best': {'type': 'boolean'}}),
        t('run_features', '计算当前结构的结构/孔隙/接触特征（中文名摘要）',
          {'groups': {'type': 'string',
                      'enum': ['all', 'structure', 'pore', 'contact']}}),
        t('set_surface', '在三维曲面板块把纤维网络铺到 OBJ 曲面并变形；'
          'obj 可填模型名片段(如 lung/heart/vans)，'
          'preset 可选 square/auxetic_bow/rhombic_bow/swirl',
          {'obj': {'type': 'string'},
           'unit': {'type': 'string',
                    'enum': ['square', 'triangle', 'hexagon', 'reentrant']},
           'preset': {'type': 'string',
                      'enum': ['square', 'auxetic_bow', 'rhombic_bow',
                               'swirl']},
           'amplitude': {'type': 'number', 'minimum': 0.0,
                         'maximum': 1.5}}),
        t('run_ml_training', '生成结构-性能数据集并训练代理模型，返回 R2',
          {'n': {'type': 'integer', 'minimum': 16, 'maximum': 120},
           'epochs': {'type': 'integer', 'minimum': 10, 'maximum': 300}}),
        t('export_structure', '导出当前结构为 json 或 svg',
          {'format': {'type': 'string', 'enum': ['json', 'svg']},
           'path': {'type': 'string'}}, ['format']),
        t('get_skill_doc', '读取内置操作技能文档（结构/仿真/逆设计/特征/曲面/ML/导出）', {}),
        t('set_tab', '切换主界面 tab',
          {'tab': {'type': 'string',
                   'enum': ['structure', 'percolation', 'stretch', 'design',
                            'replay', 'features', 'surface', 'ml']}},
          ['tab']),
        t('set_theme', '切换深/浅主题',
          {'mode': {'type': 'string', 'enum': ['dark', 'light']}}, ['mode']),
        t('set_lang', '切换语言',
          {'lang': {'type': 'string', 'enum': ['zh', 'en']}}, ['lang']),
        t('get_replay_summary', '探索回放日志摘要（agent vs R0 + 高新颖发现）', {}),
        t('load_replay_structure',
          '把探索回放第 index 条记录的结构载入结构板块',
          {'index': {'type': 'integer', 'minimum': 0}}, ['index']),
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
        import ssl
        import time
        body = json.dumps({'model': self.cfg['model'], 'messages': msgs,
                           'tools': build_tools(), 'temperature': 0.2},
                          ensure_ascii=False).encode('utf-8')
        url = self.cfg['base_url'].rstrip('/') + '/chat/completions'
        last = None
        for attempt in range(3):
            # some corporate proxies break cert chains (ASN1 errors);
            # fall back to an unverified context only after a TLS failure
            for ctx in (None, ssl._create_unverified_context()):
                req = urllib.request.Request(
                    url, data=body,
                    headers={'Authorization': 'Bearer ' + self.cfg['api_key'],
                             'Content-Type': 'application/json'})
                try:
                    with urllib.request.urlopen(req, timeout=180,
                                                context=ctx) as r:
                        return json.load(r)
                except urllib.error.HTTPError as e:
                    last = e
                    if e.code not in (429, 500, 502, 503, 529):
                        raise
                    break
                except Exception as e:
                    last = e
            time.sleep(1.0 + attempt)
        raise last

    def run(self):
        import time
        t0 = time.time()
        self.tool_names = []
        self.last_result = ''
        self.answered = False
        self.elapsed = 0.0
        try:
            msgs = [dict(m) for m in self.messages]
            for _ in range(8):
                data = self._post(msgs)
                msg = data['choices'][0]['message']
                entry = {'role': 'assistant',
                         'content': msg.get('content') or ''}
                if msg.get('tool_calls'):
                    entry['tool_calls'] = msg['tool_calls']
                    if entry['content']:
                        self.aside.emit(entry['content'])
                msgs.append(entry)
                if not msg.get('tool_calls'):
                    if entry['content']:
                        self.answered = True
                        self.answer.emit(entry['content'])
                    break
                for call in msg['tool_calls']:
                    name = call['function']['name']
                    try:
                        args = json.loads(call['function']['arguments']
                                          or '{}')
                    except Exception:
                        args = {}
                    self.tool_names.append(name)
                    self.note.emit(name)
                    result = self.executor(name, args)
                    self.last_result = result
                    msgs.append({'role': 'tool', 'tool_call_id': call['id'],
                                 'content': result})
            self.elapsed = time.time() - t0
            self._history = msgs
            self.finished_ok.emit()
        except urllib.error.HTTPError as e:
            self.failed.emit('HTTP %s: %s' % (
                e.code, e.read()[:300].decode('utf-8', 'replace')))
        except Exception as e:
            self.failed.emit('%s: %s' % (e.__class__.__name__, e))


class KeyDialog(QDialog):
    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr('ai_btn'))
        self.setMinimumWidth(420)
        v = QVBoxLayout(self)
        hint = QLabel(tr('ai_key_hint'))
        hint.setWordWrap(True)
        v.addWidget(hint)
        form = QFormLayout()
        self.key = QLineEdit(cfg.get('api_key', ''))
        self.key.setEchoMode(QLineEdit.Password)
        self.url = QLineEdit(cfg.get('base_url', ''))
        self.model = QLineEdit(cfg.get('model', ''))
        form.addRow('API Key', self.key)
        form.addRow('Base URL', self.url)
        form.addRow('Model', self.model)
        v.addLayout(form)
        row = QHBoxLayout()
        ok = QPushButton('OK')
        ok.setProperty('primary', True)
        cancel = QPushButton('Cancel')
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        row.addStretch(1)
        row.addWidget(ok)
        row.addWidget(cancel)
        v.addLayout(row)

    def result_cfg(self):
        return {'api_key': self.key.text().strip(),
                'base_url': self.url.text().strip(),
                'model': self.model.text().strip()}


# ================= chat widgets =================
class Bubble(QFrame):
    '''One chat bubble: user / assistant / aside(reasoning) / meta / err.'''

    def __init__(self, text, kind, mode, parent=None):
        super().__init__(parent)
        self.kind = kind
        c = colors(mode)
        if kind == 'user':
            bg, fg = c['accent'], c['bg']
        elif kind == 'err':
            bg, fg = c['hot'], c['bg']
        elif kind in ('meta', 'aside'):
            bg, fg = 'transparent', c['sub']
        else:
            bg, fg = c['well'], c['text']
        self.setObjectName('bubble_' + kind)
        self.setStyleSheet(
            'Bubble#bubble_%s{background:%s;border-radius:10px;}'
            % (kind, bg))
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 8, 12, 8)
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        extra = 'font-size:11px;font-style:italic;' if kind == 'aside' \
            else 'font-size:12px;'
        if kind == 'meta':
            extra = 'font-size:10px;'
        lbl.setStyleSheet('color:%s;background:transparent;%s'
                          % (fg, extra))
        v.addWidget(lbl)
        self.label = lbl

    def set_text(self, text):
        self.label.setText(text)


class ToolCard(QFrame):
    '''Collapsible tool-call card: name + status + args/result body.'''

    def __init__(self, name, mode, parent=None):
        super().__init__(parent)
        self.name = name
        self.done = False
        c = colors(mode)
        self.mode = mode
        self.setObjectName('toolcard')
        self.setStyleSheet(
            'ToolCard#toolcard{background:%s;border:1px solid %s;'
            'border-radius:8px;}' % (c['bg'], c['faint']))
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 4, 8, 4)
        v.setSpacing(2)
        head = QHBoxLayout()
        self.toggle = QPushButton('▸ ' + name)
        self.toggle.setFlat(True)
        self.toggle.setCursor(Qt.PointingHandCursor)
        self.toggle.setStyleSheet('color:%s;font-weight:600;text-align:left;'
                                  'border:none;' % c['accent2'])
        self.toggle.clicked.connect(self._toggle)
        self.status = QLabel('…')
        self.status.setStyleSheet('color:%s;font-size:10px;' % c['sub'])
        head.addWidget(self.toggle, 1)
        head.addWidget(self.status)
        v.addLayout(head)
        self.body = QPlainTextEdit()
        self.body.setReadOnly(True)
        self.body.setMaximumHeight(140)
        self.body.setVisible(False)
        self.body.setStyleSheet(
            'color:%s;background:%s;font-family:Consolas,monospace;'
            'font-size:10px;border:none;' % (c['sub'], c['bg']))
        v.addWidget(self.body)

    def _toggle(self):
        self.body.setVisible(not self.body.isVisible())
        self.toggle.setText(('▾ ' if self.body.isVisible() else '▸ ')
                            + self.name)

    def set_args(self, args):
        self.body.appendPlainText('args: ' + args)

    def finish(self, result, ok):
        self.done = True
        c = colors(self.mode)
        self.status.setText('✓' if ok else '✗')
        self.status.setStyleSheet('color:%s;font-size:11px;font-weight:700;'
                                  % (c['ok'] if ok else c['hot']))
        self.body.appendPlainText('result: ' + result)


# ================= panel =================
class AIPanel(QWidget):
    _req = Signal(object)

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.cfg = load_cfg()
        self.worker = None
        self.messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
                         {'role': 'system', 'content': ''}]
        self._req.connect(self._run_tool)
        self.setObjectName('aipanel')
        self._log = []
        self._cards = []
        self._turn = None
        self._thinking = None
        self._think_timer = QTimer(self)
        self._think_timer.setInterval(400)
        self._think_timer.timeout.connect(self._tick_think)
        self._think_n = 0

        v = QVBoxLayout(self)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)
        top = QHBoxLayout()
        self.title_lbl = QLabel(tr('ai_title'))
        self.title_lbl.setObjectName('subtitle')
        self.cfg_btn = QPushButton('⚙')
        self.cfg_btn.setFixedWidth(34)
        self.cfg_btn.clicked.connect(self._open_cfg)
        self.hide_btn = QPushButton('×')
        self.hide_btn.setFixedWidth(34)
        self.hide_btn.clicked.connect(self.hide)
        top.addWidget(self.title_lbl, 1)
        top.addWidget(self.cfg_btn)
        top.addWidget(self.hide_btn)
        v.addLayout(top)
        self.status = QLabel()
        self.status.setObjectName('hint')
        v.addWidget(self.status)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.flow_w = QWidget()
        self.flow = QVBoxLayout(self.flow_w)
        self.flow.setContentsMargins(2, 2, 2, 2)
        self.flow.setSpacing(8)
        self.flow.addStretch(1)
        self.scroll.setWidget(self.flow_w)
        v.addWidget(self.scroll, 1)

        bottom = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText(tr('ai_placeholder'))
        self.input.returnPressed.connect(self.send)
        self.send_btn = QPushButton('➤')
        self.send_btn.setProperty('primary', True)
        self.send_btn.setFixedWidth(44)
        self.send_btn.clicked.connect(self.send)
        bottom.addWidget(self.input, 1)
        bottom.addWidget(self.send_btn)
        v.addLayout(bottom)
        self.setMinimumWidth(360)
        self.status.setText(tr('ready'))
        self._build_registry()

    def retranslate(self):
        self.title_lbl.setText(tr('ai_title'))
        self.input.setPlaceholderText(tr('ai_placeholder'))

    # ---------------- flow helpers ----------------
    def _scroll_bottom(self):
        sb = self.scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _add_row(self, w, align):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        if align == 'right':
            row.addStretch(1)
            row.addWidget(w)
        elif align == 'center':
            row.addStretch(1)
            row.addWidget(w)
            row.addStretch(1)
        else:
            row.addWidget(w)
            row.addStretch(1)
        self.flow.insertLayout(self.flow.count() - 1, row)
        w.setMaximumWidth(int(self.scroll.width() * 0.92))
        self._scroll_bottom()

    def _new_turn(self):
        self._turn = QWidget()
        self._turn_layout = QVBoxLayout(self._turn)
        self._turn_layout.setContentsMargins(0, 0, 0, 0)
        self._turn_layout.setSpacing(6)
        self._add_row(self._turn, 'left')
        self._cards = []
        return self._turn_layout

    def _turn_layout_or_new(self):
        if self._turn is None:
            return self._new_turn()
        return self._turn_layout

    def _start_thinking(self):
        self._stop_thinking()
        lay = self._turn_layout_or_new()
        self._thinking = Bubble('思考中…', 'aside', self.win.mode)
        lay.addWidget(self._thinking)
        self._think_n = 0
        self._think_timer.start()
        self._scroll_bottom()

    def _tick_think(self):
        self._think_n += 1
        if self._thinking is not None:
            dots = '.' * (self._think_n % 3 + 1)
            self._thinking.set_text(tr('ai_thinking') + dots)

    def _stop_thinking(self):
        self._think_timer.stop()
        if self._thinking is not None:
            self._thinking.deleteLater()
            self._thinking = None

    # ---------------- registry ----------------
    def _dynamic_context(self):
        win = self.win
        ctx = {'tab': win.tabs.tabText(win.tabs.currentIndex()),
               'theme': win.mode}
        try:
            f = win.tab_struct.collect_factory()
            ctx['structure'] = dict(unit=f.unit, grid_x=f.grid_x,
                                    grid_y=f.grid_y,
                                    pts=f.n_pts_per_side, seed=f.seed,
                                    perturbation=round(f.perturbation, 3))
        except Exception:
            ctx['structure'] = {}
        t = win.tab_sim
        if getattr(t, 'run', None) is not None:
            ctx['last_sim'] = dict(frames=t.run.n_frames,
                                   perc_frame=int(t.perc.perc_frame))
        return json.dumps(ctx, ensure_ascii=False)

    def _build_registry(self):
        win = self.win
        r = {}

        def j(**kw):
            return kw

        r['get_app_state'] = lambda: j(
            tab=win.tabs.tabText(win.tabs.currentIndex()),
            structure=win.tab_struct.collect_factory().__dict__,
            theme=win.mode, lang=__import__(
                'studio.i18n', fromlist=['get_lang']).get_lang())
        r['get_skill_doc'] = lambda: j(skill_doc=SKILL_DOC)
        r['list_units'] = lambda: j(units=list(UNIT_PRESETS))
        r['set_structure'] = lambda unit, grid_x=3, grid_y=3, pts=5, seed=7, \
            perturbation=0.0: (
                win.tab_struct.load_spec(StructureFactory(
                    unit=unit, grid_x=grid_x, grid_y=grid_y,
                    n_pts_per_side=pts, seed=seed,
                    perturbation=perturbation)),
                j(ok=True, unit=unit))[1]
        r['set_line_displacements'] = lambda displacements: (
            win.tab_struct.editor.set_displacements(displacements),
            win.tab_struct._sync_row(),
            win.tab_struct._update_preview(),
            win.tab_struct.push_spec(),
            j(ok=True, pts=len(displacements)))[4]
        r['set_perturbation'] = lambda value: (
            win.tab_struct.pert.setValue(int(round(value * 100))),
            win.tab_struct.push_spec(), j(ok=True))[2]
        r['randomize_seed'] = lambda: (
            win.tab_struct._random_seed(), j(ok=True,
                                             seed=win.tab_struct.seed.value()))[1]
        r['run_percolation'] = lambda stretch=2.0, alpha=0.05: self._run_perc(
            stretch, alpha)
        r['run_stretch'] = lambda stretch=2.2, use_bending=True, \
            use_contact=True: self._run_stretch(stretch, use_bending,
                                                contact=use_contact)
        r['run_inverse_design'] = lambda target='J', budget=30, \
            load_best=False: self._run_inverse(target, budget, load_best)
        r['run_features'] = lambda groups='all': self._tool_features(groups)
        r['set_surface'] = lambda obj='', unit='square', \
            preset='auxetic_bow', amplitude=1.0: self._tool_surface(
                obj, unit, preset, amplitude)
        r['run_ml_training'] = lambda n=40, epochs=80: self._tool_ml(n, epochs)
        r['export_structure'] = lambda format, path=None: self._export(
            format, path)
        r['set_tab'] = lambda tab: (
            win.tabs.setCurrentIndex(
                {'structure': 0, 'percolation': 1,
                 'stretch': 1, 'features': 2,
                 'ml': 3, 'design': 4, 'replay': 4,
                 'surface': 5}[tab]), j(ok=True))[1]
        r['set_theme'] = lambda mode: (
            win._toggle_theme() if win.mode != mode else None,
            j(ok=True, mode=win.mode))[1]
        r['set_lang'] = lambda lang: (
            win.lang_combo.setCurrentIndex(0 if lang == 'zh' else 1),
            j(ok=True))[1]
        r['get_replay_summary'] = lambda: self._replay_summary()
        r['load_replay_structure'] = lambda index: self._load_replay(
            int(index))
        self.registry = r

    # ---------------- new-interface tools ----------------
    def _tool_features(self, groups):
        try:
            from fslab.features import (compute_features, FEATURE_ZH,
                                         FEATURE_GROUPS)
        except Exception as e:
            return _j(error='features module not ready: %s' % e)
        f = self.win.tab_struct.collect_factory()
        g = f.build()
        import numpy as np
        pos = np.asarray(g.node_positions(), float)[:, :2]
        edges = np.asarray(g.edge_array(), int)[:, :2]
        d = compute_features(pos, edges)
        want = None if groups in ('all', None) else groups
        key2grp = {}
        for grp, keys in FEATURE_GROUPS.items():
            for k in keys:
                key2grp[k] = grp
        out = {}
        for k, v in d.items():
            if k in ('edge_lengths', 'orientations', 'pore_areas'):
                continue
            if want and key2grp.get(k, 'structure') != want:
                continue
            if isinstance(v, (int, float)):
                out[FEATURE_ZH.get(k, k)] = round(float(v), 4)
        return _j(n_features=len(out), features=out)

    def _tool_surface(self, obj, unit, preset, amplitude):
        try:
            from fslab import surface3d
            from .surface_tab import OBJ_FILES, UNIT_ITEMS
        except Exception as e:
            return _j(error='surface module not ready: %s' % e)
        pick_idx, pick = 0, OBJ_FILES[0]
        if obj:
            for i, pth in enumerate(OBJ_FILES):
                if obj.lower() in os.path.basename(pth).lower():
                    pick_idx, pick = i, pth
                    break
        u = unit if unit in UNIT_ITEMS else 'square'
        pk = preset if preset in SPECTRUM_PRESETS else 'auxetic_bow'
        tab = getattr(self.win, 'tab_surface', None)
        if tab is not None:
            tab.obj_combo.setCurrentIndex(pick_idx)
            tab.unit_combo.setCurrentIndex(UNIT_ITEMS.index(u))
            for i in range(tab.preset_combo.count()):
                if tab.preset_combo.itemData(i) == pk:
                    tab.preset_combo.setCurrentIndex(i)
                    break
            tab.amp_slider.setValue(
                max(0, min(150, int(float(amplitude) * 100))))
            tab._recompute()
        import numpy as np
        V, F = surface3d.parse_obj(pick)
        spec = np.asarray(SPECTRUM_PRESETS[pk], float) * float(amplitude)
        P, segs = surface3d.deform_surface(V, F, spec, unit=u)
        return _j(ok=True, obj=os.path.basename(pick), unit=u,
                  verts=int(len(P)), segments=int(len(segs)))

    def _tool_ml(self, n, epochs):
        try:
            from fslab.mlmodel import gen_dataset, MLP, r2_score, \
                TARGET_NAMES
        except Exception as e:
            return _j(error='ml module not ready: %s' % e)
        import numpy as np
        X, Y = gen_dataset('square', int(n), 0.2, 0.1, 1)
        if len(X) < 8:
            return _j(error='dataset too small (%d samples)' % len(X))
        m = MLP(hidden=32, seed=0)
        hist = m.train(X, Y, lr=3e-3, epochs=int(epochs), batch=16,
                       val_frac=0.2, patience=10, min_delta=1e-5)
        pred = m.predict(X)
        r2 = {TARGET_NAMES[i]: round(
            float(r2_score(Y[:, i], pred[:, i])), 3)
            for i in range(Y.shape[1])}
        return _j(samples=int(len(X)), epochs_run=hist['epochs_run'],
                  best_epoch=hist['best_epoch'],
                  early_stopped=bool(hist['early_stopped']), r2=r2)

    # ---------------- legacy sim tools ----------------
    def _replay_summary(self):
        recs = self.win.tab_replay.recs
        top = sorted((r for r in recs if r.get('accepted')),
                     key=lambda r: -r['novelty'])[:5]
        return _j(recs=len(recs),
                  clusters_agent=self.win.tab_replay.cluster_curve['agent'][-1],
                  clusters_r0=self.win.tab_replay.cluster_curve['r0'][-1],
                  novelty_max=self.win.tab_replay.novelty_curve[-1],
                  top_discoveries=[
                      dict(id=int(r['id']), unit=r['unit'],
                           seed=int(r['seed']),
                           pert=round(float(r['pert']), 3),
                           novelty=round(float(r['novelty']), 2))
                      for r in top])

    def _load_replay(self, idx):
        recs = self.win.tab_replay.recs
        if not 0 <= idx < len(recs):
            return _j(error='index out of range', n=len(recs))
        r = recs[idx]
        self.win.tab_struct.load_spec(StructureFactory(
            unit=r['unit'], grid_x=3, grid_y=3, n_pts_per_side=PTS,
            seed=int(r['seed']), perturbation=float(r['pert'])))
        return _j(ok=True, unit=r['unit'], seed=int(r['seed']),
                  pert=float(r['pert']), novelty=float(r['novelty']))

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

    def _run_stretch(self, stretch, bend, contact=True):
        t = self.win.tab_stretch
        t.stretch.setValue(float(stretch))
        t.chk_bend.setChecked(bool(bend))
        t.chk_contact.setChecked(bool(contact))
        t.run_sync()
        run = t.run
        m = metrics_of(run)
        return _j(frames=run.n_frames,
                  contacts_max=int(run.contact_counts.max()), metrics=m,
                  final_stretch=float(run.strain_levels[-1]))

    def _run_inverse(self, target, budget, load_best):
        if target not in TARGETS and target not in SCALARS:
            return _j(error='unknown target %s; choose from %s' % (
                target, list(TARGETS) + list(SCALARS)))
        dt = self.win.tab_design

        cur = self.win.tab_struct.collect_factory()
        pts = max(1, min(6, int(cur.n_pts_per_side)))

        def builder(unit, pert, ld):
            return StructureFactory(unit=unit, grid_x=3, grid_y=3,
                                    n_pts_per_side=pts, seed=7,
                                    perturbation=pert,
                                    line_displacements=ld).build()
        res = run_inverse(builder, target, budget=int(budget), seed=7,
                          fixed_unit=cur.unit, pts=pts)
        if load_best and res.get('best_spec'):
            sp = res['best_spec']
            self.win.tab_struct.load_spec(StructureFactory(
                unit=sp['unit'], grid_x=3, grid_y=3, n_pts_per_side=pts,
                seed=7, perturbation=sp['pert'],
                line_displacements=sp['line_displacements']))
        try:
            self.win.tab_design.show_external(res, target)
        except Exception:
            pass
        m = metrics_of(res['best_run']) if res.get('best_run') else None
        return _j(best_label=res['best_label'],
                  best_obj=round(res['best_dist'], 4),
                  best_spec=res['best_spec'], metrics=m,
                  loaded_into_structure=bool(load_best))

    def _export(self, fmt, path):
        f = self.win.tab_struct.collect_factory()
        path = os.path.abspath(path or os.path.join(
            os.path.expanduser('~'), 'Documents',
            'fiberscope_%s.%s' % (f.unit, fmt)))
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        if fmt == 'json':
            export_json(f, path)
        else:
            export_svg(f, path, dark=self.win.mode == 'dark')
        return _j(ok=True, path=path)

    _TAB_BY_TOOL = {
        'set_structure': 0, 'set_line_displacements': 0,
        'set_perturbation': 0, 'randomize_seed': 0, 'export_structure': 0,
        'run_percolation': 1, 'run_stretch': 1,
        'run_features': 2, 'run_ml_training': 3,
        'run_inverse_design': 4, 'load_replay_structure': 4,
        'get_replay_summary': 4, 'set_surface': 5,
    }

    def _run_tool(self, req):
        try:
            tab_idx = self._TAB_BY_TOOL.get(req.name)
            if tab_idx is not None:
                self.win.tabs.setCurrentIndex(tab_idx)
            fn = self.registry[req.name]
            out = fn(**req.args)
            req.result = json.dumps(out, ensure_ascii=False, default=str)
            ok = '"error"' not in req.result[:200]
        except Exception as e:
            req.result = json.dumps(
                {'error': '%s: %s' % (e.__class__.__name__, e)})
            ok = False
        if len(req.result) > 4000:
            req.result = req.result[:4000] + '…(truncated)'
        for card in reversed(self._cards):
            if card.name == req.name and not card.done:
                card.finish(req.result, ok)
                break
        self._log.append(('tool', req.name, req.args, req.result, ok))
        req.event.set()

    def _executor(self, name, args):
        req = _ToolRequest(name, args)
        self._req.emit(req)
        req.event.wait(600)
        return req.result or json.dumps({'error': 'tool timeout'})

    # ---------------- chat flow ----------------
    def _open_cfg(self):
        d = KeyDialog(self.cfg, self)
        if d.exec():
            self.cfg = d.result_cfg()
            save_cfg(self.cfg)
            self._log.append(('meta', 'config saved'))
            self._add_row(Bubble('config saved', 'meta', self.win.mode),
                          'center')

    def send(self):
        text = self.input.text().strip()
        if not text:
            return
        if not self.cfg.get('api_key'):
            self._open_cfg()
            if not self.cfg.get('api_key'):
                return
        self.input.clear()
        self._log.append(('user', text))
        self._add_row(Bubble(text, 'user', self.win.mode), 'right')
        self._turn = None
        self.messages[1] = {'role': 'system',
                            'content': '当前状态（仅供你参考，勿逐字复述）: '
                                       + self._dynamic_context()}
        self.messages.append({'role': 'user', 'content': text})
        if self.worker is not None and self.worker.isRunning():
            return
        self.worker = AIWorker(self.cfg, self.messages, self._executor)
        self.worker.note.connect(self._on_note)
        self.worker.aside.connect(self._on_aside)
        self.worker.answer.connect(self._on_answer)
        self.worker.failed.connect(self._on_fail)
        self.worker.finished_ok.connect(self._on_worker_done)
        self.send_btn.setEnabled(False)
        self.status.setText(tr('ai_thinking'))
        self._start_thinking()
        self.worker.start()

    def _on_note(self, name):
        lay = self._turn_layout_or_new()
        card = ToolCard(name, self.win.mode)
        self._cards.append(card)
        lay.addWidget(card)
        self._scroll_bottom()

    def _on_aside(self, text):
        lay = self._turn_layout_or_new()
        lay.addWidget(Bubble(text, 'aside', self.win.mode))
        self._log.append(('aside', text))
        self._scroll_bottom()

    def _on_answer(self, text):
        self._stop_thinking()
        lay = self._turn_layout_or_new()
        lay.addWidget(Bubble(text, 'ai', self.win.mode))
        self._log.append(('answer', text))
        self.messages.append({'role': 'assistant', 'content': text})
        self._scroll_bottom()

    def _on_fail(self, msg):
        self._stop_thinking()
        lay = self._turn_layout_or_new()
        lay.addWidget(Bubble(msg, 'err', self.win.mode))
        self._log.append(('err', msg))
        self.send_btn.setEnabled(True)
        self.status.setText(tr('ready'))
        self._scroll_bottom()

    def _on_worker_done(self):
        self._stop_thinking()
        lay = self._turn_layout_or_new()
        w = self.worker
        names = getattr(w, 'tool_names', [])
        if not getattr(w, 'answered', False):
            uniq = ', '.join(sorted(set(names)))
            extra = ''
            lr = getattr(w, 'last_result', '')
            if lr:
                extra = ' · ' + lr[:160]
            txt = tr('ai_fallback') + uniq + extra
            lay.addWidget(Bubble(txt, 'ai', self.win.mode))
            self._log.append(('answer', txt))
            self.messages.append({'role': 'assistant', 'content': txt})
        meta = '%s · tools %d · %.1fs' % (tr('ai_done'), len(names),
                                          getattr(w, 'elapsed', 0.0))
        lay.addWidget(Bubble(meta, 'meta', self.win.mode))
        self._log.append(('meta', meta))
        self.send_btn.setEnabled(True)
        self.status.setText(tr('ready'))
        self._scroll_bottom()
        head = [m for m in self.messages if m['role'] == 'system'][:2]
        tail = [m for m in self.messages if m['role'] != 'system'][-16:]
        self.messages = head + tail

    # ---------------- theme ----------------
    def refresh_theme(self):
        '''Rebuild the whole chat with the current palette.'''
        while self.flow.count():
            item = self.flow.takeAt(0)
            w = item.widget() or item.layout()
            if item.widget():
                item.widget().deleteLater()
        self.flow.addStretch(1)
        self._cards = []
        self._turn = None
        for entry in self._log:
            kind = entry[0]
            if kind == 'user':
                self._add_row(Bubble(entry[1], 'user', self.win.mode),
                              'right')
                self._turn = None
            elif kind == 'answer':
                lay = self._turn_layout_or_new()
                lay.addWidget(Bubble(entry[1], 'ai', self.win.mode))
            elif kind == 'aside':
                lay = self._turn_layout_or_new()
                lay.addWidget(Bubble(entry[1], 'aside', self.win.mode))
            elif kind == 'err':
                lay = self._turn_layout_or_new()
                lay.addWidget(Bubble(entry[1], 'err', self.win.mode))
            elif kind == 'meta':
                lay = self._turn_layout_or_new()
                lay.addWidget(Bubble(entry[1], 'meta', self.win.mode))
            elif kind == 'tool':
                _, name, args, result, ok = entry
                lay = self._turn_layout_or_new()
                card = ToolCard(name, self.win.mode)
                card.set_args(json.dumps(args, ensure_ascii=False)[:800])
                card.finish(result, ok)
                self._cards.append(card)
                lay.addWidget(card)
        self._scroll_bottom()


AIDock = AIPanel  # legacy alias

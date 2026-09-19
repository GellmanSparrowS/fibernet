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
                               QVBoxLayout, QWidget, QComboBox)

from fslab import StructureFactory
from fslab.exporter import export_json, export_svg
from fslab.inverse import (run_inverse, metrics_of, PTS, TARGETS, SCALARS)
from fslab.structure import (UNIT_PRESETS, SPECTRUM_PRESETS,
                             resample_spectrum, all_unit_keys, unit_display)
from .i18n import tr, get_lang
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
  reentrant / chiral / star / cross / diamond。
- 方形谱预设：square / auxetic_bow(内凹弓) / rhombic_bow(外凸弓) /
  swirl(旋涡) / zigzag(锯齿) / pinwheel(风车)，本质都是方形基元 +
  单线位移谱。
- 编辑方式：set_line_displacements 给 [dx,dy]（线长比例，长度=pts，
  通常 5）；这是“调整中间点 -> 整体同构变形”的核心接口。
- 精确编辑（相对指令首选）：
  * edit_line_point(index, dx, dy, mode)：只动第 index 个中间点
    （0..pts-1，沿线从起点到终点）；mode='add' 为在现值上叠加（“再往下
    一点”），mode='set' 为绝对赋值。dx 沿线方向、dy 垂直于线（对参考线
    而言 +y 向上），单位都是该线长度的比例。返回修改后的整条谱便于核对。
  * scale_spectrum(factor)：整条谱等比放大/缩小（“弓幅加大一倍”）。
  * set_spectrum_preset(name)：直接跳到某个命名预设。
  * 相对请求（一点/稍微/再/加大/减小）一律先 get_app_state 读当前谱，再
    用上面三个工具，不要凭记忆重写整条 set_line_displacements。
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
- run_features(groups=all|structure|pore)。
- set_surface(obj, unit, preset, amplitude) 铺到 OBJ 曲面。
- run_ml_training(n, epochs) 返回 R2。

## 一条龙准备打印
- start_print_workflow 依次执行结构、仿真、特征、物理数据、训练、C型逆设计、复核、三维映射、实体与导出、Bambu交接。
- 这是异步任务，使用 get_print_workflow_status 获取真实完成阶段；stop_print_workflow 可停止。
- complete仅指准备流程完成；printer_state说明文件就绪或已打开Bambu，不表示切片或实物打印完成。

## 导出
- export_structure(format=json|svg, path 可省，默认保存到文档目录)。
- export_result(kind, path 可省)：把当前结果写成 CSV/PNG（PNG 带标题带）。
  kind = curve_csv | force_png | percolation_png | energy_png | frame_png |
  features_csv | histogram_csv | fingerprint_png | inverse_csv | best_png |
  convergence_png | best_structure_png | canvas_png。
'''

SYSTEM_PROMPT = '''你是 FiberScope（纤维网络超材料探索台）的内置 AI 助手，一位兼具
计算力学与机器学习背景的科研搭档。用户通过你操纵本软件的全部接口；你应
主动、可靠、简洁地完成任务。遇到不熟悉的操作可调用 get_skill_doc 读取
内置技能文档，再执行。

软件功能总览（用户不可见，供你决策）：
- 结构生成：方形基元 + 单线位移谱（P1 生成模型，C4 旋转复制，共享边界保留重叠双纤维）；
  谱预设 square/auxetic_bow/rhombic_bow/swirl/zigzag/pinwheel；经典基元
  triangle/hexagon/voronoi/reentrant/chiral/star/cross/
  diamond。核心：编辑一条参考线的中间点即整体同构变形。
- 原位仿真：载荷路径渗流（perc_frame、P、backbone）与拉伸（峰值力/刚度/
  韧性、接触事件数 contacts_max）。
- AI 逆设计：曲线目标 J/C/linear 或标量 max_peak/min_peak/max_stiffness/
  min_stiffness/max_toughness；只优化当前基元，不更换拓扑。
- 特征分析：run_features 返回结构/孔隙两组特征（中文名）。
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
        t('start_print_workflow', '执行所有板块直到生成打印文件并交给Bambu Studio；异步执行，查状态，不代表已物理打印',
          {'samples':{'type':'integer'},'iterations':{'type':'integer'},'curved':{'type':'boolean'},'open_slicer':{'type':'boolean'},
           'target':{'type':'string'},'stretch':{'type':'number'},'surface_model':{'type':'string'},'create_demo':{'type':'boolean'},'demo_unit':{'type':'string'},
           'diameter':{'type':'number'},'print_size':{'type':'number'}}),
        t('get_print_workflow_status','读取一条龙每阶段状态和结果文件夹',{}),
        t('stop_print_workflow','停止一条龙流程，保留已完成结果',{}),
        t('get_app_state', '当前 tab、结构规格、主题、语言等状态', {}),
        t('list_units', '列出全部可选单元类型', {}),
        t('configure_learning', '配置学习模型、主动选样及模型专属参数',
          {'model': {'type': 'string', 'enum': ['mlp','deep_mlp','ridge','knn','random_forest','extra_trees']},
           'acquisition': {'type':'string','enum':['random','diversity','committee','hybrid']},
           'parameters': {'type': 'object'}}),
        t('generate_learning_data', '生成结构并逐个计算物理标签，完成后可用于训练',
          {'n': {'type': 'integer', 'minimum': 10, 'maximum': 2000},
           'mode': {'type':'string','enum':['physics']},
           'unit': {'type': 'string'}}),
        t('get_learning_status', '读取学习任务状态和已经完成的验证评分', {}),
        t('configure_search', '配置逆设计搜索或强化学习方法及参数',
          {'algorithm': {'type':'string','enum':['cem','PPO','A2C','DQN','SAC','TD3','DDPG']},
           'parameters': {'type':'object'}, 'unit': {'type':'string'},
           'amplitude': {'type':'number', 'minimum':.05, 'maximum':1.},
           'follow_type': {'type':'boolean'}}),
        t('get_design_status', '读取逆设计任务状态和已完成结果', {}),
        t('set_structure', '设置结构并同步到所有仿真板块',
          {'unit': {'type': 'string', 'description': 'Use list_units; includes saved custom cells'},
           'grid_x': {'type': 'integer', 'minimum': 1, 'maximum': 8},
           'grid_y': {'type': 'integer', 'minimum': 1, 'maximum': 8},
           'pts': {'type': 'integer', 'minimum': 0, 'maximum': 6},
           'seed': {'type': 'integer'},
           'perturbation': {'type': 'number', 'minimum': 0, 'maximum': 1}},
          ['unit']),
        t('set_expansion', '设置当前单元扩展规律，保留谱与其他参数',
          {'rule': {'type': 'string', 'enum': list(__import__('fslab.cell_rules', fromlist=['RULES']).RULES)},
           'grid_x': {'type': 'integer', 'minimum': 1, 'maximum': 8},
           'grid_y': {'type': 'integer', 'minimum': 1, 'maximum': 8}}, ['rule']),
        t('set_line_displacements',
          '设置单线位移谱（列表，每项 [dx,dy] 为线长比例，长度=pts）；'
          '会周期复制到所有纤维线',
          {'displacements': {'type': 'array', 'items':
                             {'type': 'array', 'items': {'type': 'number'}}}},
          ['displacements']),
        t('edit_line_point',
          '只修改位移谱的第 index 个中间点（0..pts-1）；mode=add 在现值上'
          '叠加（相对微调首选），mode=set 绝对赋值；dx 沿线、dy 垂直，'
          '单位为线长比例；返回修改后的整条谱',
          {'index': {'type': 'integer', 'minimum': 0, 'maximum': 5},
           'dx': {'type': 'number'}, 'dy': {'type': 'number'},
           'mode': {'type': 'string', 'enum': ['add', 'set']}},
          ['index', 'dx', 'dy']),
        t('scale_spectrum', '整条位移谱等比缩放（factor>1 放大变形幅度）',
          {'factor': {'type': 'number', 'minimum': 0.0, 'maximum': 4.0}},
          ['factor']),
        t('set_spectrum_preset', '把位移谱设为某个命名预设',
          {'name': {'type': 'string',
                    'enum': list(SPECTRUM_PRESETS)}}, ['name']),
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
           'budget': {'type': 'integer', 'minimum': 1},
           'load_best': {'type': 'boolean'}}),
        t('run_features', '计算当前结构的结构/孔隙特征（中文名摘要）',
          {'groups': {'type': 'string',
                      'enum': ['all', 'structure', 'pore']}}),
        t('set_surface', '在三维曲面板块把纤维网络铺到 OBJ 曲面并变形；'
          'obj 可填模型名片段(如 lung/heart/vans)，'
          'preset 可选 square/auxetic_bow/rhombic_bow/swirl',
          {'obj': {'type': 'string'},
           'unit': {'type': 'string',
                    'enum': all_unit_keys()},
           'preset': {'type': 'string',
                      'enum': ['square', 'auxetic_bow', 'rhombic_bow',
                               'swirl']},
           'amplitude': {'type': 'number', 'minimum': 0.0,
                         'maximum': 1.5}}),
        t('run_ml_training', '生成结构-性能数据集并训练代理模型，返回 R2',
          {'n': {'type': 'integer', 'minimum': 10, 'maximum': 2000},
           'epochs': {'type': 'integer', 'minimum': 10, 'maximum': 300}}),
        t('export_structure', '导出当前结构为 json 或 svg',
          {'format': {'type': 'string', 'enum': ['json', 'svg']},
           'path': {'type': 'string'}}, ['format']),
        t('export_result', '把当前仿真/特征/逆设计结果导出为 CSV 或 PNG',
          {'kind': {'type': 'string', 'enum': sorted(AIPanel._EXPORT_KINDS)},
           'path': {'type': 'string'}}, ['kind']),
        t('get_skill_doc', '读取内置操作技能文档（结构/仿真/逆设计/特征/曲面/ML/导出）', {}),
        t('set_tab', '切换主界面 tab',
          {'tab': {'type': 'string',
                   'enum': ['structure', 'percolation', 'stretch', 'design',
                            'replay', 'features', 'surface', 'ml', 'manufacturing']}},
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
            bg, fg = c['card2'], c['text']
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

        target_row = QHBoxLayout()
        self.target_label = QLabel(tr('ai_target'))
        self.target_combo = QComboBox()
        self.target_combo.addItem(tr('ai_target_auto'), 'auto')
        for key in self._PAGE_ATTR:
            self.target_combo.addItem(tr(self._PAGE_LABEL[key]), key)
        self.target_combo.currentIndexChanged.connect(self._target_changed)
        target_row.addWidget(self.target_label)
        target_row.addWidget(self.target_combo, 1)
        v.addLayout(target_row)
        self.target_hint = QLabel(tr('ai_target_hint'))
        self.target_hint.setWordWrap(True)
        self.target_hint.setObjectName('hint')
        v.addWidget(self.target_hint)
        actions = QHBoxLayout()
        self.workflow_btn = QPushButton('一条龙 · 准备打印')
        self.workflow_btn.clicked.connect(self._open_workflow)
        self.workflow_stop = QPushButton('停止流程')
        self.workflow_stop.clicked.connect(lambda: self.workflow.cancel())
        self.workflow_btn.hide()
        self.workflow_stop.hide()

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
        self.input.setObjectName('ai_input')
        self.input.setMinimumHeight(40)
        self.input.setClearButtonEnabled(True)
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
        from .workflow import PrintWorkflow
        self.workflow = PrintWorkflow(self)
        self.workflow.changed.connect(self.status.setText)
        self.status.setWordWrap(True)

    def retranslate(self):
        from .i18n import get_lang
        self.workflow_btn.setText('一条龙 · 准备打印' if get_lang()=='zh' else 'Prepare print workflow')
        self.workflow_stop.setText('停止流程' if get_lang()=='zh' else 'Stop workflow')
        self.title_lbl.setText(tr('ai_title'))
        self.input.setPlaceholderText(tr('ai_placeholder'))
        self.target_label.setText(tr('ai_target'))
        self.target_hint.setText(tr('ai_target_hint'))
        self.target_combo.setItemText(0, tr('ai_target_auto'))
        for i in range(1, self.target_combo.count()):
            self.target_combo.setItemText(i, tr(self._PAGE_LABEL[self.target_combo.itemData(i)]))

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
               'theme': win.mode, 'target_page': self.target_combo.currentData(),
               'navigation': 'Use set_tab for navigation. Respect explicit target_page. '
                             'Tool operations navigate automatically. Read results before claiming success.'}
        try:
            f = win.tab_struct.collect_factory()
            ctx['structure'] = dict(unit=f.unit, grid_x=f.grid_x,
                                    grid_y=f.grid_y,
                                    pts=f.n_pts_per_side, seed=f.seed,
                                    perturbation=round(f.perturbation, 3),
                                    expansion_rule=f.expansion_rule,
                                    spectrum=f.line_displacements)
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
        r['configure_learning'] = self._configure_learning
        r['generate_learning_data'] = self._generate_learning
        r['get_learning_status'] = self._learning_status
        r['configure_search'] = self._configure_search
        r['get_design_status'] = self._design_status
        r['list_units'] = lambda: j(units=all_unit_keys())
        r['set_expansion'] = self._set_expansion
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
        r['edit_line_point'] = lambda index, dx, dy, mode='add': \
            self._edit_point(int(index), float(dx), float(dy), mode)
        r['scale_spectrum'] = lambda factor: self._scale_spectrum(
            float(factor))
        r['set_spectrum_preset'] = lambda name: self._set_preset(name)
        r['set_perturbation'] = lambda value: (
            win.tab_struct.pert.setValue(int(round(value * 100))),
            win.tab_struct.push_spec(), j(ok=True))[2]
        r['randomize_seed'] = lambda: (
            win.tab_struct._random_seed(), j(ok=True,
                                             seed=win.tab_struct.seed.value()))[1]
        r['run_percolation'] = lambda stretch=2.0, alpha=0.05: self._run_perc(
            stretch, alpha)
        r['run_stretch'] = lambda stretch=2.2, use_bending=True, \
            use_contact=False: self._run_stretch(stretch, use_bending,
                                                contact=use_contact)
        r['run_inverse_design'] = lambda target='C', budget=100, \
            load_best=False: self._run_inverse(target, budget, load_best)
        r['run_features'] = lambda groups='all': self._tool_features(groups)
        r['set_surface'] = lambda obj='', unit=None, \
            preset=None, amplitude=1.0: self._tool_surface(
                obj, unit, preset, amplitude)
        r['run_ml_training'] = lambda n=40, epochs=80: self._tool_ml(n, epochs)
        r['export_structure'] = lambda format, path=None: self._export(
            format, path)
        r['export_result'] = lambda kind, path=None: \
            self._tool_export_result(kind, path)
        r['set_tab'] = self._navigate
        r['set_theme'] = lambda mode: (
            win._toggle_theme() if win.mode != mode else None,
            j(ok=True, mode=win.mode))[1]
        r['set_lang'] = lambda lang: (
            win.lang_combo.setCurrentIndex(0 if lang == 'zh' else 1),
            j(ok=True))[1]
        r['get_replay_summary'] = lambda: self._replay_summary()
        r['load_replay_structure'] = lambda index: self._load_replay(
            int(index))
        r['start_print_workflow'] = lambda **kwargs: self.workflow.start(**kwargs)
        r['get_print_workflow_status'] = lambda: self.workflow.status()
        r['stop_print_workflow'] = lambda: self.workflow.cancel() or self.workflow.status()
        self.registry = r

    _PAGE_ATTR = {'structure': 'tab_struct', 'stretch': 'tab_sim',
                  'features': 'tab_features', 'ml': 'tab_ml',
                  'design': 'tab_design', 'replay': 'tab_design',
                  'surface': 'tab_surface', 'manufacturing': 'tab_manufacturing'}
    _PAGE_LABEL = {'structure': 'tab_structure', 'stretch': 'tab_sim',
                   'features': 'tab_features', 'ml': 'tab_ml',
                   'design': 'tab_design', 'replay': 'replay_toggle',
                   'surface': 'tab_surface', 'manufacturing': 'tab_manufacturing'}

    def _navigate(self, tab):
        tab = {'percolation': 'stretch', 'simulation': 'stretch'}.get(tab, tab)
        if tab not in self._PAGE_ATTR:
            raise ValueError('unknown target page: ' + str(tab))
        widget = getattr(self.win, self._PAGE_ATTR[tab])
        self.win.tabs.setCurrentWidget(widget)
        if tab in ('design', 'replay'):
            widget.replay_toggle.setChecked(tab == 'replay')
            widget._toggle_replay()
        return dict(ok=True, page=tab,
                    title=self.win.tabs.tabText(self.win.tabs.currentIndex()))

    def _target_changed(self):
        target = self.target_combo.currentData()
        if target != 'auto':
            self._navigate(target)

    def _set_expansion(self, rule, grid_x=None, grid_y=None):
        from dataclasses import replace
        from fslab.cell_rules import RULES
        if rule not in RULES:
            raise ValueError('unknown expansion rule')
        f = self.win.tab_struct.collect_factory()
        f = replace(f, expansion_rule=rule,
                    grid_x=f.grid_x if grid_x is None else int(grid_x),
                    grid_y=f.grid_y if grid_y is None else int(grid_y)).clamped()
        f.build()
        self.win.tab_struct.load_spec(f)
        return dict(ok=True, structure=self.win.tab_struct.collect_factory().__dict__)

    # ---------------- new-interface tools ----------------
    def _current_spectrum(self):
        f = self.win.tab_struct.collect_factory()
        spec = f.line_displacements or f.spectrum()
        return [[float(a), float(b)] for a, b in (spec or [])]

    def _push_spectrum(self, spec):
        win = self.win
        win.tab_struct.editor.set_displacements(spec)
        win.tab_struct._sync_row()
        win.tab_struct._update_preview()
        win.tab_struct.push_spec()
        return spec

    def _edit_point(self, index, dx, dy, mode):
        spec = self._current_spectrum()
        if not spec:
            return _j(ok=False, error='empty spectrum (pts=0)')
        i = max(0, min(len(spec) - 1, index))
        if mode == 'set':
            spec[i] = [dx, dy]
        else:
            spec[i] = [spec[i][0] + dx, spec[i][1] + dy]
        self._push_spectrum(spec)
        return _j(ok=True, index=i, mode=mode, spectrum=spec)

    def _scale_spectrum(self, factor):
        spec = [[a * factor, b * factor]
                for a, b in self._current_spectrum()]
        if not spec:
            return _j(ok=False, error='empty spectrum (pts=0)')
        self._push_spectrum(spec)
        return _j(ok=True, factor=factor, spectrum=spec)

    def _set_preset(self, name):
        if name not in SPECTRUM_PRESETS:
            return _j(ok=False, error='unknown preset %s' % name)
        f = self.win.tab_struct.collect_factory()
        spec = resample_spectrum(SPECTRUM_PRESETS[name],
                                 f.n_pts_per_side)
        self._push_spectrum(spec)
        return _j(ok=True, preset=name, spectrum=spec)

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
        tab = self.win.tab_features
        tab.refresh()
        if tab._feats is None:
            return dict(ok=False, error=tab.status.text())
        d = tab._feats
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
        paths = self.win.tab_surface._obj_files
        pick_idx, pick = 0, paths[0]
        if obj:
            aliases = {'金字塔': 'demo_pyramid', '铁塔': 'reference_paris', '头盔': 'flighthelmet', '椅子': 'sheenchair', '水瓶': 'waterbottle',
                       '肺': 'lung', '心脏': 'heart', '鞋': 'vans', 'T恤': 'tshirt',
                       '衣服': 'reference_shirt', '衬衫': 'reference_shirt', '参考鞋': 'reference_shoe', '背心': 'vest', '围巾': 'scarf', '袖筒': 'sleeve'}
            needle = aliases.get(obj, obj).lower()
            matches = [(i, p) for i, p in enumerate(paths) if needle in os.path.basename(p).lower()]
            if not matches:
                return _j(ok=False, error='model not found: ' + obj)
            pick_idx, pick = matches[0]
        follow_current = unit is None and preset is None
        if unit is None:
            unit = self.win.tab_struct.collect_factory().unit
            if unit in SPECTRUM_PRESETS: unit='square'
        if unit not in all_unit_keys():
            return _j(ok=False, error='unknown unit: ' + unit)
        u = unit
        pk = preset if preset in SPECTRUM_PRESETS else 'auxetic_bow'
        tab = getattr(self.win, 'tab_surface', None)
        if tab is not None:
            tab.obj_combo.setCurrentIndex(pick_idx)
            tab.chk_follow.setChecked(False)
            if tab.unit_combo.findData(u) < 0:
                tab.unit_combo.addItem(u, u)
            tab.unit_combo.setCurrentIndex(tab.unit_combo.findData(u))
            for i in range(tab.preset_combo.count()):
                if tab.preset_combo.itemData(i) == pk:
                    tab.preset_combo.setCurrentIndex(i)
                    break
            if follow_current:
                tab.chk_follow.setChecked(True)
                tab.apply_structure(self.win.tab_struct.collect_factory())
            tab.amp_slider.setValue(
                max(0, min(500, int(float(amplitude) * 100))))
            tab._recompute()
        import numpy as np
        if not hasattr(tab, 'network') or getattr(tab, '_mapping_error', None):
            return _j(ok=False, error=tab.status.text())
        P, segs = tab.network
        return _j(ok=True, obj=os.path.basename(pick), unit=u,
                  verts=int(len(P)), segments=int(len(segs)))

    def _learning_status(self):
        tab = self.win.tab_ml
        out = dict(busy=tab._busy(), status=tab.status.text(), algorithm=tab.algorithm,
                   acquisition=tab.acquisition, label_source=tab.label_mode,
                   samples=0 if tab.X is None else len(tab.X),
                   trained=tab.trained_model is not None)
        if tab.model is not None and tab.X is not None:
            from fslab.mlmodel import TARGET_NAMES, r2_score
            ids = tab.model.val_indices
            prediction = tab.model.predict(tab.X[ids])
            out['validation_r2'] = {key: float(r2_score(tab.Y[ids, i], prediction[:, i]))
                                    for i, key in enumerate(TARGET_NAMES)}
        return out

    def _configure_learning(self, model='mlp', acquisition='random', parameters=None):
        from fslab.learning import MODEL_SPECS, ACQUISITIONS, Regressor
        tab = self.win.tab_ml
        if tab._busy():
            return _j(ok=False, error='learning task already running')
        if acquisition not in ACQUISITIONS:
            return _j(ok=False, error='unknown acquisition')
        Regressor(model, parameters)
        tab.trained_model = None
        tab.algorithm, tab.parameters, tab.acquisition = model, parameters or {}, acquisition
        tab.retranslate()
        return _j(ok=True, model=model, acquisition=acquisition, parameters=tab.parameters)

    def _generate_learning(self, n=40, mode='physics', unit=None):
        if not 10<=int(n)<=2000:
            return _j(ok=False,error='sample count must be 10..2000')
        tab = self.win.tab_ml
        if tab._busy():
            return _j(ok=False, error='learning task already running')
        if mode != 'physics':
            return _j(ok=False, error='invalid label mode')
        unit = unit or self.win.tab_struct.collect_factory().unit
        if unit not in all_unit_keys():
            return _j(ok=False, error='unknown unit')
        if tab.unit_combo.findData(unit) < 0:
            tab.unit_combo.addItem(unit, unit)
        tab.unit_combo.setCurrentIndex(tab.unit_combo.findData(unit))
        tab.mode_combo.setCurrentIndex(tab.mode_combo.findData(mode))
        tab.n_spin.setValue(int(n))
        tab.start_gen()
        return _j(ok=True, state='started', mode=mode, unit=unit,
                  note='Task started; query get_learning_status for completed results.')

    def _tool_ml(self, n, epochs):
        tab = self.win.tab_ml
        if tab._busy():
            return _j(ok=False, error='learning task already running')
        tab.epochs_spin.setValue(int(epochs))
        tab._train_after_generation = True
        return self._generate_learning(n=n, mode='physics')

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

    def _run_stretch(self, stretch, bend, contact=False):
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
            return _j(ok=False, error='unknown target')
        tab = self.win.tab_design
        if tab.worker is not None and tab.worker.isRunning():
            return _j(ok=False, error='inverse task already running')
        if tab.selected_model is not None and target not in SCALARS:
            return _j(ok=False, error='Learned model supports scalar targets only; switch off Use trained model for curve objectives')
        if int(budget) < 1:
            return _j(ok=False, error='budget must be positive')
        tab.set_spec(self.win.tab_struct.collect_factory())
        tab.target_combo.setCurrentText(target)
        tab.budget.setValue(int(budget))
        tab._auto_apply = bool(load_best)
        tab.start_run()
        return _j(ok=True, state='started', algorithm=tab.algorithm,
                  note='Optimization running. Query get_design_status; no result is available yet.')

    def _configure_search(self, algorithm='cem', parameters=None, unit=None, amplitude=None, resume=False, follow_type=None):
        from fslab.search_algorithms import SEARCH_SPECS
        tab = self.win.tab_design
        if tab.worker is not None and tab.worker.isRunning():
            return _j(ok=False, error='inverse task already running')
        if algorithm not in SEARCH_SPECS:
            return _j(ok=False, error='unknown algorithm')
        if unit is not None and unit not in all_unit_keys():
            return _j(ok=False, error='unknown unit')
        if amplitude is not None and not .05 <= amplitude <= 1.:
            return _j(ok=False, error='amplitude must be 0.05..1.0')
        for key, value in (parameters or {}).items():
            descriptor = SEARCH_SPECS[algorithm][2].get(key)
            if descriptor is None or not descriptor[1] <= value <= descriptor[2]:
                return _j(ok=False, error='invalid algorithm parameter')
        tab.algorithm, tab.parameters = algorithm, parameters or {}
        if unit is not None:
            if tab.unit_combo.findData(unit) < 0:
                tab.unit_combo.addItem(unit, unit)
            tab.unit_combo.setCurrentIndex(tab.unit_combo.findData(unit))
        if amplitude is not None:
            tab.amplitude.setValue(amplitude)
        tab.resume.setChecked(False)
        if unit is not None:
            tab.follow_type.setChecked(False)
        if follow_type is not None:
            tab.follow_type.setChecked(bool(follow_type))
        tab.retranslate()
        return _j(ok=True, algorithm=algorithm, parameters=tab.parameters,
                  unit=tab.unit_combo.currentData(), amplitude=tab.amplitude.value(), resume=tab.resume.isChecked())

    def _design_status(self):
        tab = self.win.tab_design
        running = tab.worker is not None and tab.worker.isRunning()
        result = getattr(tab, '_result', None)
        return _j(busy=running, status=tab.status.text(), algorithm=tab.algorithm,
                  best_spec=None if running or not result else result.get('best_spec'),
                  best_objective=None if running or not result else result.get('best_dist'))

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

    _EXPORT_KINDS = {
        'curve_csv': ('tab_sim', '_export_csv'),
        'force_png': ('tab_sim', '_export_force_png'),
        'percolation_png': ('tab_sim', '_export_perc_png'),
        'energy_png': ('tab_sim', '_export_mode_png'),
        'frame_png': ('tab_sim', '_export_frame_png'),
        'features_csv': ('tab_features', '_export_feat_csv'),
        'histogram_csv': ('tab_features', '_export_hist_csv'),
        'fingerprint_png': ('tab_features', '_export_fp_png'),
        'inverse_csv': ('tab_design', '_export_inv_csv'),
        'best_png': ('tab_design', '_export_best_png'),
        'convergence_png': ('tab_design', '_export_conv_png'),
        'best_structure_png': ('tab_design', '_export_struct_png'),
        'canvas_png': ('tab_struct', '_do_export_png'),
    }

    def _tool_export_result(self, kind, path=None):
        spec = self._EXPORT_KINDS.get(kind)
        if spec is None:
            return _j(ok=False, error=f"unknown kind: {kind}")
        tab_attr, method = spec
        for page, attr in self._PAGE_ATTR.items():
            if attr == tab_attr:
                self._navigate(page)
                break
        tab = getattr(self.win, tab_attr)
        ext = "csv" if kind.endswith("_csv") else "png"
        path = os.path.abspath(path or os.path.join(
            os.path.expanduser('~'), 'Documents',
            f'fiberscope_{kind}.{ext}'))
        out = getattr(tab, method)(silent=path)
        if not out:
            return _j(ok=False, error=tab.status.text())
        return _j(ok=True, path=out)

    _TAB_BY_TOOL = {
        'set_structure': 'structure', 'set_line_displacements': 'structure',
        'set_perturbation': 'structure', 'randomize_seed': 'structure',
        'export_structure': 'structure', 'edit_line_point': 'structure',
        'scale_spectrum': 'structure', 'set_spectrum_preset': 'structure',
        'set_expansion': 'structure',
        'run_percolation': 'stretch', 'run_stretch': 'stretch',
        'run_features': 'features', 'run_ml_training': 'ml',
        'configure_learning': 'ml', 'generate_learning_data': 'ml', 'get_learning_status': 'ml',
        'configure_search': 'design', 'get_design_status': 'design',
        'run_inverse_design': 'design', 'load_replay_structure': 'replay',
        'get_replay_summary': 'replay', 'set_surface': 'surface',
    }

    def _run_tool(self, req):
        try:
            if self.workflow.running and req.name not in ('get_print_workflow_status','stop_print_workflow',
                    'get_app_state','get_learning_status','get_design_status','get_skill_doc','list_units'):
                raise RuntimeError('print workflow owns the application; inspect status or stop it first')
            tab_idx = self._TAB_BY_TOOL.get(req.name)
            if tab_idx is not None:
                self._navigate(tab_idx)
            fn = self.registry[req.name]
            out = fn(**req.args)
            if isinstance(out, dict):
                out.setdefault('page', self.win.tabs.tabText(self.win.tabs.currentIndex()))
                if tab_idx == 'structure' and not out.get('error'):
                    out['structure'] = self.win.tab_struct.collect_factory().__dict__
            req.result = json.dumps(out, ensure_ascii=False, default=str)
            ok = isinstance(out, dict) and not out.get('error') and out.get('ok', True) is not False
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

    def _open_workflow(self):
        from PySide6.QtWidgets import QSpinBox, QCheckBox, QDialogButtonBox
        from .i18n import get_lang
        zh = get_lang()=='zh'
        dialog = QDialog(self)
        dialog.setWindowTitle('一条龙 · 准备打印' if zh else 'Prepare print workflow')
        layout = QFormLayout(dialog)
        description = QLabel('结构 → 仿真 → 特征 → 物理数据与训练 → C型逆设计 → 复核 → 三维 → 实体 → 拓竹' if zh else 'Structure → simulation → features → data/training → C inverse design → verification → 3D → solid → Bambu')
        description.setWordWrap(True)
        layout.addRow(description)
        samples, iterations = QSpinBox(), QSpinBox()
        samples.setRange(10,200); samples.setValue(10)
        iterations.setRange(1,1000000); iterations.setValue(20)
        curved = QCheckBox('导出当前曲面形状' if zh else 'Export current curved surface')
        layout.addRow('样本数' if zh else 'Samples',samples)
        layout.addRow('优化次数' if zh else 'Iterations',iterations)
        layout.addRow(curved)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec():
            try:
                self.workflow.start(samples.value(),iterations.value(),curved.isChecked())
            except (ValueError,RuntimeError) as exc:
                self.status.setText(str(exc))

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
        if self.worker is not None and self.worker.isRunning():
            return
        text = self.input.text().strip()
        if not text:
            return
        command = text.strip(' 。!！')
        if command in ('停止流程','取消流程','停止完整流程'):
            self.workflow.cancel()
            self.input.clear()
            return
        if self.workflow.running:
            self.status.setText('流程正在运行；输入“停止流程”可取消。')
            return
        if command.startswith(('决赛录像流程','请执行决赛录像流程')):
            self.input.clear()
            self._log.append(('user', text))
            self._add_row(Bubble(text,'user',self.win.mode),'right')
            try:
                from .recording_command import recording_options
                options=recording_options(text)
                self.workflow.start(**options)
                answer=('已启动：%s · 拉伸比%g · %d个物理标注样本 · %s型逆设计%d次预算 · 金字塔 · 制造 · 拓竹。'
                        '输入“停止流程”可取消。') % (unit_display(options['demo_unit'],get_lang()),
                        options['stretch'],options['samples'],options['target'],options['iterations'])
            except (ValueError,RuntimeError) as exc:
                answer=str(exc)
            self._log.append(('answer',answer))
            self._add_row(Bubble(answer,'assistant',self.win.mode),'left')
            return
        if command in ('一条龙','一键打印','从头到尾','准备打印','完整流程','完整运行','从结构到打印'):
            self.input.clear()
            try:
                self.workflow.start(curved=True)
            except (ValueError,RuntimeError) as exc:
                self.status.setText(str(exc))
            return
        local = {'结构生成': 'structure', '结构': 'structure',
                 '力学模拟': 'stretch', '特征分析': 'features',
                 '机器学习': 'ml', '逆向设计': 'design', '逆设计': 'design',
                 '探索回放': 'replay', '三维曲面': 'surface', '制造': 'manufacturing', '连续制造': 'manufacturing'}
        request = text.strip(' 。.!！').lower()
        for prefix in ('请切换到', '切换到', '请打开', '打开', '进入', 'go to ', 'open '):
            if request.startswith(prefix):
                request = request[len(prefix):].strip()
                break
        page = local.get(request, request if request in self._PAGE_ATTR else None)
        if page:
            self._navigate(page)
            self.input.clear()
            self._log.append(('user', text))
            self._add_row(Bubble(text, 'user', self.win.mode), 'right')
            answer = tr('ai_navigated') + tr(self._PAGE_LABEL[page])
            self._log.append(('answer', answer))
            self._add_row(Bubble(answer, 'ai', self.win.mode), 'left')
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

    def _workflow_tool_started(self, stage, args):
        name='workflow.'+stage
        self._on_note(name)
        card=self._cards[-1]
        card.set_args(json.dumps(args,ensure_ascii=False))
        card.body.setVisible(True)
        card.toggle.setText('▾ '+name)
        if not hasattr(self,'_workflow_cards'): self._workflow_cards={}
        self._workflow_cards[stage]=(card,dict(args))
        self._scroll_bottom()

    def _workflow_tool_finished(self, stage, result, ok):
        pending=getattr(self,'_workflow_cards',{}).pop(stage,None)
        if pending is None: return
        card,args=pending
        text=json.dumps(result,ensure_ascii=False,default=str)
        card.finish(text,ok)
        self._log.append(('tool',card.name,args,text,ok))
        self._scroll_bottom()

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

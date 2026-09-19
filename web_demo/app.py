"""CPU judge demo reusing the FiberScope 3.0 numerical core. Run: python web_demo/app.py."""
from pathlib import Path
from dataclasses import replace
import os,sys,tempfile,json,shutil
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('FIBERNET_ROOT',str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED','False')
sys.path.insert(0,str(ROOT/'FiberScope'))
import numpy as np
import gradio as gr
import plotly.graph_objects as go
from fslab.structure import StructureFactory,all_unit_keys,unit_display
from fslab.simcache import RunConfig,run_stretch
from fslab.features import compute_features,feature_name
from fslab.mlmodel import gen_dataset,TARGET_NAMES
from fslab.learning import Regressor,MODEL_SPECS
from fslab.inverse import run_inverse,curve_of,target_curve
from fslab.surface_mapping import load_obj
from fslab.manufacturing import compile_planar,compile_surface
from fslab.print_export import PrintSettings,build_solid,export_solid,export_route

COLORS=dict(paper_bgcolor='#ffffff',plot_bgcolor='#ffffff',font=dict(color='#19344d',family='Arial'))

def cleanup_session(state):
    folder=Path(state.get('directory','')).resolve()
    if folder.parent==Path(tempfile.gettempdir()).resolve() and folder.name.startswith('fiberscope-web-'):
        shutil.rmtree(folder,ignore_errors=True)

def line_figure(points,edges,three=False,title='纤维网络'):
    points=np.asarray(points);edges=np.asarray(edges,int)
    coords=np.full((len(edges),3,points.shape[1]),np.nan)
    coords[:,:2]=points[edges];coords=coords.reshape(-1,points.shape[1])
    values=dict(x=coords[:,0],y=coords[:,1],mode='lines',line=dict(color='#126bc5',width=3),hoverinfo='skip')
    trace=go.Scatter3d(z=coords[:,2],**values) if three else go.Scatter(**values)
    figure=go.Figure(trace);figure.update_layout(title=title,margin=dict(l=20,r=20,t=45,b=20),height=470,**COLORS)
    if three:figure.update_layout(scene=dict(aspectmode='data',xaxis_title='X',yaxis_title='Y',zaxis_title='Z'))
    else:figure.update_yaxes(scaleanchor='x',scaleratio=1)
    return figure

class Demo:
    """All mutable scientific state belongs to one browser session."""
    def generate(self,unit,grid,amplitude,seed):
        if unit not in all_unit_keys() or not 1<=int(grid)<=4 or not 0<=float(amplitude)<=1:raise gr.Error('参数超出在线演示范围')
        factory=StructureFactory(unit=unit,grid_x=int(grid),grid_y=int(grid),n_pts_per_side=5,perturbation=float(amplitude),seed=int(seed))
        graph=factory.build();pos=graph.node_positions();edges=graph.edge_array()[:,:2]
        state={'factory':factory,'directory':tempfile.mkdtemp(prefix='fiberscope-web-')}
        return state,line_figure(pos,edges),f'已生成 {unit_display(unit,"zh")} · {len(pos)} 节点 / {len(edges)} 条边'

    def require(self,state):
        if not state or 'factory' not in state:raise gr.Error('请先生成结构')
        return state['factory']

    def simulate(self,state,stretch,weld):
        factory=self.require(state)
        if not 1.05<=float(stretch)<=3:raise gr.Error('拉伸比须在 1.05–3 之间')
        cfg=RunConfig(target_stretch=float(stretch),num_steps=8000,n_increments=60,use_contact=False,weld_intersections=bool(weld))
        run=run_stretch(factory,cfg,cache_dir=str(Path(state['directory'])/'sim'))
        state=dict(state,run=run)
        curve=go.Figure(go.Scatter(x=run.strain_levels,y=run.force_curve,mode='lines',line=dict(color='#126bc5')))
        curve.update_layout(title='真实物理仿真 · 反力曲线',xaxis_title='拉伸比',yaxis_title='反力（模型单位）',height=400,**COLORS)
        return state,line_figure(run.frames_xy[-1],run.edges,title='拉伸后结构'),curve,f'仿真完成 · {run.n_frames} 帧'

    def features(self,state):
        graph=self.require(state).build();feats=compute_features(np.asarray(graph.node_positions())[:,:2],graph.edge_array()[:,:2])
        rows=[[feature_name(k,'zh'),float(v)] for k,v in feats.items() if np.isscalar(v) and isinstance(v,(int,float,np.number)) and np.isfinite(v)]
        return rows

    def train(self,state,count,model,progress=gr.Progress()):
        factory=self.require(state);count=int(count)
        if not 12<=count<=300 or model not in MODEL_SPECS:raise gr.Error('请检查样本数量或模型')
        folder=Path(state['directory']);path=folder/f'{factory.unit}-{count}-physics.npz'
        def report(*args):
            if args and isinstance(args[0],(int,float)):progress(min(float(args[0])/count,.95),desc='生成并物理标注')
        X,Y=gen_dataset(factory.unit,n=count,seed0=factory.seed,path=str(path),mode='physics',progress_cb=report)
        estimator=Regressor(model,seed=factory.seed);estimator.train(X,Y,epochs=150)
        ids=estimator.val_indices;pred=estimator.predict(X[ids]);truth=Y[ids]
        score=[];figure=go.Figure()
        for i,label in enumerate(TARGET_NAMES):
            denom=float(np.sum((truth[:,i]-truth[:,i].mean())**2))
            r2=1-float(np.sum((pred[:,i]-truth[:,i])**2))/denom if denom>1e-12 else None
            score.append([label,round(r2,4) if r2 is not None else None,len(ids)])
        figure.add_trace(go.Scatter(x=truth[:,0],y=pred[:,0],mode='markers',name='留出样本',marker=dict(color='#126bc5',size=8)))
        lo=float(min(truth[:,0].min(),pred[:,0].min()));hi=float(max(truth[:,0].max(),pred[:,0].max()))
        figure.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode='lines',name='理想预测',line=dict(dash='dash')))
        figure.update_layout(title='峰值反力 · 预测与真实（留出集）',xaxis_title='真实值',yaxis_title='预测值',height=400,**COLORS)
        return dict(state,model=estimator),figure,score,str(path),f'{count} 个真实物理标签 · {len(ids)} 个留出样本'

    def inverse(self,state,budget,target,progress=gr.Progress()):
        factory=self.require(state);budget=int(budget)
        if not 4<=budget<=80 or target not in ('J','C','linear'):raise gr.Error('请检查搜索预算或目标')
        def build(unit,pert,ld):return replace(factory,unit=unit,perturbation=pert,line_displacements=ld,spectrum_resolved=False).build()
        def callback(record,run):progress(min(record.eval_id/budget,.99),desc='仿真式逆向设计')
        result=run_inverse(build,target,budget=budget,seed=factory.seed,fixed_unit=factory.unit,pts=5,amplitude=.6,callback=callback)
        spec=result['best_spec'];best=replace(factory,unit=spec['unit'],perturbation=spec['pert'],line_displacements=spec['line_displacements'],spectrum_resolved=False)
        graph=best.build();figure=go.Figure()
        figure.add_trace(go.Scatter(y=target_curve(target),name='目标'))
        figure.add_trace(go.Scatter(y=curve_of(result['best_run']),name='最优仿真'))
        figure.update_layout(title='归一化力学曲线',height=400,**COLORS)
        updated=dict(state,factory=best);updated.pop('network',None);updated.pop('run',None)
        return updated,line_figure(graph.node_positions(),graph.edge_array()[:,:2],title='最优结构'),figure,f'完成 {result["evaluations"]} 次物理评估 · 目标误差 {result["best_dist"]:.4g}'

    def surface(self,state,model):
        factory=self.require(state)
        choices={'金字塔':'demo_pyramid.obj','衣袖':'demo_sleeve.obj','围巾':'demo_scarf.obj'}
        if model not in choices:raise gr.Error('未知曲面')
        vertices,faces=load_obj(ROOT/'FiberScope/assets/obj'/choices[model],target_faces=48)
        network=compile_surface(vertices,faces,factory)
        return dict(state,network=network),line_figure(network.positions,network.edges,True,'真实单元曲面映射'),f'{model} · {len(faces)} 个曲面网格 · 闭合路径 {len(network.route_edges)} 条边'

    def manufacture(self,state,source,diameter,extent,progress=gr.Progress()):
        factory=self.require(state);curved=source=='曲面结构'
        if curved and 'network' not in state:raise gr.Error('请先映射曲面')
        if not 1<=float(diameter)<=4 or not 60<=float(extent)<=250:raise gr.Error('请检查制造尺寸')
        network=state['network'] if curved else compile_planar(factory)
        if len(network.edges)>18000:raise gr.Error('在线资源有限，请减少网络数量；完整规模可使用桌面版')
        settings=PrintSettings(width=float(extent),depth=float(extent),height=float(diameter),diameter=float(diameter),curved=curved)
        solid=build_solid(network,settings,progress=lambda p:progress(min(p/100,.99),desc='构建圆柱纤维实体'))
        folder=Path(state['directory']);stl=folder/'FiberScope.stl';route=folder/'FiberScope-route.csv'
        export_solid(stl,solid);export_route(route,network,solid.centers)
        return str(stl),str(route),f'封闭实体已生成 · {len(solid.faces):,} 三角面 · STL 单位 mm'

def create_app():
    demo=Demo()
    with gr.Blocks(title='FiberScope 3.0 · 纤维网络材料设计',theme=gr.themes.Soft(primary_hue='blue'),delete_cache=(3600,86400)) as app:
        gr.Markdown('# FiberScope 3.0\n### 从基本单元到可制造的纤维网络\n结构生成 · 物理仿真 · 机器学习 · 逆向设计 · 三维曲面 · STL 制造')
        gr.Markdown('[GitHub 开源代码](https://github.com/GellmanSparrowS/fibernet) · [下载 Windows 完整版](https://github.com/GellmanSparrowS/fibernet/releases/tag/fiberscope-v3.0.0)')
        state=gr.State({},time_to_live=3600,delete_callback=cleanup_session)
        with gr.Tab('1 · 结构生成'):
            with gr.Row():
                unit=gr.Dropdown([(unit_display(k,'zh'),k) for k in all_unit_keys()],value='square',label='基本单元')
                grid=gr.Slider(1,4,value=2,step=1,label='网络数量（每方向）')
                amp=gr.Slider(0,1,value=.35,step=.01,label='共享参数扰动幅度')
                seed=gr.Number(value=7,precision=0,label='随机种子')
            generate=gr.Button('生成结构',variant='primary');structure=gr.Plot();status=gr.Textbox(label='状态',interactive=False)
            generate.click(demo.generate,[unit,grid,amp,seed],[state,structure,status],concurrency_id='science',api_name='generate')
        with gr.Tab('2 · 拉伸仿真'):
            stretch=gr.Slider(1.05,3,value=2,step=.05,label='目标拉伸比');weld=gr.Checkbox(True,label='焊接锚点')
            simulate=gr.Button('运行物理仿真',variant='primary')
            with gr.Row():deformed=gr.Plot();curve=gr.Plot()
            sim_status=gr.Textbox(label='状态',interactive=False)
            simulate.click(demo.simulate,[state,stretch,weld],[state,deformed,curve,sim_status],concurrency_id='science',api_name='simulate')
        with gr.Tab('3 · 特征分析'):
            analyze=gr.Button('计算结构与孔隙特征',variant='primary');features=gr.Dataframe(headers=['特征','数值'],interactive=False)
            analyze.click(demo.features,state,features,concurrency_id='science',api_name='features')
        with gr.Tab('4 · 机器学习'):
            with gr.Row():
                count=gr.Slider(12,300,value=30,step=1,label='物理标注样本数量')
                model=gr.Dropdown([(v[0],k) for k,v in MODEL_SPECS.items()],value='mlp',label='模型')
            train=gr.Button('生成样本并训练',variant='primary');prediction=gr.Plot()
            scores=gr.Dataframe(headers=['预测目标','留出集 R²','留出样本数'],interactive=False)
            dataset=gr.File(label='下载物理标注数据');ml_status=gr.Textbox(label='状态',interactive=False)
            train.click(demo.train,[state,count,model],[state,prediction,scores,dataset,ml_status],concurrency_id='science',api_name='train')
        with gr.Tab('5 · AI 逆向设计'):
            with gr.Row():target=gr.Dropdown(['J','C','linear'],value='J',label='目标曲线');budget=gr.Slider(4,80,value=12,step=1,label='物理评估预算')
            optimize=gr.Button('仿真式搜索并应用最优结构',variant='primary')
            with gr.Row():best=gr.Plot();comparison=gr.Plot()
            inverse_status=gr.Textbox(label='状态',interactive=False)
            optimize.click(demo.inverse,[state,budget,target],[state,best,comparison,inverse_status],concurrency_id='science',api_name='inverse')
        with gr.Tab('6 · 三维曲面'):
            surface_model=gr.Dropdown(['金字塔','衣袖','围巾'],value='金字塔',label='曲面模型')
            map_button=gr.Button('映射当前结构',variant='primary');mapped=gr.Plot();surface_status=gr.Textbox(label='状态',interactive=False)
            map_button.click(demo.surface,[state,surface_model],[state,mapped,surface_status],concurrency_id='science',api_name='surface')
        with gr.Tab('7 · 制造导出'):
            with gr.Row():
                source=gr.Radio(['平面结构','曲面结构'],value='曲面结构',label='结构来源')
                diameter=gr.Slider(1,4,value=2,step=.1,label='纤维直径 / mm')
                extent=gr.Slider(60,250,value=150,step=10,label='打印空间 / mm')
            manufacture=gr.Button('生成实体并导出 STL',variant='primary')
            with gr.Row():stl=gr.File(label='STL · 可导入拓竹');route=gr.File(label='闭合路径 CSV')
            print_status=gr.Textbox(label='状态',interactive=False)
            manufacture.click(demo.manufacture,[state,source,diameter,extent],[stl,route,print_status],concurrency_id='science',api_name='manufacture')
        gr.Markdown('在线版使用共享 CPU，任务依次计算。下载 STL 后可在本地拓竹软件中切片。完整 AI 助手、自定义单元、OBJ 导入和长预算任务请使用桌面版。\n\n制作：复旦大学高分子科学系 杨云浩 · 致谢世界人工智能开源大赛')
    return app.queue(default_concurrency_limit=1,max_size=12)

if __name__=='__main__':create_app().launch(server_name='0.0.0.0',server_port=int(os.environ.get('PORT','7860')),show_error=False)

'use client';

import {
  Activity, AlertTriangle, ArrowRight, Braces, Check, ChevronDown,
  ChevronLeft, ChevronRight, CircleDot, Clock3, Database, GitBranch, Globe2,
  MessageSquareText, Pause, Play, RotateCcw, Search, Sparkles,
  TerminalSquare, Wrench, Zap,
} from 'lucide-react';
import Link from 'next/link';
import { useEffect, useMemo, useRef, useState } from 'react';

type NodeStatus = 'success' | 'running' | 'waiting' | 'failed';
type AgentNode = { id: string; label: string; type: string; status: NodeStatus; confidence?: number; tokens?: number; duration?: string; dependencies: string[]; x: number; y: number };
type LoopStep = { turn: number; kind: 'thought' | 'tool' | 'result' | 'error'; title: string; detail: string; meta?: string; data?: unknown; toolName?: string; compression?:Record<string,unknown>; compressionIndex?:number };
type TaskOutput = { task_id?: string; status?: string; confidence?: number; token_usage?: number; output?: unknown; metadata?: Record<string,unknown>; [key:string]:unknown };
type RunSummary = { id: string; query: string; status: string; started_at: number; elapsed_ms: number; event_count: number };
type RunDetail = RunSummary & { nodes: Array<Omit<AgentNode, 'x' | 'y' | 'duration'> & { layer: number; row: number }>; loops: Record<string, Record<string, unknown>[]>; outputs:Record<string,TaskOutput>; states: Record<string, unknown>[]; events: Record<string, unknown>[] };

function str(value: unknown, fallback = ''): string {
  if (typeof value === 'string') return value || fallback;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  return fallback;
}

function normalizeRunDetail(raw: unknown): RunDetail {
  const value = raw && typeof raw === 'object' ? raw as Record<string, unknown> : {};
  return {
    id: str(value.id), query: str(value.query), status: str(value.status, 'running'),
    started_at: Number(value.started_at || 0), elapsed_ms: Number(value.elapsed_ms || 0), event_count: Number(value.event_count || 0),
    nodes: Array.isArray(value.nodes) ? value.nodes as RunDetail['nodes'] : [],
    loops: value.loops && typeof value.loops === 'object' ? value.loops as RunDetail['loops'] : {},
    outputs: value.outputs && typeof value.outputs === 'object' ? value.outputs as RunDetail['outputs'] : {},
    states: Array.isArray(value.states) ? value.states as RunDetail['states'] : [],
    events: Array.isArray(value.events) ? value.events as RunDetail['events'] : [],
  };
}

const flow = [['PLAN', '规划', 'done'], ['DISPATCH', '调度', 'done'], ['RESEARCH', '研究', 'active'], ['COLLECT', '汇总', 'idle'], ['SYNTHESIZE', '合成', 'idle'], ['RED / BLUE', '对抗', 'idle']];

function StatusDot({ status }: { status: string }) { return <span className={`status-dot status-${status}`} />; }
function Metric({ label, value, hint }: { label: string; value: string; hint: string }) { return <div className="metric"><span>{label}</span><strong>{value}</strong><small>{hint}</small></div>; }

function cleanUrl(value: unknown) {
  const raw=str(value).trim();
  const markdown=raw.match(/^\[https?:\/\/[^\]]+\]\((https?:\/\/[^)]+)\)$/);
  const candidate=markdown?.[1]||raw;
  try{const url=new URL(candidate);return ['http:','https:'].includes(url.protocol)?url.toString():''}catch{return ''}
}

function ToolPayload({ step }: { step: LoopStep }) {
  if(step.toolName==='web_search'&&Array.isArray(step.data)){
    const sources=step.data.filter((item):item is Record<string,unknown>=>Boolean(item&&typeof item==='object'));
    return <div className="source-results"><div className="source-count"><Globe2/>{sources.length} 个搜索结果</div>{sources.map((source,index)=>{
      const url=cleanUrl(source.url);let host='未知来源';try{host=url?new URL(url).hostname.replace(/^www\./,''):host}catch{}
      return <details className="source-card" key={`${url}-${index}`} open={index===0}><summary><span className="source-index">{index+1}</span><span className="source-heading"><strong>{str(source.title,'无标题')}</strong><small>{host}</small></span><ChevronDown/></summary><div className="source-body"><p>{str(source.snippet,'暂无摘要')}</p>{url&&<a href={url} target="_blank" rel="noreferrer">打开来源 <ArrowRight/></a>}</div></details>;
    })}</div>
  }
  if(step.toolName==='browser_batch'&&step.data&&typeof step.data==='object'){
    const batch=step.data as Record<string,unknown>;
    const pages=Array.isArray(batch.results)?batch.results.filter((item):item is Record<string,unknown>=>Boolean(item&&typeof item==='object')):[];
    return <div className="browser-batch-results">
      <div className="source-count"><Globe2/>{pages.length} 个网页已批量读取</div>
      <div className="browser-batch-grid">{pages.map((page,index)=>{
        const url=cleanUrl(page.url);let host='未知来源';try{host=url?new URL(url).hostname.replace(/^www\./,''):host}catch{}
        const failed=Boolean(page.error);
        return <article className={`browser-batch-card ${failed?'batch-failed':''}`} key={`${url}-${index}`}>
          <div className="batch-card-head"><span className="source-index">{index+1}</span><span><strong>{str(page.title||host,'网页正文')}</strong><small>{host}</small></span></div>
          {failed?<p className="batch-error">{str(page.error)}</p>:<details className="batch-preview"><summary>查看提取正文 <ChevronDown/></summary><p>{str(page.content,'未提取到正文')}</p></details>}
          {url&&<a className="batch-open-link" href={url} target="_blank" rel="noreferrer">打开原网页 <ArrowRight/></a>}
        </article>
      })}</div>
    </div>
  }
  if(step.toolName==='browser'&&step.data&&typeof step.data==='object'){
    const page=step.data as Record<string,unknown>,url=cleanUrl(page.url);
    return <div className="browser-result"><div><Globe2/><span><strong>{str(page.title,'网页正文')}</strong><small>{url||'未知 URL'}</small></span></div><p>{str(page.content||page.error,'未提取到正文')}</p>{url&&<a href={url} target="_blank" rel="noreferrer">打开原网页 <ArrowRight/></a>}</div>
  }
  if(step.data!==undefined)return <pre className="tool-json">{JSON.stringify(step.data,null,2)}</pre>;
  return <p>{step.detail}</p>;
}

function ToolResult({ step, runId, nodeId }: { step: LoopStep; runId: string; nodeId: string }) {
  const batchCount=step.toolName==='browser_batch'&&step.data&&typeof step.data==='object'&&Array.isArray((step.data as Record<string,unknown>).results)?((step.data as Record<string,unknown>).results as unknown[]).length:null;
  const summary=batchCount!==null
    ? `返回 ${batchCount} 个网页`
    : Array.isArray(step.data)
    ? `返回 ${step.data.length} 条结果`
    : step.data&&typeof step.data==='object'
      ? `返回 ${Object.keys(step.data as Record<string,unknown>).length} 个字段`
      : step.detail||'查看返回内容';
  const compression=step.compression;
  const before=Number(compression?.before_chars||0),after=Number(compression?.after_chars||0);
  const ratio=before?Math.round(after/before*100):0;
  const compressionHref=`/compression?run=${encodeURIComponent(runId)}&node=${encodeURIComponent(nodeId)}&call=${encodeURIComponent(step.meta||'')}&ci=${step.compressionIndex??0}`;
  return <details className="tool-result-details">
    <summary><span>{summary}</span><span className="result-toggle"><span className="show-result">展开结果</span><span className="hide-result">收起结果</span><ChevronDown/></span></summary>
    <div className="tool-result-body"><ToolPayload step={step}/>{compression&&<a className="compression-link" href={compressionHref}><Database/><span className="compression-link-text"><strong>上下文压缩</strong><small>{str(compression.strategy,'context compression')} · {str(compression.tool,'context')}</small></span><span className="compression-link-stats">{before.toLocaleString()} → {after.toLocaleString()} chars</span><span className="compression-ratio-chip">{ratio}% 保留</span><span className="compression-link-cta">新页面查看对比 <ArrowRight/></span></a>}</div>
  </details>;
}

function ToolInvocation({ step }: { step: LoopStep }) {
  const args=step.data&&typeof step.data==='object'?step.data as Record<string,unknown>:{};
  const entries=Object.entries(args);
  return <div className="tool-call-card">
    <div className="tool-call-head"><Wrench/><div><span>TOOL</span><strong>{step.toolName||'unknown_tool'}</strong></div></div>
    {step.meta&&<div className="tool-call-id"><span>call_id</span><code>{step.meta}</code></div>}
    <div className="tool-param-label">PARAMETERS</div>
    {entries.length===0?<div className="tool-empty">无参数</div>:<div className="tool-params">{entries.map(([key,value])=>{
      const isUrl=key.toLowerCase().includes('url'),url=isUrl?cleanUrl(value):'';
      return <div className="tool-param" key={key}><span>{key}</span>{url?<a href={url} target="_blank" rel="noreferrer">{url}<ArrowRight/></a>:<code>{typeof value==='string'?value:JSON.stringify(value,null,2)}</code>}</div>;
    })}</div>}
  </div>;
}

function parseToolArguments(raw: unknown): Record<string,unknown> {
  if(raw&&typeof raw==='object')return raw as Record<string,unknown>;
  if(typeof raw!=='string'||!raw.trim())return {};
  try{const parsed=JSON.parse(raw);return parsed&&typeof parsed==='object'?parsed:{value:parsed}}catch{return {raw_arguments:raw}}
}

function DagGraph({ graphNodes, selected, onSelect }: { graphNodes: AgentNode[]; selected: string; onSelect: (id: string) => void }) {
  const map = Object.fromEntries(graphNodes.map((n) => [n.id, n]));
  const edges=graphNodes.flatMap(node=>node.dependencies.filter(dep=>map[dep]&&dep!==node.id).map(dep=>({from:dep,to:node.id})));
  const adjacency=edges.reduce<Record<string,string[]>>((acc,edge)=>{(acc[edge.from]??=[]).push(edge.to);return acc},{});
  const hasAlternatePath=(candidate:{from:string;to:string})=>{
    const queue=(adjacency[candidate.from]||[]).filter(next=>next!==candidate.to),seen=new Set<string>();
    while(queue.length){const current=queue.shift()!;if(current===candidate.to)return true;if(seen.has(current))continue;seen.add(current);queue.push(...(adjacency[current]||[]))}
    return false;
  };
  const visibleEdges=edges.filter(edge=>!hasAlternatePath(edge));
  const width=Math.max(980,...graphNodes.map(node=>node.x+290));
  const height=Math.max(590,...graphNodes.map(node=>node.y+170));
  const layers=[...new Set(graphNodes.map(node=>node.x))].sort((a,b)=>a-b);
  return <div className="dag-scroll"><div className="dag-stage" style={{width,height}}>
    <svg className="dag-lines" viewBox={`0 0 ${width} ${height}`}><defs><marker id="dag-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L8,4 L0,8 Z" /></marker></defs>
      {visibleEdges.map(edge=>{const source=map[edge.from],target=map[edge.to];const x1=source.x+230,y1=source.y+50,x2=target.x-8,y2=target.y+50,mid=x1+(x2-x1)/2;const related=selected===edge.from||selected===edge.to;return <path className={`${related?'edge-related':''} ${selected&&!related?'edge-muted':''}`} key={`${edge.from}-${edge.to}`} d={`M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`} markerEnd="url(#dag-arrow)"/>})}
    </svg>
    {layers.map((x,index)=><div className="layer-label" style={{left:x}} key={x}>LAYER {String(index+1).padStart(2,'0')}</div>)}
    {graphNodes.map((n) => <button key={n.id} title={n.label} className={`dag-node node-${n.status} ${selected === n.id ? 'selected' : ''}`} style={{ left: n.x, top: n.y }} onClick={() => onSelect(n.id)}>
      <div className="node-topline"><span className="node-id">{n.id}</span><StatusDot status={n.status} /></div><strong>{n.label}</strong>
      <div className="node-meta"><span>{n.type}</span>{n.confidence !== undefined && <span>{Math.round(n.confidence * 100)}%</span>}<span>{n.duration}</span></div>{n.status === 'running' && <span className="node-progress"><i /></span>}
    </button>)}
  </div></div>;
}

function DataValue({ value, depth=0 }: { value:unknown; depth?:number }) {
  if(value===null||value===undefined)return <span className="data-null">null</span>;
  if(typeof value==='boolean')return <span className="data-bool">{String(value)}</span>;
  if(typeof value==='number')return <span className="data-number">{value.toLocaleString()}</span>;
  if(typeof value==='string'){
    const url=cleanUrl(value);
    if(url)return <a className="data-link" href={url} target="_blank" rel="noreferrer">{url}<ArrowRight/></a>;
    return <span className="data-string">{value}</span>;
  }
  if(Array.isArray(value))return <div className="data-array">{value.map((item,index)=><div className="data-array-row" key={index}><b>{index+1}</b><DataValue value={item} depth={depth+1}/></div>)}</div>;
  if(typeof value==='object')return <div className={`data-object depth-${Math.min(depth,2)}`}>{Object.entries(value as Record<string,unknown>).map(([key,item])=><div className="data-field" key={key}><code>{key}</code><DataValue value={item} depth={depth+1}/></div>)}</div>;
  return <span>{str(value)}</span>;
}

function UpstreamOutput({ source, output, onSelect }: { source:AgentNode; output?:TaskOutput; onSelect:(id:string)=>void }) {
  const payload=output?.output;
  const chars=typeof payload==='string'?payload.length:payload===undefined?0:JSON.stringify(payload).length;
  return <details className="upstream-card">
    <summary><StatusDot status={source.status}/><span><code>{source.id}</code><strong>{source.label}</strong></span><span className="upstream-size">{chars?`${chars.toLocaleString()} chars`:'暂无输出'}</span><ChevronDown/></summary>
    <div className="upstream-body">
      <div className="upstream-meta"><span>TYPE <b>{source.type}</b></span><span>STATUS <b>{output?.status||source.status}</b></span><span>CONFIDENCE <b>{typeof output?.confidence==='number'?`${Math.round(output.confidence*100)}%`:'—'}</b></span><span>TOKENS <b>{output?.token_usage?.toLocaleString()||'—'}</b></span></div>
      <div className="upstream-payload"><div className="payload-label">INJECTED OUTPUT</div>{payload===undefined?<p className="context-empty">该上游任务尚未产生可注入的数据。</p>:<DataValue value={payload}/>}</div>
      {output?.metadata&&Object.keys(output.metadata).length>0&&<details className="nested-data"><summary>METADATA <ChevronDown/></summary><DataValue value={output.metadata}/></details>}
      <button className="jump-upstream" onClick={()=>onSelect(source.id)}>查看该 Agent 的完整执行过程 <ArrowRight/></button>
    </div>
  </details>;
}

function InputContext({ node, query, allNodes, outputs, onSelect }: { node: AgentNode; query: string; allNodes: AgentNode[]; outputs:Record<string,TaskOutput>; onSelect: (id:string)=>void }) {
  const upstream=node.dependencies.map(id=>allNodes.find(item=>item.id===id)).filter((item):item is AgentNode=>Boolean(item));
  return <section className="context-panel">
    <div className="context-heading"><span>INPUT CONTEXT</span><b>{upstream.length?`${upstream.length} UPSTREAM`:'ROOT INPUT'}</b></div>
    <div className="research-goal"><MessageSquareText/><div><span>RESEARCH QUESTION</span><p>{query||'未记录研究问题'}</p></div></div>
    {upstream.length>0&&<div className="upstream-section">
      <div className="context-divider"><span>MERGED AGENT OUTPUTS</span><i/></div>
      <div className="upstream-list">{upstream.map(source=><UpstreamOutput key={source.id} source={source} output={outputs[source.id]} onSelect={onSelect}/>)}</div>
      <p className="context-note"><GitBranch/>以上是构建当前 Agent prompt 时注入的上游结果；点击卡片可检查实际字段和值。</p>
    </div>}
  </section>;
}

function TaskResultPanel({ node, result }: { node:AgentNode; result?:TaskOutput }) {
  const output=result?.output;
  const outputLength=typeof output==='string'?output.length:output===undefined?0:JSON.stringify(output).length;
  const metadata=result?.metadata&&Object.keys(result.metadata).length?result.metadata:undefined;
  return <section className="task-result-panel">
    <div className="task-result-heading"><span>AGENT RESULT</span><b>{result?'RECORDED':'PENDING'}</b></div>
    <div className="task-result-fields">
      <div><span>task_id</span><code>{result?.task_id||node.id}</code></div>
      <div><span>task_type</span><code>{node.type}</code></div>
      <div><span>status</span><code className={`value-${result?.status||node.status}`}>{result?.status||node.status}</code></div>
      <div><span>confidence</span><code>{typeof result?.confidence==='number'?result.confidence.toFixed(2):'—'}</code></div>
      <div><span>token_usage</span><code>{result?.token_usage?.toLocaleString()||'—'}</code></div>
    </div>
    <details className="result-output-details">
      <summary><span><Braces/><span><b>output</b><small>{outputLength?`${outputLength.toLocaleString()} characters`:'暂无输出'}</small></span></span><span className="result-toggle"><span className="show-result">展开正文</span><span className="hide-result">收起正文</span><ChevronDown/></span></summary>
      <div className="result-output-body">{output===undefined?<p className="context-empty">任务尚未产生输出。</p>:typeof output==='string'?<pre>{output}</pre>:<DataValue value={output}/>}</div>
    </details>
    {metadata&&<details className="result-metadata"><summary><span>metadata</span><span>{Object.keys(metadata).length} fields <ChevronDown/></span></summary><DataValue value={metadata}/></details>}
  </section>;
}

function LoopPanel({ node, liveSteps, query, allNodes, outputs, onSelect, runId }: { node: AgentNode; liveSteps?: LoopStep[]; query:string; allNodes:AgentNode[]; outputs:Record<string,TaskOutput>; onSelect:(id:string)=>void; runId:string }) {
  const [open, setOpen] = useState<number[]>([1,2,3]); const steps = liveSteps || []; const turns = [...new Set(steps.map((s) => s.turn))];
  const icons = { thought: MessageSquareText, tool: Wrench, result: Braces, error: AlertTriangle };
  return <div className="loop-panel">
    <div className="detail-heading"><div><h2>{node.id} / {node.label}</h2></div><StatusDot status={node.status}/></div>
    <div className="detail-stats"><span><Clock3/> {node.duration || '—'}</span><span><Zap/> {(node.tokens || 0).toLocaleString()} tok</span><span><CircleDot/> {node.confidence ? `${Math.round(node.confidence*100)}%` : '—'}</span></div>
    <InputContext node={node} query={query} allNodes={allNodes} outputs={outputs} onSelect={onSelect}/>
    <TaskResultPanel node={node} result={outputs[node.id]}/>
    <div className="turn-list">{turns.map((turn) => { const shown = open.includes(turn); const turnSteps = steps.filter((s) => s.turn === turn); return <section className="turn" key={turn}>
      <button className="turn-toggle" onClick={() => setOpen(shown ? open.filter((x) => x !== turn) : [...open,turn])}>{shown ? <ChevronDown/> : <ChevronRight/>}<strong>Loop {turn}</strong><span>{turnSteps.length} events</span></button>
      {shown && <div className="turn-events">{turnSteps.map((step,i) => { const Icon = icons[step.kind]; return <article className={`loop-event event-${step.kind}`} key={i}><div className="event-icon"><Icon/></div><div className="event-content"><strong>{step.title}</strong>{step.kind==='result'?<ToolResult step={step} runId={runId} nodeId={node.id}/>:step.kind==='tool'?<ToolInvocation step={step}/>:<p>{step.detail}</p>}{step.kind!=='tool'&&<small>{step.meta}</small>}</div></article>; })}</div>}
    </section>; })}</div>
  </div>;
}

function buildLoopSteps(events: Record<string, unknown>[]): LoopStep[] {
  let pendingCompression:Record<string,unknown>|undefined;
  let compressionCount=0;
  return events.flatMap((event,index):LoopStep[] => {
    const role=str(event.role,'assistant'), turn=Number(event.turn??index)+1;
    // 兼容已经生成的旧事件：旧版把 compression 单独写在紧随其后的
    // tool result 前；新版则直接把它挂在 tool event.compression 上。
    if(role==='compression'){pendingCompression=event;return []}
    if(event.error) return [{turn,kind:'error',title:'执行异常',detail:str(event.error)||JSON.stringify(event.error),meta:'runtime error'}];
    if(role==='tool'){
      const toolName=str(event.name,'tool result'),data=event.result;
      const embedded=event.compression&&typeof event.compression==='object'?event.compression as Record<string,unknown>:undefined;
      const compression=embedded||pendingCompression;pendingCompression=undefined;
      return [{turn,kind:'result',title:`${toolName} · 返回结果`,detail:Array.isArray(data)?`返回 ${data.length} 条结果`:'工具执行完成',meta:str(event.tool_call_id,'tool result'),data,toolName,compression,compressionIndex:compression?compressionCount++:undefined}];
    }
    const calls=Array.isArray(event.tool_calls)?event.tool_calls as Record<string,unknown>[]:[];
    const output:LoopStep[]=[];
    if(event.content||calls.length===0)output.push({turn,kind:'thought',title:calls.length?'模型决策':'模型响应',detail:str(event.content,'（空响应）'),meta:'assistant'});
    for(const call of calls){
      const fn=call.function&&typeof call.function==='object'?call.function as Record<string,unknown>:{};
      const toolName=str(fn.name,'unknown_tool');
      output.push({turn,kind:'tool',title:`调用 ${toolName}`,detail:'',meta:str(call.id),data:parseToolArguments(fn.arguments),toolName});
    }
    return output;
  });
}

export default function Home() {
  const [selected,setSelected] = useState('task_1'); const [tab,setTab] = useState<'dag'|'timeline'|'payload'>('dag'); const [paused,setPaused] = useState(false);
  const [serverRuns,setServerRuns] = useState<RunSummary[]>([]); const [detail,setDetail] = useState<RunDetail|null>(null); const [apiOnline,setApiOnline] = useState(false);
  const [showCreate,setShowCreate] = useState(false); const [newQuery,setNewQuery] = useState(''); const [startError,setStartError] = useState(''); const [starting,setStarting] = useState(false);
  const [inspectorWidth,setInspectorWidth] = useState<number>(396);
  const [inspectorCollapsed,setInspectorCollapsed] = useState<boolean>(false);
  // localStorage 只能在挂载后读取，否则 SSR HTML 与客户端首次渲染不一致（hydration mismatch）
  const prefsHydrated = useRef(false);

  useEffect(() => {
    const saved = Number(window.localStorage.getItem('dr-inspector-width'));
    // 挂载后恢复用户偏好是必须的 setState，react-compiler 的 EffectSetState 在此不适用
    // eslint-disable-next-line react-compiler/react-compiler
    if (Number.isFinite(saved) && saved >= 300 && saved <= 760) setInspectorWidth(saved);
    // eslint-disable-next-line react-compiler/react-compiler
    if (window.localStorage.getItem('dr-inspector-collapsed') === '1') setInspectorCollapsed(true);
    prefsHydrated.current = true;
  }, []);

  useEffect(() => { if (prefsHydrated.current) window.localStorage.setItem('dr-inspector-width', String(inspectorWidth)); }, [inspectorWidth]);
  useEffect(() => { if (prefsHydrated.current) window.localStorage.setItem('dr-inspector-collapsed', inspectorCollapsed ? '1' : '0'); }, [inspectorCollapsed]);

  function startInspectorResize(event: React.MouseEvent) {
    event.preventDefault();
    const startX = event.clientX;
    const startWidth = inspectorWidth;
    document.body.classList.add('resizing-inspector');
    const onMove = (move: MouseEvent) => setInspectorWidth(Math.min(760, Math.max(300, startWidth + (startX - move.clientX))));
    const onUp = () => {
      document.body.classList.remove('resizing-inspector');
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }
  const graphNodes = useMemo<AgentNode[]>(() => (Array.isArray(detail?.nodes) ? detail.nodes : []).map((n) => ({...n, dependencies:Array.isArray(n.dependencies)?n.dependencies:[], duration:'—', x:60+n.layer*350, y:48+n.row*130})), [detail]);
  const node = useMemo(() => graphNodes.find((n) => n.id === selected) || graphNodes[0], [selected,graphNodes]);
  const liveSteps = useMemo<LoopStep[]|undefined>(() => {
    if (!detail || !node) return undefined;
    return buildLoopSteps(detail.loops[node.id] || []);
  },[detail,node]);

  useEffect(() => {
    let active=true;
    async function refresh(){
      if(paused)return;
      try{
        const listResponse=await fetch('/api/runs');
        if(!listResponse.ok)throw new Error(`Runs API ${listResponse.status}`);
        const listRaw=await listResponse.json();const list=Array.isArray(listRaw)?listRaw as RunSummary[]:[];
        if(!active)return; setApiOnline(true); setServerRuns(list);
        const target=detail?.id||list[0]?.id;
        if(target){const response=await fetch(`/api/runs/${target}`);if(response.ok){const next=normalizeRunDetail(await response.json());if(active){setDetail(next);if(next.nodes.length&&!next.nodes.some(n=>n.id===selected))setSelected(next.nodes[0].id)}}}
      }catch{if(active)setApiOnline(false)}
    }
    void refresh();const timer=window.setInterval(refresh,1500);return()=>{active=false;window.clearInterval(timer)};
  },[paused,detail?.id,selected]);

  async function selectRun(id:string){try{const response=await fetch(`/api/runs/${id}`);if(!response.ok)return;const next=normalizeRunDetail(await response.json());setDetail(next);if(next.nodes[0])setSelected(next.nodes[0].id)}catch{setApiOnline(false)}}
  async function startRun(){
    const query=newQuery.trim();if(!query)return;
    setStarting(true);setStartError('');
    try{
      const response=await fetch('/api/runs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query})});
      const payload=await response.json() as {error?:unknown};
      if(!response.ok)throw new Error(str(payload.error,`启动失败 (${response.status})`));
      setShowCreate(false);setNewQuery('');setDetail(null);
    }catch(error){setStartError(error instanceof Error?error.message:'无法连接本地 Dashboard API')}finally{setStarting(false)}
  }
  const visibleRuns=serverRuns;
  const completedCount=graphNodes.filter(n=>n.status==='success').length, failedCount=graphNodes.filter(n=>n.status==='failed').length;
  const toolCount=detail?Object.values(detail.loops||{}).flat().filter(e=>e.role==='tool').length:0;
  const tokenCount=graphNodes.reduce((sum,n)=>sum+(n.tokens||0),0);
  const reached=new Set((detail?.states||[]).map(s=>str(s.to)));
  const currentState=detail?.states.length?str(detail.states[detail.states.length-1].to):'';
  const flowView=flow.map(([code,label],index)=>{const stateName=['planning','dispatching','dispatching','collecting','synthesizing','adversarial'][index];const state=currentState===stateName?'active':reached.has(stateName)?'done':'idle';return[code,label,state]});
  return <main className="app-shell" style={{gridTemplateColumns:`252px minmax(0,1fr) ${inspectorCollapsed?44:inspectorWidth}px`}}>
    <header className="topbar"><div className="brand-mark"><GitBranch/></div><div className="brand-copy"><strong>Deep Research</strong><span>Agent Observatory</span></div><div className="run-state"><span className={`live-pulse ${apiOnline?'':'offline'}`}/> {detail?.status?.toUpperCase()||(apiOnline?'IDLE':'OFFLINE')} {detail&&<><b>·</b> {Math.round(detail.elapsed_ms/1000)}s</>}</div><div className="top-actions"><Link className="top-link" href="/research">用户版 →</Link><button onClick={() => setPaused(!paused)}>{paused?<Play/>:<Pause/>}{paused?'继续刷新':'暂停刷新'}</button><button className="primary-button" onClick={()=>{setStartError('');setShowCreate(true)}}><Play/> 新建研究</button></div></header>
    <aside className="run-sidebar"><div className="sidebar-title"><span>RESEARCH RUNS</span><button onClick={()=>setDetail(null)}><RotateCcw/></button></div><label className="search-box"><Search/><input aria-label="搜索运行" placeholder="搜索历史任务…"/></label><div className="run-list">{visibleRuns.length===0&&<div className="empty-runs">暂无研究记录</div>}{visibleRuns.map((r,i)=><button onClick={()=>selectRun(r.id)} className={`run-item ${(detail?.id===r.id||(!detail&&i===0))?'active':''}`} key={r.id}><StatusDot status={r.status}/><span><strong>{r.query}</strong><small>{new Date(r.started_at*1000).toLocaleString()} · {r.event_count} events</small></span></button>)}</div><div className="system-health"><span className="eyebrow">SYSTEM HEALTH</span><div><span>Dashboard API</span><b className={apiOnline?'healthy':'warning'}>{apiOnline?'healthy':'offline'}</b></div><div><span>Event stream</span><b className={apiOnline?'healthy':'warning'}>{apiOnline?'connected':'disconnected'}</b></div><div><span>Memory DB</span><b className="healthy">healthy</b></div></div></aside>
    <section className="workspace"><div className="query-header"><div><span className="eyebrow">ACTIVE QUERY</span><h1>{detail?.query||'尚未开始研究'}</h1></div>{detail&&<div className="query-meta"><span># {detail.id}</span><span>DeepSeek</span><span>并发 3</span></div>}</div>
      <div className="flow-strip">{flowView.map(([code,label,state],i) => <div className={`flow-step ${state}`} key={code}><span>{state==='done'?<Check/>:state==='active'?<Activity/>:i+1}</span><div><strong>{code}</strong><small>{label}</small></div>{i<flow.length-1&&<ArrowRight/>}</div>)}</div>
      <div className="metrics-row"><Metric label="完成任务" value={`${completedCount} / ${graphNodes.length}`} hint={`${failedCount} 个异常`}/><Metric label="工具返回" value={String(toolCount)} hint="逐轮事件已记录"/><Metric label="Token 用量" value={tokenCount?`${(tokenCount/1000).toFixed(1)}k`:'—'} hint="按子任务累计"/><Metric label="累计延迟" value={detail?`${Math.round(detail.elapsed_ms/1000)}s`:'—'} hint={detail?'实时运行时钟':'暂无运行'}/><Metric label="事件总数" value={String(detail?.event_count||'—')} hint="append-only JSONL"/></div>
      <div className="content-tabs">{([['dag',GitBranch,'DAG 任务图'],['timeline',Activity,'执行时间线'],['payload',TerminalSquare,'原始事件']] as const).map(([id,Icon,label]) => <button className={tab===id?'active':''} onClick={() => setTab(id)} key={id}><Icon/>{label}</button>)}</div>
      <div className="main-canvas">{!detail&&<div className="empty-workspace"><GitBranch/><h2>暂无研究运行</h2><p>点击右上角“新建研究”，Planner 生成 DAG 后会在这里实时展示。</p><button onClick={()=>setShowCreate(true)}><Play/>新建研究</button></div>}{detail&&tab==='dag'&&<DagGraph graphNodes={graphNodes} selected={selected} onSelect={setSelected}/>} {detail&&tab==='timeline'&&<div className="timeline-view">{graphNodes.map((n,i)=><button key={n.id} onClick={()=>setSelected(n.id)}><span>{n.id}</span><strong>{n.label}</strong><i style={{width:`${25+i*8}%`}} className={`bar-${n.status}`}/><small>{n.duration||'等待'}</small></button>)}</div>} {detail&&tab==='payload'&&<pre className="payload-view">{JSON.stringify(detail.events,null,2)}</pre>}</div>
    </section>
    <aside className={`inspector ${inspectorCollapsed?'collapsed':''}`}>
      {!inspectorCollapsed&&(
        // eslint-disable-next-line jsx-a11y/no-noninteractive-element-interactions, jsx-a11y/no-static-element-interactions
        <div className="inspector-resize-handle" onMouseDown={startInspectorResize}/>
      )}
      {inspectorCollapsed
        ? <button className="inspector-expand" onClick={()=>setInspectorCollapsed(false)} aria-label="展开检查器"><ChevronLeft/><span>INSPECTOR</span></button>
        : <div className="inspector-body">
            <div className="inspector-chrome"><span className="eyebrow">AGENT INSPECTOR</span><button className="inspector-collapse" onClick={()=>setInspectorCollapsed(true)} aria-label="收起检查器"><ChevronRight/></button></div>
            {node?<LoopPanel node={node} liveSteps={liveSteps} query={detail?.query||''} allNodes={graphNodes} outputs={detail?.outputs||{}} onSelect={setSelected} runId={detail?.id||''}/>:<div className="empty-inspector"><CircleDot/><span>选择一个 Agent 节点<br/>查看执行 Loop</span></div>}
          </div>}
    </aside>
    {showCreate&&(
      // eslint-disable-next-line jsx-a11y/no-noninteractive-element-interactions
      <div className="modal-backdrop" role="presentation" onMouseDown={event=>{if(event.target===event.currentTarget&&!starting)setShowCreate(false)}}>
        <form className="create-dialog" onSubmit={event=>{event.preventDefault();void startRun()}}>
          <span className="eyebrow">NEW RESEARCH RUN</span>
          <h2>新建研究</h2>
          <p>任务启动后，Planner、DAG 和各 Agent Loop 会自动出现在调试台。</p>
          {/* eslint-disable-next-line jsx-a11y/no-autofocus */}
          <textarea autoFocus value={newQuery} onChange={event=>setNewQuery(event.target.value)} placeholder="例如：分析 2026 年 AI 编程助手的市场格局与核心技术路线" maxLength={1000}/>
          {startError&&<div className="create-error"><AlertTriangle/>{startError}</div>}
          <div className="dialog-actions"><button type="button" onClick={()=>setShowCreate(false)} disabled={starting}>取消</button><button className="primary-button" type="submit" disabled={starting||!newQuery.trim()}>{starting?<Activity/>:<Play/>}{starting?'正在启动…':'开始研究'}</button></div>
        </form>
      </div>
    )}
    <footer className="event-footer"><span><Globe2/> web_search <b>8</b></span><span><Database/> memory_hit <b>1</b></span><span><Sparkles/> llm_call <b>14</b></span><span className="footer-alert"><AlertTriangle/> warnings <b>2</b></span><span className="event-stream"><i/> event stream connected</span></footer>
  </main>;
}

'use client';

import {
  AlertTriangle, ArrowLeft, ArrowUp, ArrowUpRight, Check, Download,
  GitBranch, History, Loader2, Shield, Sparkles,
} from 'lucide-react';
import Link from 'next/link';
import { Fragment, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';

const API = '';

type RunSummaryItem = { id: string; query: string; status: string; started_at: number; elapsed_ms: number; adversarial?: boolean };
type RunProgress = { id: string; query: string; status: string; started_at: number; elapsed_ms: number; states: Record<string, unknown>[]; nodes: { id: string; status: string }[]; events: Record<string, unknown>[] };
type View = 'home' | 'progress' | 'report';

const STATUS_LABELS: Record<string, string> = { success: '已完成', running: '进行中', failed: '失败' };

const BASE_STAGES = [
  { key: 'planning', label: '规划任务' },
  { key: 'dispatching', label: '检索与研究' },
  { key: 'collecting', label: '汇总结果' },
  { key: 'synthesizing', label: '合成报告' },
];

const EXAMPLES = [
  '对比分析 GPT-4o 与 Claude 3.5 的多语言能力',
  '2026 年具身智能的技术路线与产业化进展',
  'CRISPR 基因编辑在罕见病治疗的最新临床进展',
];

function str(value: unknown, fallback = ''): string {
  if (typeof value === 'string') return value || fallback;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  return fallback;
}

function normalizeDetail(raw: unknown): RunProgress {
  const value = raw && typeof raw === 'object' ? raw as Record<string, unknown> : {};
  return {
    id: str(value.id), query: str(value.query), status: str(value.status, 'running'),
    started_at: Number(value.started_at || 0),
    elapsed_ms: Number(value.elapsed_ms || 0),
    states: Array.isArray(value.states) ? value.states as Record<string, unknown>[] : [],
    nodes: Array.isArray(value.nodes) ? value.nodes as RunProgress['nodes'] : [],
    events: Array.isArray(value.events) ? value.events as Record<string, unknown>[] : [],
  };
}

/* ---------- 轻量 Markdown 渲染（组件级，无 dangerouslySetInnerHTML） ---------- */

function renderInline(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)\s]+\))/g).map((part, index) => {
    if (!part) return null;
    const bold = part.match(/^\*\*([^*]+)\*\*$/);
    if (bold) return <strong key={index}>{bold[1]}</strong>;
    const code = part.match(/^`([^`]+)`$/);
    if (code) return <code key={index}>{code[1]}</code>;
    const link = part.match(/^\[([^\]]+)\]\(([^)\s]+)\)$/);
    if (link) return <a key={index} href={link[2]} target="_blank" rel="noreferrer">{link[1]}</a>;
    return <Fragment key={index}>{part}</Fragment>;
  });
}

function splitRow(line: string): string[] {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((cell) => cell.trim());
}

function MarkdownView({ markdown }: { markdown: string }) {
  const lines = markdown.split(/\r?\n/);
  const blocks: ReactNode[] = [];
  let i = 0;
  let key = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.trim().startsWith('```')) {
      const buf: string[] = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith('```')) { buf.push(lines[i]); i += 1; }
      i += 1;
      blocks.push(<pre key={key++}><code>{buf.join('\n')}</code></pre>);
      continue;
    }
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      const Tag = `h${heading[1].length}` as 'h1' | 'h2' | 'h3' | 'h4';
      blocks.push(<Tag key={key++}>{renderInline(heading[2])}</Tag>);
      i += 1;
      continue;
    }
    if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) { blocks.push(<hr key={key++} />); i += 1; continue; }
    if (line.trim().startsWith('|') && i + 1 < lines.length && lines[i + 1].includes('-') && /^\s*\|?[\s:|-]+\|?\s*$/.test(lines[i + 1])) {
      const header = splitRow(line);
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && lines[i].trim().startsWith('|')) { rows.push(splitRow(lines[i])); i += 1; }
      blocks.push(<div key={key++} className="studio-table-wrap"><table><thead><tr>{header.map((cell, ci) => <th key={ci}>{renderInline(cell)}</th>)}</tr></thead><tbody>{rows.map((row, ri) => <tr key={ri}>{row.map((cell, ci) => <td key={ci}>{renderInline(cell)}</td>)}</tr>)}</tbody></table></div>);
      continue;
    }
    if (/^\s*[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) { items.push(lines[i].replace(/^\s*[-*]\s+/, '')); i += 1; }
      blocks.push(<ul key={key++}>{items.map((item, ii) => <li key={ii}>{renderInline(item)}</li>)}</ul>);
      continue;
    }
    if (/^\s*\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) { items.push(lines[i].replace(/^\s*\d+\.\s+/, '')); i += 1; }
      blocks.push(<ol key={key++}>{items.map((item, ii) => <li key={ii}>{renderInline(item)}</li>)}</ol>);
      continue;
    }
    if (!line.trim()) { i += 1; continue; }
    const buf: string[] = [];
    while (i < lines.length && lines[i].trim()
      && !/^(#{1,4}\s|```|\s*[-*]\s|\s*\d+\.\s|\s*\|)/.test(lines[i])
      && !/^\s*(-{3,}|\*{3,})\s*$/.test(lines[i])) { buf.push(lines[i]); i += 1; }
    blocks.push(<p key={key++}>{renderInline(buf.join(' '))}</p>);
  }
  return <div className="studio-md">{blocks}</div>;
}

/* ---------- 对抗前后行级 Diff ---------- */

type DiffLine = { kind: 'same' | 'add' | 'del'; text: string };

function diffLines(before: string, after: string): DiffLine[] {
  const a = before.split(/\r?\n/);
  const b = after.split(/\r?\n/);
  const n = a.length;
  const m = b.length;
  const dp: number[][] = Array.from({ length: n + 1 }, () => Array.from({ length: m + 1 }, () => 0));
  for (let i = n - 1; i >= 0; i -= 1) {
    for (let j = m - 1; j >= 0; j -= 1) {
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const out: DiffLine[] = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { out.push({ kind: 'same', text: a[i] }); i += 1; j += 1; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) { out.push({ kind: 'del', text: a[i] }); i += 1; }
    else { out.push({ kind: 'add', text: b[j] }); j += 1; }
  }
  while (i < n) { out.push({ kind: 'del', text: a[i] }); i += 1; }
  while (j < m) { out.push({ kind: 'add', text: b[j] }); j += 1; }
  return out;
}

function DiffView({ before, after }: { before: string; after: string }) {
  const lines = useMemo(() => diffLines(before, after), [before, after]);
  const added = lines.filter((l) => l.kind === 'add').length;
  const removed = lines.filter((l) => l.kind === 'del').length;

  const renderLine = (line: DiffLine, key: number) => <div className={`diff-line diff-${line.kind}`} key={key}>
    <span className="diff-sign">{line.kind === 'add' ? '+' : line.kind === 'del' ? '−' : ''}</span>
    <span className="diff-text">{line.text || ' '}</span>
  </div>;

  const blocks: ReactNode[] = [];
  let key = 0;
  let i = 0;
  while (i < lines.length) {
    if (lines[i].kind !== 'same') { blocks.push(renderLine(lines[i], key++)); i += 1; continue; }
    let j = i;
    while (j < lines.length && lines[j].kind === 'same') j += 1;
    const run = lines.slice(i, j);
    if (run.length > 7) {
      run.slice(0, 3).forEach((line) => blocks.push(renderLine(line, key++)));
      blocks.push(<details className="diff-collapse" key={key++}>
        <summary>展开中间 {run.length - 6} 行未变化内容</summary>
        {run.slice(3, -3).map((line, ri) => renderLine(line, key++ + ri))}
      </details>);
      key += run.length;
      run.slice(-3).forEach((line) => blocks.push(renderLine(line, key++)));
    } else {
      run.forEach((line) => blocks.push(renderLine(line, key++)));
    }
    i = j;
  }

  return <div className="studio-diff">
    <div className="studio-diff-stats">
      <span className="diff-stat-add">+{added} 行新增</span>
      <span className="diff-stat-del">−{removed} 行删除</span>
      <span className="diff-stat-note">红底为对抗前被移除/改写的内容，绿底为对抗后新增的内容</span>
    </div>
    <div className="studio-diff-body">{blocks}</div>
  </div>;
}

/* ---------- 页面 ---------- */

export default function ResearchPage() {
  const [view, setView] = useState<View>('home');
  const [query, setQuery] = useState('');
  const [adversarial, setAdversarial] = useState<boolean>(false);
  // localStorage 只能在挂载后读取，否则 SSR HTML 与客户端首次渲染不一致（hydration mismatch）
  const prefsHydrated = useRef(false);
  const [activeAdversarial, setActiveAdversarial] = useState(false);
  const [runId, setRunId] = useState('');
  const [progress, setProgress] = useState<RunProgress | null>(null);
  const [report, setReport] = useState('');
  const [runError, setRunError] = useState('');
  const [starting, setStarting] = useState(false);
  const [runs, setRuns] = useState<RunSummaryItem[]>([]);
  const [apiOnline, setApiOnline] = useState(true);
  const [fallbackStartedAt, setFallbackStartedAt] = useState(0);
  const [upgrading, setUpgrading] = useState(false);
  const [reportBefore, setReportBefore] = useState<string | null>(null);
  const [confidenceBefore, setConfidenceBefore] = useState<number | null>(null);
  const [adversarialScores, setAdversarialScores] = useState<number[]>([]);
  const [reportVersion, setReportVersion] = useState<'after' | 'before' | 'diff'>('after');
  const [nowMs, setNowMs] = useState(0);

  useEffect(() => {
    // 挂载后恢复用户偏好是必须的 setState，react-compiler 的 EffectSetState 在此不适用
    // eslint-disable-next-line react-compiler/react-compiler
    if (window.localStorage.getItem('dr-research-adversarial') === '1') setAdversarial(true);
    prefsHydrated.current = true;
  }, []);
  useEffect(() => { if (prefsHydrated.current) window.localStorage.setItem('dr-research-adversarial', adversarial ? '1' : '0'); }, [adversarial]);

  // 运行中的耗时在本地每秒跳动：后端 elapsed_ms 只随新事件更新，
  // 长时间 LLM 调用期间不动，会让计时看起来卡死。
  useEffect(() => {
    if (view !== 'progress') return;
    const timer = window.setInterval(() => setNowMs(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [view]);

  // 历史报告列表：挂载时及回到首页时刷新
  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const response = await fetch(`${API}/api/runs`);
        if (!response.ok) throw new Error(`Runs API ${response.status}`);
        const list = await response.json() as unknown;
        if (active) { setRuns(Array.isArray(list) ? list as RunSummaryItem[] : []); setApiOnline(true); }
      } catch { if (active) setApiOnline(false); }
    })();
    return () => { active = false; };
  }, [view]);

  // 进行中任务：每 2 秒轮询，success 后拉取报告
  useEffect(() => {
    if (view !== 'progress' || !runId) return;
    let active = true;
    async function tick() {
      try {
        const response = await fetch(`${API}/api/runs/${runId}`);
        if (!response.ok) throw new Error(`Runs API ${response.status}`);
        const detail = normalizeDetail(await response.json());
        if (!active) return;
        setApiOnline(true);
        setProgress(detail);
        if (detail.status === 'success') {
          const reportResponse = await fetch(`${API}/api/runs/${runId}/report`);
          if (!active) return;
          if (reportResponse.ok) {
            const payload = await reportResponse.json() as { markdown?: unknown; before_markdown?: unknown; before_confidence?: unknown; adversarial_scores?: unknown };
            setReport(str(payload.markdown));
            setReportBefore(typeof payload.before_markdown === 'string' && payload.before_markdown ? payload.before_markdown : null);
            setConfidenceBefore(typeof payload.before_confidence === 'number' ? payload.before_confidence : null);
            setAdversarialScores(Array.isArray(payload.adversarial_scores) ? payload.adversarial_scores.filter((s): s is number => typeof s === 'number') : []);
            setReportVersion('after');
            setView('report');
          } else {
            setRunError('运行已完成，但报告文件尚未生成。');
          }
        } else if (detail.status === 'failed') {
          setRunError('研究流程执行失败，可到调试台查看各 Agent 的执行详情。');
        }
      } catch { if (active) setApiOnline(false); }
    }
    void tick();
    const timer = window.setInterval(() => { void tick(); }, 2000);
    return () => { active = false; window.clearInterval(timer); };
  }, [view, runId]);

  async function startResearch() {
    const text = query.trim();
    if (!text) return;
    setStarting(true);
    setRunError('');
    try {
      const response = await fetch(`${API}/api/runs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text, adversarial }),
      });
      const payload = await response.json() as { run_id?: unknown; error?: unknown };
      if (!response.ok) throw new Error(str(payload.error, `启动失败 (${response.status})`));
      setRunId(str(payload.run_id));
      setActiveAdversarial(adversarial);
      setFallbackStartedAt(Date.now() / 1000);
      setProgress(null);
      setReport('');
      setReportBefore(null);
      setConfidenceBefore(null);
      setAdversarialScores([]);
      setApiOnline(true);
      setView('progress');
    } catch (error) {
      setRunError(error instanceof Error ? error.message : '无法连接本地 Dashboard API');
      setApiOnline(false);
    } finally { setStarting(false); }
  }

  function openRun(run: RunSummaryItem) {
    setRunId(run.id);
    setActiveAdversarial(Boolean(run.adversarial));
    setFallbackStartedAt(run.started_at);
    setProgress(null);
    setReport('');
    setReportBefore(null);
    setConfidenceBefore(null);
    setAdversarialScores([]);
    setRunError('');
    setView('progress');
  }

  function backToHome() {
    setView('home');
    setRunId('');
    setRunError('');
  }

  async function startUpgrade() {
    if (!runId || upgrading) return;
    setUpgrading(true);
    setRunError('');
    try {
      const response = await fetch(`${API}/api/runs/${runId}/adversarial`, { method: 'POST' });
      if (response.status === 202) {
        setActiveAdversarial(true);
        setFallbackStartedAt(Date.now() / 1000);
        setProgress(null);
        setReport('');
        setReportBefore(null);
        setConfidenceBefore(null);
        setAdversarialScores([]);
        setApiOnline(true);
        setView('progress');
      } else if (response.status === 409) {
        setRunError('该研究正在升级中，请稍候。');
      } else {
        const payload = await response.json() as { error?: unknown };
        setRunError(str(payload.error, `升级失败 (${response.status})`));
      }
    } catch {
      setRunError('无法连接本地 Dashboard API');
      setApiOnline(false);
    } finally { setUpgrading(false); }
  }

  function downloadReport() {
    const viewingBefore = reportVersion === 'before' && reportBefore;
    const safeName = (progress?.query || 'research').slice(0, 30).replace(/[\\/:*?"<>|]/g, '_');
    const blob = new Blob([viewingBefore ? reportBefore : report], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `report_${safeName}${viewingBefore ? '_对抗前' : ''}.md`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  const rawStage = progress?.states.length ? str(progress.states[progress.states.length - 1].to) : '';
  const stageKey = rawStage === 'replanning' ? 'dispatching' : rawStage;
  const steps = activeAdversarial ? [...BASE_STAGES, { key: 'adversarial', label: '对抗优化' }] : BASE_STAGES;
  const stageIndex = Math.max(0, steps.findIndex((step) => step.key === stageKey));
  const totalNodes = progress?.nodes.length || 0;
  const doneNodes = progress ? progress.nodes.filter((n) => n.status === 'success').length : 0;
  const failed = Boolean(runError) && progress?.status === 'failed';
  const terminal = progress ? progress.status === 'success' || progress.status === 'failed' : false;
  const startedAtSec = fallbackStartedAt || progress?.started_at || 0;
  const elapsedMs = terminal
    ? progress?.elapsed_ms || 0
    : startedAtSec
      ? Math.max(progress?.elapsed_ms || 0, nowMs - startedAtSec * 1000)
      : progress?.elapsed_ms || 0;

  const completedEvent = progress ? [...progress.events].reverse().find((event) => str(event.type) === 'run_completed') : undefined;
  const completedPayload = completedEvent?.payload && typeof completedEvent.payload === 'object' ? completedEvent.payload as Record<string, unknown> : undefined;
  const confidence = typeof completedPayload?.confidence === 'number' ? completedPayload.confidence : undefined;
  const sources = typeof completedPayload?.sources === 'number' ? completedPayload.sources : undefined;

  return <main className="studio-page">
    <header className="studio-nav">
      <div className="studio-nav-inner">
        <span className="studio-logo"><GitBranch /></span>
        <span className="studio-wordmark">Deep Research</span>
        <Link className="studio-nav-link" href="/">开发者调试台<ArrowUpRight /></Link>
      </div>
    </header>

    {!apiOnline && <div className="studio-offline studio-enter"><AlertTriangle />无法连接本地研究服务（127.0.0.1:8765），请确认 Dashboard API 已启动。</div>}

    {view === 'home' && <div className="studio-view" key="home">
      <section className="studio-hero">
        <span className="studio-badge studio-enter"><Sparkles />MULTI-AGENT RESEARCH SYSTEM</span>
        <h1 className="studio-enter studio-d1">问一个问题，<br />得到一份有据可查的研究报告</h1>
        <p className="studio-sub studio-enter studio-d2">Planner 自动拆解任务，多个 Agent 并行检索与阅读，最终合成一份结构清晰、附引用来源的中文研究报告。</p>

        <div className="studio-composer studio-enter studio-d3">
          <textarea aria-label="研究问题" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="输入你的研究问题…" maxLength={1000} rows={4} />
          <div className="studio-composer-bar">
            <div className="studio-adv">
              <button type="button" role="switch" aria-checked={adversarial} aria-label="Red/Blue 对抗优化" className={`studio-toggle ${adversarial ? 'on' : ''}`} onClick={() => setAdversarial(!adversarial)}><span className="studio-toggle-knob" /></button>
              <div className="studio-adv-copy"><strong>Red/Blue 对抗优化</strong><small>红队多轮攻击修复，更慢但更严谨</small></div>
            </div>
            <button className="studio-send" aria-label="开始研究" onClick={() => { void startResearch(); }} disabled={starting || !query.trim()}>{starting ? <Loader2 className="studio-spin" /> : <ArrowUp />}</button>
          </div>
        </div>
        {runError && <div className="studio-error studio-enter"><AlertTriangle />{runError}</div>}

        <div className="studio-examples studio-enter studio-d4">
          {EXAMPLES.map((example) => <button key={example} onClick={() => setQuery(example)}>{example}</button>)}
        </div>
      </section>

      <section className="studio-history">
        <div className="studio-history-head studio-enter"><h2><History />历史报告</h2></div>
        {runs.length === 0
          ? <p className="studio-empty studio-enter">暂无研究记录，从上方提出第一个问题开始。</p>
          : <div className="studio-history-grid">{runs.map((run, index) => <button key={run.id} className="studio-card studio-enter" style={{ animationDelay: `${Math.min(index, 8) * 50}ms` }} onClick={() => openRun(run)}>
              <span className="studio-card-top"><span className={`studio-dot st-${run.status}`} /><span className="studio-card-status">{STATUS_LABELS[run.status] || run.status}</span>{run.adversarial && <em className="studio-card-adv"><Shield />对抗</em>}</span>
              <span className="studio-card-query">{run.query}</span>
              <span className="studio-card-meta">{new Date(run.started_at * 1000).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })} · 耗时 {Math.round(run.elapsed_ms / 1000)}s</span>
            </button>)}
          </div>}
      </section>
    </div>}

    {view === 'progress' && <div className="studio-view studio-progress-wrap" key="progress">
      <blockquote className="studio-quote studio-enter">{progress?.query || query}</blockquote>
      {failed
        ? <div className="studio-failed studio-enter studio-d1">
            <AlertTriangle />
            <h2>研究失败</h2>
            <p>{runError}</p>
            <div className="studio-failed-actions">
              <button onClick={backToHome}><ArrowLeft />返回首页</button>
              <Link href="/">打开调试台<ArrowUpRight /></Link>
            </div>
          </div>
        : <>
            <div className="studio-steps">
              {steps.map((step, index) => {
                const state = index < stageIndex ? 'done' : index === stageIndex ? 'active' : 'todo';
                return <div className={`studio-step ${state} studio-enter`} style={{ animationDelay: `${index * 60}ms` }} key={step.key}>
                  <span className="studio-step-dot">{state === 'done' ? <Check /> : state === 'active' ? <i /> : null}</span>
                  <span className="studio-step-label">{step.label}</span>
                </div>;
              })}
            </div>
            <div className="studio-progress-meta studio-enter studio-d3">
              <span>已用时间 <b>{Math.round(elapsedMs / 1000)}s</b></span>
              <span>完成节点 <b>{doneNodes} / {totalNodes || '—'}</b></span>
              {activeAdversarial && <span className="studio-adv-mark"><Shield />对抗优化已开启</span>}
            </div>
            {runError && <p className="studio-warn">{runError}</p>}
            <button className="studio-back studio-enter studio-d4" onClick={backToHome}><ArrowLeft />取消等待，返回首页</button>
          </>}
    </div>}

    {view === 'report' && <div className="studio-view" key="report">
      <div className="studio-report-bar">
        <div className="studio-report-bar-inner">
          <button className="studio-back" onClick={backToHome}><ArrowLeft />新的研究</button>
          <div className="studio-report-meta">
            {reportBefore !== null && <span className="studio-version-toggle" role="tablist" aria-label="报告版本">
              <button role="tab" aria-selected={reportVersion === 'after'} className={reportVersion === 'after' ? 'active' : ''} onClick={() => setReportVersion('after')}>对抗后<em>当前</em></button>
              <button role="tab" aria-selected={reportVersion === 'before'} className={reportVersion === 'before' ? 'active' : ''} onClick={() => setReportVersion('before')}>对抗前<em>原始</em></button>
              <button role="tab" aria-selected={reportVersion === 'diff'} className={reportVersion === 'diff' ? 'active' : ''} onClick={() => setReportVersion('diff')}>对比<em>Diff</em></button>
            </span>}
            {confidence !== undefined && <span>置信度 <b>{confidenceBefore !== null && confidenceBefore !== confidence ? `${confidenceBefore.toFixed(2)} → ${confidence.toFixed(2)}` : confidence.toFixed(2)}</b></span>}
            {adversarialScores.length > 0 && <span>对抗评分 <b>{adversarialScores[0].toFixed(1)} → {adversarialScores[adversarialScores.length - 1].toFixed(1)}</b><em className="studio-conf-note">第1轮 → 第{adversarialScores.length}轮</em></span>}
            {sources !== undefined && <span>引用来源 <b>{sources}</b></span>}
            {progress !== null && <span>耗时 <b>{Math.round(progress.elapsed_ms / 1000)}s</b></span>}
            {activeAdversarial && <span className="studio-adv-mark"><Shield />对抗优化</span>}
          </div>
          <button className="studio-upgrade" onClick={() => { void startUpgrade(); }} disabled={upgrading}>
            {upgrading ? <Loader2 className="studio-spin" /> : <Shield />}对抗升级
          </button>
          <button className="studio-download" onClick={downloadReport}><Download />下载 Markdown</button>
        </div>
      </div>
      {runError && <div className="studio-offline studio-enter"><AlertTriangle />{runError}</div>}
      {reportVersion === 'before' && reportBefore !== null && <div className="studio-before-banner studio-enter"><History />这是对抗升级前的原始版本，仅用于对比查看。</div>}
      <article className="studio-article studio-enter studio-d1">
        {reportVersion === 'diff' && reportBefore !== null
          ? <DiffView before={reportBefore} after={report} />
          : (reportVersion === 'before' && reportBefore !== null ? reportBefore : report)
            ? <MarkdownView markdown={reportVersion === 'before' && reportBefore !== null ? reportBefore : report} />
            : <p className="studio-empty">报告内容为空。</p>}
      </article>
    </div>}
  </main>;
}

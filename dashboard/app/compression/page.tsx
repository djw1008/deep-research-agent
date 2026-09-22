'use client';

import { AlertTriangle, ArrowLeft, ArrowRight, Database, Loader2, SearchX } from 'lucide-react';
import Link from 'next/link';
import { useEffect, useState } from 'react';

type CompressionRecord = Record<string, unknown>;
type LocatedCompression = { record: CompressionRecord; toolCallId: string; index: number };
type PageParams = { run: string; node: string; call: string; ci: number };
type PageState = 'loading' | 'offline' | 'missing' | 'ready';

function str(value: unknown, fallback = ''): string {
  if (typeof value === 'string') return value || fallback;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  return fallback;
}

function collectCompressions(events: Record<string, unknown>[]): LocatedCompression[] {
  const found: LocatedCompression[] = [];
  let pending: CompressionRecord | undefined;
  for (const event of events) {
    const role = str(event.role, 'assistant');
    // 与调试台首页一致：旧版把 compression 单独写在紧随其后的
    // tool result 前；新版则直接把它挂在 tool event.compression 上。
    if (role === 'compression') { pending = event; continue; }
    if (role !== 'tool') continue;
    const embedded = event.compression && typeof event.compression === 'object' ? event.compression as CompressionRecord : undefined;
    const record = embedded || pending;
    pending = undefined;
    if (record) found.push({ record, toolCallId: str(event.tool_call_id, 'tool result'), index: found.length });
  }
  return found;
}

function readParams(): PageParams {
  if (typeof window === 'undefined') return { run: '', node: '', call: '', ci: 0 };
  const search = new URLSearchParams(window.location.search);
  const ciRaw = Number(search.get('ci') ?? 0);
  return {
    run: search.get('run') || '',
    node: search.get('node') || '',
    call: search.get('call') || '',
    ci: Number.isFinite(ciRaw) && ciRaw >= 0 ? Math.floor(ciRaw) : 0,
  };
}

function formatContent(value: unknown): string {
  if (value === undefined || value === null) return '';
  return typeof value === 'string' ? value : JSON.stringify(value, null, 2);
}

function MetricCard({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return <div className="cmp-metric"><span>{label}</span><strong>{value}</strong>{hint && <small>{hint}</small>}</div>;
}

function ContentPanel({ title, accent, value }: { title: string; accent: 'before' | 'after'; value: unknown }) {
  const text = formatContent(value);
  return <section className={`cmp-panel cmp-panel-${accent}`}>
    <header className="cmp-panel-head"><span>{title}</span><b className="cmp-badge">{text ? `${text.length.toLocaleString()} chars` : '空'}</b></header>
    {text ? <pre className="cmp-content">{text}</pre> : <p className="cmp-content-empty">该字段为空或未记录。</p>}
  </section>;
}

export default function CompressionPage() {
  const [params] = useState<PageParams>(readParams);
  const [state, setState] = useState<PageState>('loading');
  const [located, setLocated] = useState<LocatedCompression | null>(null);
  const [runQuery, setRunQuery] = useState('');

  useEffect(() => {
    let active = true;
    void (async () => {
      if (!params.run || !params.node) { if (active) setState('missing'); return; }
      try {
        const response = await fetch(`/api/runs/${encodeURIComponent(params.run)}`);
        if (!response.ok) throw new Error(`Runs API ${response.status}`);
        const detail = await response.json() as Record<string, unknown>;
        if (!active) return;
        setRunQuery(str(detail.query));
        const loops = detail.loops && typeof detail.loops === 'object' ? detail.loops as Record<string, Record<string, unknown>[]> : {};
        const events = Array.isArray(loops[params.node]) ? loops[params.node] : [];
        const all = collectCompressions(events);
        const byCall = params.call ? all.find((item) => item.toolCallId === params.call) : undefined;
        const found = byCall || all[params.ci];
        if (!found) { setState('missing'); return; }
        setLocated(found);
        setState('ready');
      } catch {
        if (active) setState('offline');
      }
    })();
    return () => { active = false; };
  }, [params]);

  const record = located?.record;
  const before = Number(record?.before_chars || 0);
  const after = Number(record?.after_chars || 0);
  const tokensBefore = Number(record?.estimated_tokens_before || 0);
  const tokensAfter = Number(record?.estimated_tokens_after || 0);
  const ratio = before ? Math.round(after / before * 100) : 0;
  const saved = Math.max(0, before - after);

  return <main className="compression-page">
    <div className="cmp-container">
      <Link className="cmp-back" href="/"><ArrowLeft />返回调试台</Link>

      {state === 'loading' && <div className="cmp-state"><Loader2 className="cmp-spin" /><h2>正在加载压缩记录…</h2><p>正在从 Dashboard API 读取运行事件流。</p></div>}

      {state === 'offline' && <div className="cmp-state cmp-state-error"><AlertTriangle /><h2>无法连接 Dashboard API</h2><p>请确认调试服务已在 http://127.0.0.1:8765 启动，然后刷新本页。</p></div>}

      {state === 'missing' && <div className="cmp-state cmp-state-error"><SearchX /><h2>未找到压缩记录</h2><p>节点 <code>{params.node || '—'}</code> 中没有匹配 call <code>{params.call || '—'}</code>（兜底序号 {params.ci}）的上下文压缩事件。</p></div>}

      {state === 'ready' && record && <>
        <header className="cmp-hero">
          <div className="cmp-hero-icon"><Database /></div>
          <div className="cmp-hero-text">
            <span className="eyebrow">CONTEXT COMPRESSION</span>
            <h1>{str(record.strategy, 'context compression')}</h1>
            <p>{runQuery || '未记录研究问题'}</p>
          </div>
          <div className="cmp-hero-meta">
            <span>run <code>{params.run}</code></span>
            <span>node <code>{params.node}</code></span>
            <span>tool <code>{str(record.tool, 'context')}</code></span>
            <span>call <code>{located?.toolCallId}</code></span>
          </div>
        </header>

        <div className="cmp-metrics">
          <MetricCard label="压缩前" value={`${before.toLocaleString()} chars`} hint={`≈ ${tokensBefore.toLocaleString()} tokens`} />
          <MetricCard label="压缩后（模型实际收到）" value={`${after.toLocaleString()} chars`} hint={`≈ ${tokensAfter.toLocaleString()} tokens`} />
          <MetricCard label="保留比例" value={`${ratio}%`} hint="压缩后 / 压缩前" />
          <MetricCard label="节省字符" value={saved.toLocaleString()} hint={`≈ ${Math.max(0, tokensBefore - tokensAfter).toLocaleString()} tokens`} />
        </div>

        <div className="cmp-bar-row">
          <span>0</span>
          <div className="cmp-bar"><i style={{ width: `${Math.min(100, ratio)}%` }} /></div>
          <span>{ratio}% 保留</span>
        </div>

        <div className="cmp-budget">
          <span>调用前上下文 <b>{Number(record.context_tokens_before || 0).toLocaleString()} tok</b></span>
          <span>模型预算 <b>{Number(record.context_budget || 0).toLocaleString()} tok</b></span>
          {record.items_before !== undefined && <span>条目 <b>{str(record.items_before)} <ArrowRight /> {str(record.items_after)}</b></span>}
        </div>

        <div className="cmp-panels">
          <ContentPanel title="压缩前上下文 · original_content" accent="before" value={record.original_content} />
          <ContentPanel title="压缩后实际上下文 · retained_content" accent="after" value={record.retained_content} />
        </div>
      </>}
    </div>
  </main>;
}

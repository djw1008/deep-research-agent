# 引用来源传递方案

## 目标

在保留现有 `[SRC-n]` 局部引用和 `[n]` 全局引用机制的基础上，统一正常研究与记忆召回的来源传递方式，并保证只持久化 ResearchAgent 最终输出实际引用过的来源。

## 数据约定

每个成功的 ResearchAgent 都通过 `AgentResult.metadata["sources"]` 暴露来源：

```python
{
    "sources": [
        {
            "source_label": "SRC-1",
            "url": "https://example.com/article",
            "title": "Article title",
            "snippet": "Short evidence summary",
        }
    ]
}
```

`sources` 只包含 `AgentResult.output` 中实际出现的 `[SRC-n]` 所对应的 URL。URL 使用原始字符串进行精确去重，不做规范化。

## 正常研究流程

1. 工具返回标题、URL 和摘要或正文。
2. ResearchAgent 为工具结果中的 URL 分配当前子任务内稳定的 `SRC-n` 标签。
3. 带有 `source_label` 的工具结果作为 tool message 返回给 LLM。
4. LLM 在最终调研结果中使用 `[SRC-n]`。
5. ResearchAgent 从最终 output 提取实际使用的标签，并从内部 trajectory 中筛选相应来源。
6. 筛选结果写入 `AgentResult.metadata["sources"]`。

trajectory 仍用于调试和过程记录，但不再作为 Summarizer 的来源接口。

## 持久化与召回

Orchestrator 持久化成功子任务时直接保存：

- `AgentResult.output` → `KnowledgeEntry.content`
- `AgentResult.metadata["sources"]` → `KnowledgeEntry.sources`

召回时执行反向恢复：

- `KnowledgeEntry.content` → `AgentResult.output`
- `KnowledgeEntry.sources` → `AgentResult.metadata["sources"]`

不恢复完整工具轨迹，也不把历史搜索过程或网页正文重新放入上下文。

旧知识库记录如果没有 `source_label`，不会按列表顺序猜测标签；对应引用无法建立可靠绑定时将被现有清理逻辑移除。

## Summarizer 全局引用收集

Summarizer 的 collection 阶段统一读取每个成功结果的：

```text
result.output + result.metadata["sources"]
```

处理步骤：

1. 从 output 提取 `[SRC-n]`。
2. 在该结果的 `metadata["sources"]` 中查找同名标签。
3. 以原始 URL 为 key 构建全局来源字典。
4. 相同 URL 合并来自不同任务的 bindings。
5. 为去重后的 URL 分配连续全局编号 `[1]`、`[2]`。
6. 使用 `(task_id, source_label)` 将各子任务 output 中的局部标签替换为全局编号。

示例：

```text
research_1 / SRC-1 → https://example.com/a
research_2 / SRC-3 → https://example.com/a

全局结果：
research_1 / SRC-1 → [1]
research_2 / SRC-3 → [1]
```

替换后的研究材料和全局来源注册表再交给总结 LLM。最终仍由现有后处理删除无效引用、移除未使用来源并压缩编号。

## 最终报告持久化

Summarizer 生成的最终结果使用 `ResearchReport` 统一承载：

```text
ResearchReport.content + ResearchReport.sources
```

Orchestrator 完成运行时将两部分同时写入 Session Memory：

- `ResearchReport.content` → `SessionMemoryEntry.report`
- `ResearchReport.sources` → `SessionMemoryEntry.sources`

`session_rounds.sources` 使用 JSON 文本存储。初始化 Session Memory 时会检查旧表结构；旧数据库缺少该列时自动执行增量迁移，并为历史记录填充空数组。最终 Markdown 仍由输出层根据同一份 `ResearchReport.sources` 渲染参考链接。

## 对抗阶段

Red/Blue 阶段直接接收包含 `content` 和 `sources` 的 `ResearchReport`，不再从 ResearchAgent trajectory 重建来源：

1. Red Agent 使用 `report.sources` 审查来源质量和正文引用。
2. Blue Agent 使用已有 `report.sources`，搜索修复发现的新来源按原始 URL 合并。
3. 每个 Blue 修复批次完成后，统一执行引用收敛：
   - 删除不存在的数字引用；
   - 删除正文不再使用的旧来源和未采用的新来源；
   - 按正文首次出现顺序重新连续编号；
   - 同步更新 `report.content` 和 `report.sources`。
4. 对抗完成后的最终 `ResearchReport.sources` 写入 Session Memory 和最终 Markdown。

历史最佳报告回退使用完整的 `ResearchReport` 深拷贝，因此正文和来源清单会一起回退，不会产生版本错配。

## 边界

- URL 字符串不同即视为不同来源。
- 无 URL 的工具结果不能成为来源。
- output 未引用的搜索结果不会持久化，也不会进入全局来源表。
- Summarizer 不读取 ResearchAgent 的 trajectory 来获取来源。
- `from_memory` 仅描述结果来源，不影响引用收集逻辑。
- Summarizer、Blue Agent 和最终输出层共用同一套引用收敛函数。

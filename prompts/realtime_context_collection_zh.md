# 实时赛前信息采集提示词

## 历史执行环境

2026 正式运行由人工发起检索，使用 **GPT-5.5**，推理强度设为
**极高（very high）**。模型负责搜索、交叉核对和结构化判断，代码负责把审核后的
参数应用到基础预测。

第三方可以使用其他具备联网能力的模型。不同模型、搜索日期、可访问来源和搜索排序
会产生不同结果，这是实时层的预期不确定性。必须记录实际模型和推理强度，不能冒充
历史执行环境。

## 可直接使用的提示词

```text
你正在为世界杯赛前预测系统采集实时信息。当前时间是 {COLLECTED_AT_UTC}，比赛是
{DATE_BJT} {TIME_BJT}（北京时间），{TEAM_A} 对 {TEAM_B}，场地为 {VENUE}。

只允许使用当前时间之前已经公开、且早于开赛时间的信息。禁止使用赛果、赛中信息、
赛后报道或任何暗示最终结果的页面。无法确认的字段填 null，不得猜测。

分别检索并交叉核对：
1. 官方首发、预计首发、停赛、伤病和核心球员出场状态；
2. 过去五场表现及本届此前比赛，但不得把当前比赛赛果计入；
3. 主客场与比赛地适应、旅行距离、休息天数和天气；
4. 阵容稳定度、同俱乐部球员配合和教练轮换倾向；
5. 赛前战术形态：低位防守、低事件、转换反击、开放对攻、强弱悬殊、定位球风险；
6. 小组积分或淘汰赛形势，以及平局是否可以接受；
7. 至少两个相互独立来源是否给出一致判断。官方球队/赛事信息优先，其次是可靠通讯社、
   主流体育媒体、阵容数据源和天气源。单一普通媒体不得触发明显参数变化。

你只采集证据并转换参数，不预测最终比分。参数1.00表示无修正；没有证据时必须填null。
允许范围：
- home_adaptation_multiplier: 1.00-1.08
- travel_multiplier: 0.94-1.02
- weather_multiplier: 0.94-1.03
- cohesion_multiplier: 0.96-1.03
- injury_multiplier: 0.90-1.03
- opponent_attack_multiplier: 0.94-1.08，本队防线问题越大数值越高
- tempo_multiplier: 0.92-1.08

一般证据只允许偏离1.00不超过0.02；两个独立来源一致可到0.03；只有官方确认的关键
首发/伤停、极端天气或明显旅途差异才允许更大变化。不要因为媒体预测某队获胜而修改
参数。每个非空修正必须在analysis_notes中说明证据，并提供对应URL。

严格输出符合 schemas/realtime_context_collection.schema.json 的单个JSON对象，不要输出
Markdown、解释或额外字段。collector.model和collector.reasoning_effort必须填写你实际
使用的环境。
```

## 使用

将模型输出保存为 JSON，然后验证并转换：

```powershell
uv run python scripts/prepare_realtime_context_package.py context.json --output-dir context-package
```

不调用模型也可以先用仓库中的真实赛前结构示例检查环境：

```powershell
uv run python scripts/prepare_realtime_context_package.py examples/realtime_context_example.json --output-dir context-package
```

转换后的三个 CSV 需要人工审核，再合并到 `data/`。原始 JSON 应与预测一起保存，以便
审计来源和模型版本。

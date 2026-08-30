# AUDIT-RUNBOOK · 审查循环执行手册

新窗口从这里开始。**照做，不要重新设计流程。**

---

## STATE（每轮结束后由主窗口更新此块）

```json
{
  "round": 14,
  "head": "9e8c3db",
  "target": "F:/mathematical modeling/skills/figure-forge",
  "verdict_last": "未达标",
  "pending": [
    "N-2 SKILL.md/api.md 未记 ptx / text_color / FF_PRESET；§5 教的 ink() 在 nature 会硬拒",
    "N-5 run_qa 放行图题渲染成 nan / 数组 repr；NaN 坐标的 callout 静默消失",
    "nature 档 5/25 仍失败：algo_convergence comparison_rank fit_residual phase_transition route_map"
  ],
  "fixed_last_round": [
    "N-1 apply_style 认 FF_PRESET，recipe 回到 apply_style()（照文档删 _common 行不再 NameError）",
    "N-3 contour_field grid(False,**props) 反向生效",
    "N-4 ptx(v,'lw') 在 cn 改为恒等；comparison_rank 三元尾巴；4 处光晕改走 kind='pt'"
  ],
  "gate": "交付给用户的东西（产物 + 公开 API + 文档）有没有会让用户拿到错误结果、或在正确用法上被无理由硬拒的缺陷。不看内部整洁度。"
}
```

---

## 每轮六步

### 1 · 交出干净状态

```bash
git status --porcelain      # 必须为空
git log --oneline -1        # 记下 sha，填进 STATE.head
```
不干净就先提交或还原。**不干净不要派评审。**

### 2 · 派评审

用下面的模板，填 `{{}}` 槽位。**必须新起 agent，不能用 fork。**

```
Agent(subagent_type="general-purpose", model=<与上轮不同>, prompt=<模板>)
```

`model` 每轮换（opus / sonnet），两个并行时用不同模型。

### 3 · 逐条验证，不要照收

对评审报的每一条：

```bash
# 跑它给的复现方式；跑不出来的标为"未证实"，不动
```

- 复现成功 → 修
- 复现失败 → 回问或忽略，**不要因为"它说了"就改**
- 它给的**修法**同样要审：问题为真、解法为假的情况出现过多次

### 4 · 修：先红后绿

```bash
# ① 先写测试，确认它失败
python -m pytest tests/test_qa_checks.py -q -k <新测试名>   # 必须 FAILED
# ② 再改实现
# ③ 确认变绿
python -m pytest tests/test_qa_checks.py -q
```

每条检查配**两侧**用例：该触发 + 不该触发。

### 5 · 变异测试

把本轮每处修复**逐个退回**，确认对应测试变红、且红的是该红的那条：

```bash
git stash                                     # 或手工改回单点
python -m pytest tests/test_qa_checks.py -q   # 必须 FAILED，且失败的是预期那条
git stash pop
```
退回后仍全绿 = 这条修复**没有测试守着**，补测试再继续。

### 6 · 全量回归 + 目测 + 提交

```bash
$env:PYTHONIOENCODING="utf-8"
python -m pytest tests/test_qa_checks.py -q          # 全绿
python recipes/run_all.py                            # ALL PASS
$env:FF_PRESET="nature"; python recipes/run_all.py   # 记录 N/25
$env:FF_PRESET=""
python tools/contact_sheet.py                        # 联络表随产物重出
```

然后**用 Read 打开至少 6 张 PNG 看**。自动检查是底线不是上限——出现过多次「QA 全绿但图是坏的」。

提交前逐句核对 commit 正文：**每句"已修复"都要有一条刚跑过的命令支撑。**

```bash
git add -A && git commit -F -    # 用 heredoc，不要 -m（中文引号会被 shell 吃掉）
```

最后回来更新本文件的 STATE 块。

---

## 评审提示模板

```
你是独立评审，第 {{N}} 轮。审查 {{target}}（HEAD = {{sha}}，工作树干净）。

第 {{N-1}} 轮判"{{verdict_last}}"，主窗口声称已修完：
{{fixed_last_round 逐条列出}}

尚未处理：
{{pending 逐条列出}}

**验证，并给出是否达标的判断。**

判定口径：{{gate}}

不要客气，也不要为了凑数编造问题；某类查完没问题就明说"查过，无发现"。
**如果确实达标，请直接说"达到"** —— 不要为了显得尽职而硬凑；反过来若仍有
实质缺陷也不要放水。

A. 逐条核验上面列出的修复（已修 / 部分修 / 未修 / 修错了），附可复现验证。
   本轮重点：{{本轮方向，见下表}}
B. 找本轮新引入的问题。改动集中在 {{文件清单}}。
C. 实测：
   cd {{target}}
   $env:PYTHONIOENCODING="utf-8"
   python -m pytest tests/test_qa_checks.py -q
   python recipes/run_all.py
   $env:FF_PRESET="nature"; python recipes/run_all.py; $env:FF_PRESET=""
   跑完 `git checkout -- gallery/` 还原，离开时工作树干净。
   目测至少 6 张 gallery 图。

输出：1) 核验表 2) 新问题（无就写"无"，每条给可复现验证方式）
      3) 明确判断：是否达标？

**每条发现都要给出可复现的验证方式**，主窗口会逐条验证后才采纳。
```

---

## 本轮方向（每轮换一个，不要重复）

| 轮次特征 | 方向 |
|---|---|
| 前期 | 正确性 bug |
| | 测试质量（断言有没有判别力） |
| | 本轮新引入的回归 |
| | 单点检查的边界压测（构造偏离测试用例的输入） |
| **中期最有效** | **用户视角实跑**：只读入口文档，照骨架写 2–3 张图，记录被硬拒次数与错误信息可操作性 |
| 后期 | 全局收尾：公开 API 签名一致性、文档互相矛盾、逐一目测全部产物 |
| 收尾 | 交叉验证：同时派两个不同模型，比对结论 |

---

## 停止条件

评审说"达到"才停，**主窗口不自己判**。

出现下列任一，说明这轮方向选错了，换方向重派而不是收工：
- 评审说"无发现"但你知道还有没修完的项
- 报回来的全是打磨项（命名、注释、行长）

---

## 本项目的具体命令

| 用途 | 命令 |
|---|---|
| 单元测试 | `python -m pytest tests/test_qa_checks.py -q` |
| 全量出图（cn） | `python recipes/run_all.py` |
| 全量出图（nature） | `FF_PRESET=nature python recipes/run_all.py` |
| 联络表重出 | `python tools/contact_sheet.py` |
| 出图到临时目录 | `FF_GALLERY=<dir> python recipes/<name>.py` |
| 还原产物 | `git checkout -- gallery/` |

编码：所有命令前置 `$env:PYTHONIOENCODING="utf-8"`（PowerShell）或 `export PYTHONIOENCODING=utf-8`（bash），否则中文输出乱码。

---

## 换到别的项目时，替换这四项

1. **STATE.target / gate** —— gate 必须写成一句话，否则评审不知道拿什么尺子量
2. **第 6 步的命令表** —— 换成该项目的"全量重建交付物"
3. **目测环节** —— 交付物不是图像时，换成人工走一遍真实使用路径；这一环不能省
4. **变异测试的退回方式** —— 换成该语言的等价操作

---

## 已知会踩的坑（照这些检查，不要重新发现）

| 症状 | 检查方式 |
|---|---|
| 批处理脚本中途失败、前面改动全丢 | 逐条 `replace` → 立刻语法校验 → 立刻写盘；改完 `grep -c '<旧词>'` 应为 0 |
| 判据解析"渲染结果"（刻度文本、包围盒） | 改成问对象自己是什么（`ax.get_xscale()`、`isinstance`、`artist.filled`），或实测像素 |
| 豁免 fail-open | 任何"满足条件就跳过"的分支，问：取不到值时会怎样？应 fail-closed |
| 检查被 `except` 静默吞掉 | 加一条测试：正常图上不允许出现 "未执行" 字样 |
| 死参数（文档登记、函数体不读） | `grep` 参数名，只出现在签名/docstring/文档三处就是死的 |
| 测试恰好绕开失效区 | 用**偏离**测试用例的输入复验：不同量级、不同轴类型、不同布局 |

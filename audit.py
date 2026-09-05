#!/usr/bin/env python
"""审查循环执行器。新窗口只需要跑这个，不必重新推导流程。

在被审项目目录下运行，或用 --target <路径> / AUDIT_TARGET 指定。
状态存在 <target>/audit.state.json。

    python audit.py status        # 当前轮次、上轮结论、待办
    python audit.py prompt        # 打印填好的评审提示（直接派给子 agent）
    python audit.py check         # 全量回归（测试 + 两档出图 + 干净度）
    python audit.py mutate <文件> # 变异测试：退回某文件到 HEAD~1，跑测试，还原
    python audit.py bump          # 本轮收尾：更新状态、提示下一轮方向

状态存在 audit.state.json（纯 JSON，机器读写，不做字符串手术）。
流程说明在 AUDIT-RUNBOOK.md（静态，不随轮次改动）。
"""
import json
import os
import pathlib
import subprocess
import sys

def _target():
    """被审项目的根目录：优先 --target / AUDIT_TARGET，否则用状态文件里的。"""
    for i, a in enumerate(sys.argv):
        if a == "--target" and i + 1 < len(sys.argv):
            return pathlib.Path(sys.argv[i + 1])
    if os.environ.get("AUDIT_TARGET"):
        return pathlib.Path(os.environ["AUDIT_TARGET"])
    return pathlib.Path.cwd()


ROOT = _target()
STATE = ROOT / "audit.state.json"
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")

JUDGMENTS = """
── 每轮都要守的四条 ──────────────────────────────────────────
① 逐条验证，不要照收。先跑评审给的复现方式；复现不出来的标"未证实"，不动。
   它给的**修法**同样要审——问题为真、解法为假的情况反复出现。
② 先红后绿。先写测试并确认 FAILED，再改实现。每条检查配两侧用例：
   该触发 + 不该触发。
③ 变异测试是硬门槛。`python audit.py mutate <改动文件>` 退回后必须变红，
   **且红的是该红的那条**。仍全绿 = 这处修复没有测试守着，补测试再继续。
④ 停止由评审说了算，主窗口不自己判。评审说"无发现"但你知道还有没修完的项，
   或报回来的全是打磨项 → 方向选错，换方向重派而不是收工。

提交前：commit 正文里每句"已修复"都要有一条刚跑过的命令支撑。
        这一环栽过三次，每次都发生在"总结战果"的时候。

下一步  python audit.py prompt   拿评审提示，派新 agent（别 fork，每轮换 model）
        python audit.py traps    开工前过一遍「已知会踩的坑」
"""

TRAPS = """
── 已知会踩的坑（照着查，不要重新发现）──────────────────────
批处理脚本中途失败、前面改动全丢
    逐条 replace → 立刻语法校验 → 立刻写盘；改完 grep -c '<旧词>' 应为 0
判据解析"渲染结果"（刻度文本、包围盒）
    改成问对象自己是什么（get_xscale() / isinstance / 属性），或实测像素
豁免 fail-open
    任何"满足条件就跳过"的分支，问：取不到值时会怎样？应 fail-closed
检查被 except 静默吞掉
    audit.py check 已自动检测「未执行/未完成」
死参数（文档登记、函数体不读）
    grep 参数名，只出现在签名/docstring/文档三处就是死的
测试恰好绕开失效区
    用**偏离**现有用例的输入复验：不同量级、类型、布局
库函数当场警告但没人看
    跑全量时留意 UserWarning——grid(False, **props) 那次就是这么漏的
文档说 A、库要 B
    照文档主路径**真的走一遍**（复制模板、删掉文档让删的行、跑起来）
公开 API 的字符串参数无校验
    传一个语义合理但非法的值（orientation="horizontal"），看是静默走错分支还是报错
并行派两个评审
    各给一个 git worktree，否则会撞上对方的中间状态

换到别的项目：改 audit.state.json 的 target / gate，改 cmd_check() 的命令表。
    gate 必须写成一句话（评审据此判达标）；交付物不是图像时，
    "目测产物"换成人工走一遍真实使用路径——这一环不能省。
"""

DIRECTIONS = [
    "正确性 bug",
    "测试质量：断言有没有判别力（把修复退回，测试是否真的变红）",
    "本轮新引入的回归",
    "单点检查的边界压测：构造**偏离**现有测试用例的输入（不同量级、轴类型、布局）",
    "用户视角实跑：只读入口文档、照骨架写 2–3 张图，记录被硬拒次数与错误信息可操作性",
    "全局收尾：公开 API 签名一致性、文档互相矛盾、逐一目测全部产物",
    "交叉验证：同时派两个不同模型，比对结论",
]


def _load():
    return json.loads(STATE.read_text(encoding="utf-8"))


def _save(d):
    STATE.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n",
                     encoding="utf-8")


def _run(cmd, env_extra=None):
    e = dict(ENV, **(env_extra or {}))
    return subprocess.run(cmd, shell=True, cwd=ROOT, env=e,
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def _sha():
    return _run("git log --oneline -1").stdout.split()[0]


def _dirty():
    return bool(_run("git status --porcelain").stdout.strip())


def cmd_traps():
    print(TRAPS)


def cmd_status():
    d = _load()
    if d.get("what"):
        print("项目      " + d["what"])
    if d.get("history"):
        print("历史      " + d["history"])
    b = d.get("baseline") or {}
    if b:
        print("基线      " + " / ".join(f"{k}={v}" for k, v in b.items()))
    print()
    ip = d.get("in_progress")
    if ip:
        # 状态文件停在上一轮 = 新窗口以为什么都没发生，那就废了它的全部
        # 价值。轮次没收尾也要如实写在这里。
        print(f"★进行中  第 {ip['round']} 轮：{ip['状态']}")
        if ip.get("打分历史"):
            print("  打分    " + " / ".join(ip["打分历史"]))
        print()
    print(f"轮次      第 {d['round']} 轮（最后收尾的一轮）")
    print(f"HEAD      {_sha()}   （状态里记的是 {d['head']}）")
    print(f"工作树    {'★不干净——先提交或还原' if _dirty() else '干净'}")
    print(f"上轮结论  {d['verdict_last']}")
    print(f"判定口径  {d['gate']}")
    if d["fixed_last_round"]:
        print("\n上轮已修：")
        for x in d["fixed_last_round"]:
            print("  -", x)
    if d["pending"]:
        print("\n待处理：")
        for x in d["pending"]:
            print("  -", x)
    print(f"\n本轮方向  {DIRECTIONS[d['round'] % len(DIRECTIONS)]}")
    print(JUDGMENTS)


def cmd_prompt():
    d = _load()
    fixed = "\n".join(f"- {x}" for x in d["fixed_last_round"]) or "（无）"
    pend = "\n".join(f"- {x}" for x in d["pending"]) or "（无）"
    print(f"""你是独立评审，第 {d['round'] + 1} 轮。审查 {d['target']}（HEAD = {_sha()}，工作树干净）。

第 {d['round']} 轮判"{d['verdict_last']}"，主窗口声称已修完：
{fixed}

尚未处理：
{pend}

**验证，并给出是否达标的判断。**

判定口径：{d['gate']}

不要客气，也不要为了凑数编造问题；某类查完没问题就明说"查过，无发现"。
**如果确实达标，请直接说"达到"** —— 不要为了显得尽职而硬凑；反过来若仍有
实质缺陷也不要放水。

A. 逐条核验上面列出的修复（已修 / 部分修 / 未修 / 修错了），附可复现验证。
   本轮重点：{DIRECTIONS[d['round'] % len(DIRECTIONS)]}
B. 找本轮新引入的问题。改动集中在 {d.get('changed', '见 git diff')}。
C. 实测：
   cd {d['target']}
   $env:PYTHONIOENCODING="utf-8"
   python audit.py check
   跑完 `git checkout -- gallery/` 还原，离开时工作树干净。
   目测至少 6 张 gallery 图。

输出：1) 核验表 2) 新问题（无就写"无"，每条给可复现验证方式）
      3) 明确判断：是否达标？

**每条发现都要给出可复现的验证方式**，主窗口会逐条验证后才采纳。""")


def cmd_check():
    bad = 0
    summary = {}
    print("[1/4] 单元测试")
    # 跑 tests/ 全量，不是只跑 test_qa_checks.py。此前只跑那一个文件，
    # 第 16 轮新增的 test_manifest.py（15 条）与 test_packaging.py（7 条）
    # 完全不在审查门禁内——门禁盖不住的测试等于没有门禁，而这个工具的
    # 全部价值就是别让我把部分完成写成完成。zcode 在 P1/P2 交付时如实
    # 报了这个缺口。
    # -rf：失败时把「哪一条」打出来。此前只印汇总行，于是第 17 轮出现的
    # 那条「只在 audit 里红、单跑 pytest 又全绿」的间歇失败，两次都抓不到
    # 名字——报了「1 failed」却说不出是谁，等于没报。
    r = _run("python -m pytest tests/ -q -rf")
    lines = [l for l in r.stdout.strip().split(chr(10)) if l.strip()]
    print("     ", *lines[-1:])
    if r.returncode:
        bad += 1
        for l in lines:
            if l.startswith(("FAILED", "ERROR")):
                print("      ★", l[:150])

    # nature 先跑、cn **后**跑。此前顺序相反，nature 的产物覆盖掉 cn 的，
    # 于是「谁最后 commit 谁定档」——入库的 gallery/ 实际是 nature 产物，
    # 而 README 写的是「python recipes/run_all.py 可复现」（默认 cn 档）：
    # 照文档跑，39 张里 36 张对不上。第 16 轮 opus 评审逮到的。
    for i, (label, extra) in enumerate((("nature", {"FF_PRESET": "nature"}),
                                        ("cn", {})), 2):
        print(f"[{i}/4] 全量出图 · {label}")
        r = _run("python recipes/run_all.py", extra)
        ok = r.stdout.count("[OK ]")
        total = ok + r.stdout.count("[FAIL]")
        allpass = "ALL PASS" in r.stdout
        print(f"      {ok}/{total} {'ALL PASS' if allpass else ''}")
        for ln in r.stdout.split("\n"):
            if ln.startswith("- ") or ln.startswith("[FAIL]"):
                print("      ", ln.strip()[:110])
        # 两档都要求全过。此前只有 cn 计入 bad，nature 挂了照样打印
        # "全部通过"——第 16 轮 codex 评审逮到：审查工具自己在粉饰战果，
        # 而这个工具的全部价值就是别让我把部分完成写成完成。
        if not allpass:
            bad += 1
        summary[label] = f"{ok}/{total}" + ("" if allpass else "  ★未全过")
        notes = [l for l in r.stdout.split("\n")
                 if "未执行" in l or "未完成" in l]
        if notes:
            bad += 1
            print("      ★检查被静默跳过：", notes[0][:90])

    print("[4/4] 入库产物是否与默认档一致 + 干净度")
    # 必须在 `git checkout -- gallery/` **之前**比对：只还原不比对，
    # 等于把出图漂移整个抹掉后才报「工作树 干净」，这类漂移在这套流程里
    # 原理上不可见。上一步刚跑完 cn（默认档），此刻 gallery/ 里就是默认档
    # 产物，与 HEAD 有差异即说明入库的不是默认档产物。
    # 只比 PNG：pdf/svg 里嵌了 /CreationDate 时间戳，每次出图必然不同。
    # 把它们算进来，这条检查每次都报警，很快就没人看了——检查失效正是
    # 这么发生的。PNG 是视觉产物，逐字节可复现，正好是该比的那一层。
    drift = _run("git status --porcelain -- gallery/*.png").stdout.strip()
    n_drift = len([l for l in drift.splitlines() if l.strip()])
    if n_drift:
        bad += 1
        print(f"      ★入库产物与默认档不符：{n_drift} 个文件"
              f"（README 承诺 `python recipes/run_all.py` 可复现）")
    else:
        print("      入库产物与默认档一致")
    _run("git checkout -- gallery/")
    print("      工作树", "★不干净" if _dirty() else "干净")
    print("\n=>", "cn " + summary.get("cn", "?")
          + " / nature " + summary.get("nature", "?"))
    print("=>", "全部通过" if not bad else f"★{bad} 项需处理")
    return bad


def cmd_mutate(path):
    """把某文件退回 HEAD~1，跑测试，确认变红，再还原。"""
    if _dirty():
        sys.exit("工作树不干净，先提交或还原")
    prev = _run(f"git show HEAD~1:{path}")
    if prev.returncode:
        sys.exit(f"取不到 HEAD~1:{path}")
    p = ROOT / path
    keep = p.read_text(encoding="utf-8")
    p.write_text(prev.stdout, encoding="utf-8")
    try:
        r = _run("python -m pytest tests/test_qa_checks.py -q")
        failed = [l for l in r.stdout.split("\n") if l.startswith("FAILED")]
        print(f"退回 {path} 后：")
        if failed:
            for l in failed:
                print("   红 ", l.replace("FAILED ", "")[:100])
        else:
            print("   ★全绿 —— 这处修复没有测试守着，补测试再继续")
    finally:
        p.write_text(keep, encoding="utf-8")
    print("已还原")


def cmd_bump():
    d = _load()
    if _dirty():
        sys.exit("工作树不干净，先提交本轮改动再 bump")
    d["round"] += 1
    d["head"] = _sha()
    print(f"→ 第 {d['round']} 轮，HEAD {d['head']}")
    print(f"→ 下轮方向：{DIRECTIONS[d['round'] % len(DIRECTIONS)]}")
    print("\n手工更新这三项后保存 audit.state.json：")
    print("  verdict_last      本轮评审的结论原话")
    print("  fixed_last_round  本轮实际修了什么（每条要有跑过的命令支撑）")
    print("  pending           评审报了但本轮没处理的")
    _save(d)


if __name__ == "__main__":
    a = sys.argv[1:] or ["status"]
    fn = {"status": cmd_status, "prompt": cmd_prompt, "check": cmd_check,
          "bump": cmd_bump, "traps": cmd_traps}.get(a[0])
    if a[0] == "mutate":
        cmd_mutate(a[1])
    elif fn:
        sys.exit(fn() or 0)
    else:
        print(__doc__)

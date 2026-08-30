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
    print(f"轮次      第 {d['round']} 轮")
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
    print("[1/4] 单元测试")
    r = _run("python -m pytest tests/test_qa_checks.py -q")
    tail = [l for l in r.stdout.strip().split("\n") if l.strip()][-1:]
    print("     ", *tail)
    if r.returncode:
        bad += 1

    for i, (label, extra) in enumerate((("cn", {}),
                                        ("nature", {"FF_PRESET": "nature"})), 2):
        print(f"[{i}/4] 全量出图 · {label}")
        r = _run("python recipes/run_all.py", extra)
        ok = r.stdout.count("[OK ]")
        allpass = "ALL PASS" in r.stdout
        print(f"      {ok}/25 {'ALL PASS' if allpass else ''}")
        for ln in r.stdout.split("\n"):
            if ln.startswith("- ") or ln.startswith("[FAIL]"):
                print("      ", ln.strip()[:110])
        # 只有 cn 档要求全过；nature 档记录进度
        if label == "cn" and not allpass:
            bad += 1
        notes = [l for l in r.stdout.split("\n")
                 if "未执行" in l or "未完成" in l]
        if notes:
            bad += 1
            print("      ★检查被静默跳过：", notes[0][:90])

    print("[4/4] 产物还原 + 干净度")
    _run("git checkout -- gallery/")
    print("      工作树", "★不干净" if _dirty() else "干净")
    print("\n=>", "全部通过" if not bad else f"★{bad} 项需处理")
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
          "bump": cmd_bump}.get(a[0])
    if a[0] == "mutate":
        cmd_mutate(a[1])
    elif fn:
        sys.exit(fn() or 0)
    else:
        print(__doc__)

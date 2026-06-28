"""
Bot 大乱斗 — 从 bot_All_Versions 动态加载任意 bot 对战

用法:
    python3 tournament.py v1_starter v2_baseline v3_cnn_Res_Mish_SE_standard v14best_cnnSE_preload2x
    python3 tournament.py -n 200 v14best v14best v16_dual_light v14best
    python3 tournament.py --list
"""

import os, sys, time, random, argparse, importlib.util
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch

# ── 路径 ──────────────────────────────────────────────
TOOLS_DIR = Path(__file__).resolve().parent
CODE_ROOT = TOOLS_DIR.parent
BOT_ALL = CODE_ROOT / "bot_All_Versions"
ENGINE_DIR = TOOLS_DIR / "baseline_engine"

sys.path.insert(0, str(ENGINE_DIR))
from env import MahjongGBEnv

torch.set_num_threads(1)

# ══════════════════════════════════════════════════════════════
#  Bot 加载
# ══════════════════════════════════════════════════════════════

WEIGHT_GLOB = ["*.pt", "checkpoint/*.pt", "weights/*.pt"]


def _find_weight(bot_dir: Path) -> Path | None:
    for pattern in WEIGHT_GLOB:
        for p in sorted(bot_dir.glob(pattern)):
            return p
    return None


def _find_model_class(module):
    """在模块中找到 nn.Module 子类（非 nn.Module 本身）"""
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, torch.nn.Module) and obj is not torch.nn.Module:
            return obj
    return None


def load_bot(version_name: str) -> dict:
    """
    加载一个 bot 版本。
    返回 dict: {name, feat_cls, model_cls, in_channels, weight_path}
    """
    bot_dir = BOT_ALL / version_name
    if not bot_dir.is_dir():
        # 尝试模糊匹配
        matches = [d.name for d in BOT_ALL.iterdir()
                   if d.is_dir() and version_name in d.name]
        if len(matches) == 1:
            bot_dir = BOT_ALL / matches[0]
            version_name = matches[0]
        else:
            print(f"❌ 版本不存在: '{version_name}'")
            if matches:
                print(f"   相近: {matches}")
            sys.exit(1)

    sys.path.insert(0, str(bot_dir))

    # ── feature.py ──
    feat_path = bot_dir / "feature.py"
    if not feat_path.exists():
        print(f"❌ {version_name}: 缺少 feature.py")
        sys.exit(1)

    spec_f = importlib.util.spec_from_file_location(f"_{version_name}_f", feat_path)
    mod_f = importlib.util.module_from_spec(spec_f)
    sys.modules[f"_{version_name}_f"] = mod_f
    spec_f.loader.exec_module(mod_f)

    FeatCls = getattr(mod_f, "FeatureAgent", None)
    if FeatCls is None:
        for n in dir(mod_f):
            obj = getattr(mod_f, n)
            if isinstance(obj, type) and hasattr(obj, "OBS_SIZE"):
                FeatCls = obj
                break
    if FeatCls is None:
        print(f"❌ {version_name}: 未找到 FeatureAgent 类")
        sys.exit(1)

    in_channels = getattr(FeatCls, "OBS_SIZE", 160)

    # ── model.py ──
    model_path = bot_dir / "model.py"
    if not model_path.exists():
        print(f"❌ {version_name}: 缺少 model.py")
        sys.exit(1)

    spec_m = importlib.util.spec_from_file_location(f"_{version_name}_m", model_path)
    mod_m = importlib.util.module_from_spec(spec_m)
    sys.modules[f"_{version_name}_m"] = mod_m
    spec_m.loader.exec_module(mod_m)

    ModelCls = _find_model_class(mod_m)
    if ModelCls is None:
        print(f"❌ {version_name}: 未找到 nn.Module 子类")
        sys.exit(1)

    # ── 权重 ──
    weight_path = _find_weight(bot_dir)

    return {
        "name": version_name,
        "feat_cls": FeatCls,
        "model_cls": ModelCls,
        "in_channels": in_channels,
        "weight_path": weight_path,
    }


# ══════════════════════════════════════════════════════════════
#  Agent 工厂
# ══════════════════════════════════════════════════════════════

def make_agent(bot_info: dict, seatWind: int):
    FeatCls = bot_info["feat_cls"]
    ModelCls = bot_info["model_cls"]
    in_channels = bot_info["in_channels"]
    weight_path = bot_info["weight_path"]

    class Agent(FeatCls):
        def __init__(self, seatWind):
            super().__init__(seatWind)
            try:
                self._m = ModelCls(in_channels=in_channels)
            except TypeError:
                self._m = ModelCls()
            if weight_path:
                ckpt = torch.load(str(weight_path), map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict):
                    ckpt = ckpt.get("model", ckpt)
                self._m.load_state_dict(ckpt, strict=False)
            self._m.train(False)

        def request2obs(self, r):
            return super().request2obs(r)

    return Agent(seatWind)


# ══════════════════════════════════════════════════════════════
#  Tournament
# ══════════════════════════════════════════════════════════════

def run(bot_infos: list, n: int = 600, seed: int | None = None):
    if seed is None:
        seed = int(time.time())
    rng = random.Random(seed)

    names = [b["name"] for b in bot_infos]
    pn = [f"player_{i}" for i in range(1, 5)]

    wins = Counter()
    scores = defaultdict(list)
    rank_counts = Counter()
    errs = 0
    t0 = time.time()

    for gi in range(n):
        # 每局随机选 4 个 bot（允许同 bot 占多家）
        players = rng.choices(names, k=4)

        try:
            env = MahjongGBEnv(config={
                "agent_clz": bot_infos[0]["feat_cls"],
                "duplicate": True,
                "variety": 1000000,
            })
            env.reset()
            agents = [make_agent(bot_infos[names.index(p)], i) for i, p in enumerate(players)]
            env.agents = agents

            for i, a in enumerate(agents):
                a.request2obs(f"Wind {env.prevalentWind}")
                a.request2obs(" ".join(["Deal", *env.hands[i]]))
            env.obs = {0: agents[0].request2obs(f"Draw {env.curTile}")}
            obs = env._obs()

            while True:
                acts = {}
                for pk, ob in obs.items():
                    idx = pn.index(pk)
                    a = agents[idx]
                    st = {
                        "observation": torch.from_numpy(np.expand_dims(ob["observation"], 0)),
                        "action_mask": torch.from_numpy(np.expand_dims(ob["action_mask"], 0)),
                    }
                    with torch.no_grad():
                        output = a._m(st)
                    # 适配不同输出格式
                    if isinstance(output, tuple):
                        logits = output[0]
                    elif isinstance(output, dict):
                        logits = output.get("logits", output.get("policy",
                                  list(output.values())[0]))
                    else:
                        logits = output
                    acts[pk] = logits.numpy().flatten().argmax()

                obs, rewards, done = env.step(acts)
                if done:
                    break

            for i, nm in enumerate(players):
                scores[nm].append(rewards.get(pn[i], 0))
            order = sorted(zip(players, [rewards.get(nn, 0) for nn in pn]),
                           key=lambda x: -x[1])
            for rk, (nm, _) in enumerate(order):
                rank_counts[(nm, rk)] += 1
                if rk == 0:
                    wins[nm] += 1

        except Exception as e:
            errs += 1
            if errs <= 3:
                import traceback
                traceback.print_exc()
            if errs > 10:
                print("太多错误，退出")
                break
            continue

        if (gi + 1) % 20 == 0 or gi < 3:
            el = time.time() - t0
            gps = (gi + 1) / el if el > 0 else 0
            print(f"  [{gi+1}/{n}] {gps:.1f} 局/秒 | 报错 {errs}", flush=True)
            ranked = sorted(names, key=lambda nm: -(sum(scores.get(nm, []))))
            print("  ", " | ".join(
                f"{nm}: {sum(scores.get(nm, [])):+.0f}" for nm in ranked), flush=True)

    el = time.time() - t0
    total_games = gi + 1
    print(f"\n{'='*60}")
    print(f"{total_games} 局  {el:.0f}s  报错 {errs}")
    print(f"{'='*60}\n")

    hdr = f"{'Bot':<40} {'总分':>8} {'场均':>8} {'胜率':>7}  {'🥇':>7}  {'🥈':>7}  {'🥉':>7}"
    print(hdr)
    print("-" * len(hdr))
    ranked_final = sorted(names, key=lambda nm: -sum(scores.get(nm, [])))
    for nm in ranked_final:
        w = wins.get(nm, 0)
        wr = w / max(total_games, 1) * 100
        s = scores.get(nm, [])
        total_s = sum(s) if s else 0
        avg = np.mean(s) if s else 0
        t = sum(rank_counts.get((nm, r), 0) for r in range(4))
        if t:
            print(f"{nm:<40} {total_s:+8.0f}  {avg:+7.1f}  {wr:5.1f}%  "
                  f"{rank_counts.get((nm,0),0)/t*100:5.1f}%  "
                  f"{rank_counts.get((nm,1),0)/t*100:5.1f}%  "
                  f"{rank_counts.get((nm,2),0)/t*100:5.1f}%")
    sys.stdout.flush()


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def _list_bots():
    print(f"可用 bot ({BOT_ALL}):")
    for d in sorted(BOT_ALL.iterdir()):
        if d.is_dir() and (d / "feature.py").exists() and (d / "model.py").exists():
            print(f"  {d.name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Bot 大乱斗 — 从 bot_All_Versions 动态加载任意 bot 对战")
    p.add_argument("bots", nargs="*", help="4 个 bot 版本名称 (可重复，支持模糊匹配)")
    p.add_argument("-n", type=int, default=600, help="对局数 (默认 600)")
    p.add_argument("--seed", type=int, help="随机种子")
    p.add_argument("--list", action="store_true", help="列出可用 bot")
    args = p.parse_args()

    if args.list:
        _list_bots()
        sys.exit(0)

    if len(args.bots) < 1:
        print("用法: python3 tournament.py [-n 200] <bot1> <bot2> <bot3> <bot4>")
        print("      python3 tournament.py --list")
        print("\n示例:")
        print("  python3 tournament.py v1_starter v2_baseline v3_cnn_Res_Mish_SE_standard v14best_cnnSE_preload2x")
        print("  python3 tournament.py -n 200 v14best v14best v14best v14best")
        sys.exit(1)

    if len(args.bots) > 4:
        print(f"⚠️  指定了 {len(args.bots)} 个 bot，只用前 4 个")
        args.bots = args.bots[:4]

    # ── 加载 bot ──
    print("加载 bot...")
    bot_infos = []
    for bname in args.bots:
        info = load_bot(bname)
        w = info["weight_path"]
        if w:
            print(f"  ✅ {info['name']:<45s} 权重: {w.name}")
        else:
            print(f"  ⚠️  {info['name']:<45s} 无 .pt 文件 → 随机初始化")
        bot_infos.append(info)

    # 如果少于 4 个，从已有中补足
    while len(bot_infos) < 4:
        bot_infos.append(bot_infos[-1])
        print(f"  🔄 复制 {bot_infos[-1]['name']} 补齐 4 个")

    print(f"\n开始 {args.n} 局...\n")
    run(bot_infos, n=args.n, seed=args.seed)

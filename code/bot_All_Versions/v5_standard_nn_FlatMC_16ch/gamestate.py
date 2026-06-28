"""国标麻将 GameState — NN + FlatMC 混合决策."""

import sys
import time
import random
import math
import numpy as np
from collections import defaultdict, Counter
from MahjongGB import MahjongShanten, MahjongFanCalculator

from state_encoder import StateEncoder, OFFSET_OBS, OFFSET_ACT, OFFSET_TILE, TILE_LIST, OBS_SIZE
from nn_policy import NNEvaluator

# 常量

TILE_TYPES = [
    *('W%d' % (i + 1) for i in range(9)),
    *('T%d' % (i + 1) for i in range(9)),
    *('B%d' % (i + 1) for i in range(9)),
    *('F%d' % (i + 1) for i in range(4)),
    *('J%d' % (i + 1) for i in range(3)),
]
TILE_TO_IDX = {t: i for i, t in enumerate(TILE_TYPES)}
TOTAL_COPIES = {t: 4 for t in TILE_TYPES}

# SIMCAT 模拟参数
SIM_TOTAL_BUDGET = 3000
SIM_INITIAL_PER_ACTION = 60
SIM_BATCH_SIZE = 40
PP_RD = 2.0
SIM_MAX_ROUNDS = 10
TOP_K_CANDIDATES = 5

TIME_LIMIT = 4.5
TIME_CHECK_INTERVAL = 25

# NN 融合参数
NN_ALPHA = 0.5          # 候选排序: alpha * nn_prob + (1-alpha) * shanten_score
NN_BETA_BASE = 0.3      # VHR: 基础beta（NN权重）
NN_BETA_MAX = 0.8        # VHR: 最大beta
NN_VHR_SHORT_DRAWS = 3   # VHR 短rollout步数
NN_VALUE_BATCH_SIZE = 32  # 批量推理大小
NN_CONFIDENCE_SKIP = 0.85  # NN combined_score超过此值时减少FlatMC预算
NN_MELD_THRESHOLD = 0.03  # 鸣牌 value 提升阈值


def classify_tatsu_breaking(hand, pack, base_shanten):
    """将候选弃牌分为 tatsu-breaking set 和 non-tatsu-breaking set。"""
    tb_set = []
    ntb_set = []
    for t in set(hand):
        tmp = list(hand)
        tmp.remove(t)
        try:
            s = MahjongShanten(hand=tuple(tmp), pack=tuple(pack))
        except Exception:
            s = 999
        if s > base_shanten:
            tb_set.append(t)
        else:
            ntb_set.append(t)
    return tb_set, ntb_set


def greedy_discard(hand, pack):
    """贪心弃牌：选弃后向听最低的牌。注意：会临时修改 hand 列表！"""
    best_t = hand[0]
    best_s = 999
    for t in set(hand):
        hand.remove(t)
        try:
            s = MahjongShanten(hand=tuple(hand), pack=tuple(pack))
        except Exception:
            s = 999
        hand.append(t)
        if s < best_s:
            best_s = s
            best_t = t
    return best_t, best_s


def _sigmoid(x):
    """将 value head 输出映射到 [0, 1]"""
    return 1.0 / (1.0 + math.exp(-max(-10, min(10, x))))


# GameState — 游戏状态 + NN+FlatMC 混合决策引擎


class GameState:
    """追踪完整游戏状态，提供 NN增强版 SIMCAT 决策。"""

    def __init__(self, seatWind, nn_evaluator=None):
        self.seatWind = seatWind
        self.prevalentWind = 0
        self.hand = []
        self.packs = [[] for _ in range(4)]
        self.tileWall = [21] * 4
        self.shownTiles = defaultdict(int)
        self.oppDiscards = [[] for _ in range(4)]
        self.wallLast = False
        self.isAboutKong = False
        self.curTile = ''
        self.tileFrom = 0
        self._start_time = 0.0

        # NN 评估器
        self.nn = nn_evaluator

        # 缓存当前局面的 NN 评估结果（每步只推理一次）
        self._nn_cache = None  # (probs, value, obs, mask)


    def can_hu(self, winTile, isSelfDrawn=False, isAboutKong=False):
        try:
            fans = MahjongFanCalculator(
                pack=tuple(self.packs[0]), hand=tuple(self.hand),
                winTile=winTile, flowerCount=0, isSelfDrawn=isSelfDrawn,
                is4thTile=(self.shownTiles[winTile] + isSelfDrawn) == 4,
                isAboutKong=isAboutKong, isWallLast=self.wallLast,
                seatWind=self.seatWind, prevalentWind=self.prevalentWind,
                verbose=True)
            return sum(p * c for p, c, _, _ in fans) >= 8
        except Exception:
            return False


    def shanten(self, hand=None, pack=None):
        try:
            h = tuple(hand if hand is not None else self.hand)
            p = tuple(pack if pack is not None else self.packs[0])
            return MahjongShanten(hand=h, pack=p)
        except Exception:
            return 999


    def _time_left(self, margin=0.15):
        if self._start_time == 0.0:
            return True
        return time.time() - self._start_time < TIME_LIMIT - margin


    def _get_nn_eval(self):
        """获取当前局面的 NN 评估结果（带缓存）"""
        if self._nn_cache is not None:
            return self._nn_cache
        if self.nn is None:
            return None
        obs, mask = StateEncoder.encode(self)
        probs, value = self.nn.evaluate(obs, mask)
        self._nn_cache = (probs, value, obs, mask)
        return self._nn_cache

    def _invalidate_nn_cache(self):
        """局面变化后清除缓存"""
        self._nn_cache = None

    # 1: 剩余牌池构造

    def _remaining_pool(self):
        count = dict(TOTAL_COPIES)
        for t in self.hand:
            count[t] = max(0, count[t] - 1)
        for i in range(4):
            for pkt, tile, _ in self.packs[i]:
                if pkt == 'CHI':
                    c, n = tile[0], int(tile[1])
                    for dn in (-1, 0, 1):
                        ct = c + str(n + dn)
                        count[ct] = max(0, count.get(ct, 0) - 1)
                elif pkt == 'GANG' and tile == 'CONCEALED':
                    pass  # 暗杠看不到
                else:
                    count[tile] = max(0, count.get(tile, 0) - (4 if pkt == 'GANG' else 3))
        for t, cnt in self.shownTiles.items():
            count[t] = max(0, count.get(t, 0) - cnt)
        pool = []
        for t, c in count.items():
            pool.extend([t] * c)
        return pool

    # 2: 待牌计数 (听牌时)

    def _wait_count(self, hand, pack):
        wait_types = 0
        wait_copies = 0
        pool_counter = Counter(self._remaining_pool())
        for t in hand:
            pool_counter[t] = pool_counter.get(t, 0) + 1
        saved = list(hand)
        for tile in TILE_TYPES:
            copies = pool_counter.get(tile, 0)
            if copies == 0:
                continue
            saved.append(tile)
            try:
                s = MahjongShanten(hand=tuple(saved), pack=tuple(pack))
            except Exception:
                s = 999
            saved.pop()
            if s == -1:
                wait_types += 1
                wait_copies += copies
        return wait_types, wait_copies

    # 3: Flat Monte Carlo 胜率模拟

    def _simulate_win_rate(self, hand, pack, n_sims, max_draws, pool=None):
        """
        FlatMC 模拟：对给定 afterstate 做 n_sims 次模拟。
        返回 (win_rate, tenpai_rate)。
        """
        if pool is None:
            pool = self._remaining_pool()
        if len(pool) < max_draws * 4:
            max_draws = max(1, len(pool) // 4)

        win_count = 0
        tenpai_count = 0

        for i in range(n_sims):
            if i > 0 and i % TIME_CHECK_INTERVAL == 0:
                if not self._time_left(margin=0.15):
                    n_done = i
                    return (win_count / n_done if n_done > 0 else 0.0,
                            tenpai_count / n_done if n_done > 0 else 0.0)
            sim_pool = list(pool)
            random.shuffle(sim_pool)
            sim_hand = list(hand)
            sim_pack = list(pack)
            reached_tenpai = False

            for draw_idx in range(max_draws):
                if len(sim_pool) < 4:
                    break
                for _ in range(3):
                    sim_pool.pop()
                draw_tile = sim_pool.pop()
                sim_hand.append(draw_tile)

                try:
                    s_quick = MahjongShanten(hand=tuple(sim_hand), pack=tuple(sim_pack))
                    if s_quick == -1:
                        fans = MahjongFanCalculator(
                            pack=tuple(sim_pack), hand=tuple(sim_hand),
                            winTile=draw_tile, flowerCount=0, isSelfDrawn=True,
                            is4thTile=self.shownTiles[draw_tile] == 4,
                            isAboutKong=False, isWallLast=False,
                            seatWind=self.seatWind, prevalentWind=self.prevalentWind,
                            verbose=True)
                        if sum(p * c for p, c, _, _ in fans) >= 8:
                            win_count += 1
                            tenpai_count += 1
                            break
                except Exception:
                    pass

                best_t, best_s = greedy_discard(sim_hand, sim_pack)
                sim_hand.remove(best_t)

                if best_s <= 0 and not reached_tenpai:
                    reached_tenpai = True
                    tenpai_count += 1
            else:
                if not reached_tenpai:
                    try:
                        s = MahjongShanten(hand=tuple(sim_hand), pack=tuple(sim_pack))
                        if s <= 0:
                            tenpai_count += 1
                    except Exception:
                        pass

        win_rate = win_count / n_sims if n_sims > 0 else 0.0
        tenpai_rate = tenpai_count / n_sims if n_sims > 0 else 0.0
        return win_rate, tenpai_rate

    # 3b: Value-Head Rollout Replacement (VHR) — NN增强版模拟

    def _simulate_win_rate_vhr(self, hand, pack, n_sims, max_draws, pool=None):
        """
        VHR: 短rollout (1-3步) + NN value 批量评估叶节点

        相比纯 FlatMC:
        - 减少每条轨迹的 rollout 步数（3步 → 1-3步）
        - 用 NN value head 评估叶节点，替代长随机 rollout
        - 批量推理提高效率
        """
        if self.nn is None:
            return self._simulate_win_rate(hand, pack, n_sims, max_draws, pool)

        if pool is None:
            pool = self._remaining_pool()
        short_draws = min(max_draws, NN_VHR_SHORT_DRAWS)
        if len(pool) < short_draws * 4:
            short_draws = max(1, len(pool) // 4)

        # 收集所有模拟的叶节点状态
        leaf_hands = []
        leaf_packs = []
        sim_results = []  # (win, tenpai) 每条轨迹的直接结果

        for i in range(n_sims):
            if i > 0 and i % TIME_CHECK_INTERVAL == 0:
                if not self._time_left(margin=0.15):
                    break

            sim_pool = list(pool)
            random.shuffle(sim_pool)
            sim_hand = list(hand)
            sim_pack = list(pack)
            reached_tenpai = False
            won = False

            for draw_idx in range(short_draws):
                if len(sim_pool) < 4:
                    break
                for _ in range(3):
                    sim_pool.pop()
                draw_tile = sim_pool.pop()
                sim_hand.append(draw_tile)

                try:
                    s_quick = MahjongShanten(hand=tuple(sim_hand), pack=tuple(sim_pack))
                    if s_quick == -1:
                        fans = MahjongFanCalculator(
                            pack=tuple(sim_pack), hand=tuple(sim_hand),
                            winTile=draw_tile, flowerCount=0, isSelfDrawn=True,
                            is4thTile=self.shownTiles[draw_tile] == 4,
                            isAboutKong=False, isWallLast=False,
                            seatWind=self.seatWind, prevalentWind=self.prevalentWind,
                            verbose=True)
                        if sum(p * c for p, c, _, _ in fans) >= 8:
                            won = True
                            reached_tenpai = True
                            break
                except Exception:
                    pass

                best_t, best_s = greedy_discard(sim_hand, sim_pack)
                sim_hand.remove(best_t)
                if best_s <= 0 and not reached_tenpai:
                    reached_tenpai = True

            if not won:
                if not reached_tenpai:
                    try:
                        s = MahjongShanten(hand=tuple(sim_hand), pack=tuple(sim_pack))
                        if s <= 0:
                            reached_tenpai = True
                    except Exception:
                        pass

            sim_results.append((won, reached_tenpai))
            if not won:
                leaf_hands.append(list(sim_hand))
                leaf_packs.append(list(sim_pack))

        # 批量 NN value 评估叶节点
        nn_win_contributions = [0.0] * len(sim_results)
        if leaf_hands and self.nn is not None:
            obs_batch = StateEncoder.encode_batch_for_value(
                list(zip(leaf_hands, leaf_packs)), self
            )
            if obs_batch.shape[0] > 0:
                # 分批推理避免内存问题
                values = []
                for start in range(0, obs_batch.shape[0], NN_VALUE_BATCH_SIZE):
                    end = min(start + NN_VALUE_BATCH_SIZE, obs_batch.shape[0])
                    batch_vals = self.nn.evaluate_value_only(obs_batch[start:end])
                    values.append(batch_vals)
                values = np.concatenate(values) if values else np.array([])

                # 将 value 映射到胜率贡献
                leaf_idx = 0
                for sim_idx, (won, reached_tenpai) in enumerate(sim_results):
                    if won:
                        continue  # 已确认胡牌，不需要NN补充
                    nn_wr = _sigmoid(values[leaf_idx])
                    leaf_idx += 1
                    if not reached_tenpai and nn_wr > 0.2:
                        # rollout 未听牌但 NN 认为有希望，加权计入
                        nn_win_contributions[sim_idx] = nn_wr * 0.3

        # 综合结果
        win_count = 0
        tenpai_count = 0
        for sim_idx, (won, reached_tenpai) in enumerate(sim_results):
            if won:
                win_count += 1
                tenpai_count += 1
            elif reached_tenpai:
                tenpai_count += 1
            win_count += nn_win_contributions[sim_idx]

        n_done = len(sim_results)
        win_rate = win_count / n_done if n_done > 0 else 0.0
        tenpai_rate = tenpai_count / n_done if n_done > 0 else 0.0
        return win_rate, tenpai_rate

    # 4: Progressive Pruning + FlatMC 综合评估

    def _flatmc_with_pruning(self, candidates, total_budget, max_rounds, use_vhr=True):
        """
        Progressive Pruning + FlatMC/VHR。

        Args:
            candidates: [(tile, shanten_after), ...]
            total_budget: 总模拟次数
            max_rounds: 每次模拟最大轮数
            use_vhr: 是否使用 VHR (NN value head 评估叶节点)

        返回: dict(tile -> (win_rate, tenpai_rate, sim_count))
        """
        if not candidates:
            return {}

        pool = self._remaining_pool()
        stats = {}
        for tile, _ in candidates:
            stats[tile] = {'win': 0, 'tenpai': 0, 'count': 0}

        actions = [tile for tile, _ in candidates]
        remaining_budget = total_budget

        sim_func = self._simulate_win_rate_vhr if (use_vhr and self.nn is not None) else self._simulate_win_rate

        initial_sims = min(SIM_INITIAL_PER_ACTION, remaining_budget // len(actions))
        if initial_sims < 1:
            initial_sims = 1

        for tile in actions:
            tmp_hand = list(self.hand)
            tmp_hand.remove(tile)
            wr, tr = sim_func(tmp_hand, self.packs[0], initial_sims, max_rounds, pool)
            stats[tile]['win'] += wr * initial_sims
            stats[tile]['tenpai'] += tr * initial_sims
            stats[tile]['count'] += initial_sims
            remaining_budget -= initial_sims

        while remaining_budget > 0 and len(actions) > 1:
            if not self._time_left(margin=0.15):
                break
            means = {}
            stds = {}
            for tile in actions:
                cnt = stats[tile]['count']
                win_sum = stats[tile]['win']
                p = win_sum / cnt if cnt > 0 else 0.0
                means[tile] = p
                stds[tile] = math.sqrt(p * (1 - p) / cnt) if cnt > 0 else 1.0

            best_tile = max(actions, key=lambda t: means[t])
            best_mean = means[best_tile]

            new_actions = []
            for tile in actions:
                upper = means[tile] + PP_RD * stds[tile]
                lower_best = best_mean - PP_RD * stds[best_tile]
                if upper < lower_best and tile != best_tile:
                    pass
                else:
                    new_actions.append(tile)

            actions = new_actions
            if len(actions) <= 1:
                break

            batch = min(SIM_BATCH_SIZE * len(actions), remaining_budget)
            per_action = batch // len(actions)
            if per_action < 1:
                per_action = 1

            for tile in actions:
                if remaining_budget < per_action:
                    break
                tmp_hand = list(self.hand)
                tmp_hand.remove(tile)
                wr, tr = sim_func(tmp_hand, self.packs[0], per_action, max_rounds, pool)
                stats[tile]['win'] += wr * per_action
                stats[tile]['tenpai'] += tr * per_action
                stats[tile]['count'] += per_action
                remaining_budget -= per_action

        results = {}
        for tile, s in stats.items():
            cnt = s['count']
            results[tile] = (
                s['win'] / cnt if cnt > 0 else 0.0,
                s['tenpai'] / cnt if cnt > 0 else 0.0,
                cnt
            )
        return results

    # 5: Discard-Twice Method (H1 + H2)

    def _discard_twice_eval(self, tile, pack, sim_count_per_path, max_rounds):
        """对给定弃牌 tile，模拟"弃 tile → 摸牌 → 再弃牌"的双层路径期望胜率。"""
        hand_after = list(self.hand)
        hand_after.remove(tile)
        second_candidates = list(set(hand_after))
        if not second_candidates:
            pool = self._remaining_pool()
            wr, _ = self._simulate_win_rate(hand_after, pack, sim_count_per_path, max_rounds, pool)
            return wr

        pool = self._remaining_pool()
        best_second_wr = 0.0
        for t2 in second_candidates:
            if not self._time_left(margin=0.15):
                break
            hand2 = list(hand_after)
            hand2.remove(t2)
            wr, _ = self._simulate_win_rate(hand2, pack, sim_count_per_path, max_rounds, pool)
            if wr > best_second_wr:
                best_second_wr = wr
        return best_second_wr

    def _apply_discard_twice(self, candidates, flatmc_results, base_shanten):
        """应用 Discard-Twice H1 + H2。返回最终 tile。"""
        if len(candidates) == 1:
            return candidates[0][0]

        tb_set, ntb_set = classify_tatsu_breaking(self.hand, self.packs[0], base_shanten)

        def get_wr(t):
            return flatmc_results.get(t, (0, 0, 0))[0]

        if ntb_set:
            ntb_wrs = [(t, get_wr(t)) for t in ntb_set if t in [c[0] for c in candidates]]
            if ntb_wrs:
                best_ntb_tile, best_ntb_wr = max(ntb_wrs, key=lambda x: x[1])
                best_ntb_cnt = flatmc_results.get(best_ntb_tile, (0, 0, 0))[2]
                best_ntb_sigma = math.sqrt(best_ntb_wr * (1 - best_ntb_wr) / max(1, best_ntb_cnt))
                lower_best = best_ntb_wr - PP_RD * best_ntb_sigma

                good_ntb = []
                for t, wr in ntb_wrs:
                    cnt = flatmc_results.get(t, (0, 0, 0))[2]
                    if cnt > 0:
                        sigma = math.sqrt(wr * (1 - wr) / cnt)
                        if wr + PP_RD * sigma >= lower_best:
                            good_ntb.append(t)
                    else:
                        good_ntb.append(t)
                if not good_ntb:
                    good_ntb = [t for t, _ in ntb_wrs]

                h2_candidates = []
                for t1 in good_ntb:
                    hand1 = list(self.hand)
                    hand1.remove(t1)
                    try:
                        s1 = MahjongShanten(hand=tuple(hand1), pack=tuple(self.packs[0]))
                    except Exception:
                        s1 = 999
                    tb1, _ = classify_tatsu_breaking(hand1, self.packs[0], s1)
                    if tb1:
                        h2_candidates.append(t1)

                if h2_candidates:
                    return max(h2_candidates, key=get_wr)

        top3 = sorted(candidates, key=lambda c: get_wr(c[0]), reverse=True)[:3]
        best_h1_tile = top3[0][0]
        best_h1_score = get_wr(best_h1_tile)

        sim_per_path = max(20, SIM_TOTAL_BUDGET // (len(top3) * 12))
        for tile, _ in top3:
            if not self._time_left(margin=0.15):
                break
            two_ply_wr = self._discard_twice_eval(tile, self.packs[0], sim_per_path,
                                                   max(3, SIM_MAX_ROUNDS // 2))
            combined = get_wr(tile) * 0.4 + two_ply_wr * 0.6
            if combined > best_h1_score:
                best_h1_score = combined
                best_h1_tile = tile

        tb_in_candidates = [t for t in tb_set if t in [c[0] for c in candidates]]
        ntb_in_candidates = [t for t in ntb_set if t in [c[0] for c in candidates]]

        if tb_in_candidates and ntb_in_candidates:
            best_tb_wr = max(get_wr(t) for t in tb_in_candidates)
            best_ntb_wr = max(get_wr(t) for t in ntb_in_candidates)
            if best_tb_wr >= best_ntb_wr:
                return max(tb_in_candidates, key=get_wr)

        return best_h1_tile

    # 6: 安全牌判断

    def _safety_score(self, tile):
        score = 0
        for i in range(1, 4):
            if tile in self.oppDiscards[i]:
                score += 1
        if tile[0] in 'FJ':
            shown = self.shownTiles.get(tile, 0)
            if shown >= 2:
                score += 1
        return score

    # 7: NN 候选排序

    def _nn_rank_candidates(self, candidates):
        """
        用 NN policy 给候选弃牌排序，结合 shanten 和 NN 概率。

        Returns:
            [(tile, shanten, combined_score), ...] 按 combined_score 降序
        """
        nn_eval = self._get_nn_eval()
        if nn_eval is None:
            # 无 NN，退回纯 shanten 排序
            return [(c['tile'], c['shanten'], -c['shanten']) for c in candidates]

        probs, value, obs, mask = nn_eval

        # 提取手牌中每张牌的 Play 动作概率
        tile_probs = {}
        for tile in set(self.hand):
            idx = OFFSET_ACT['Play'] + OFFSET_TILE[tile]
            tile_probs[tile] = float(probs[idx])

        # 归一化概率（只看候选牌的概率分布）
        total_prob = sum(tile_probs.get(c['tile'], 0) for c in candidates)
        if total_prob < 1e-8:
            total_prob = 1.0

        alpha = NN_ALPHA
        scored = []
        for c in candidates:
            t = c['tile']
            nn_score = tile_probs.get(t, 0.0) / total_prob
            shanten_score = 1.0 / (1.0 + c['shanten'])
            combined = alpha * nn_score + (1 - alpha) * shanten_score
            scored.append((t, c['shanten'], combined))
        scored.sort(key=lambda x: -x[2])
        return scored

    # 核心: NN增强版 弃牌选择

    def best_discard(self):
        """决策层次: 听牌→待牌最大化; 1-2向听→NN+FlatMC+PP+Discard-Twice; ≥3→NN+受入枚数"""
        self._start_time = time.time()
        self._invalidate_nn_cache()
        h = self.hand
        pack = list(self.packs[0])

        candidates = []
        saved = list(h)
        for t in set(h):
            saved.remove(t)
            try:
                s = MahjongShanten(hand=tuple(saved), pack=tuple(pack))
            except Exception:
                s = 999
            saved.append(t)
            candidates.append({'tile': t, 'shanten': s})

        candidates.sort(key=lambda x: x['shanten'])
        best_s = candidates[0]['shanten']

        # base_s: 弃牌前向听数
        # MahjongShanten 不接受14张手牌 (3n+2)，此时 base_s 等于弃牌后最优向听
        base_s = self.shanten()
        if base_s == 999 and len(h) % 3 == 2:
            base_s = best_s

        top = [c for c in candidates if c['shanten'] == best_s]

        if best_s <= 0:
            return self._best_discard_tenpai(top)

        if best_s <= 2:
            return self._best_discard_simcat(top, base_s)

        return self._best_discard_early(top)

    def _best_discard_tenpai(self, candidates):
        """听牌时: 最大化待牌，同分用启发式 tie-break"""
        h = self.hand
        pack = list(self.packs[0])
        best_tile = None
        best_wait_types = -1
        best_wait_copies = -1

        for c in candidates:
            t = c['tile']
            saved = list(h)
            saved.remove(t)
            w_types, w_copies = self._wait_count(saved, pack)
            if (w_types > best_wait_types or
                    (w_types == best_wait_types and w_copies > best_wait_copies)):
                best_wait_types = w_types
                best_wait_copies = w_copies
                best_tile = t

        ties = []
        for c in candidates:
            t = c['tile']
            saved = list(h)
            saved.remove(t)
            w_types, _ = self._wait_count(saved, pack)
            if w_types == best_wait_types:
                ties.append(t)

        for t in ties:
            if t[0] in 'FJ' and h.count(t) == 1:
                return t
        for t in ties:
            if t[1] in '19' and h.count(t) == 1:
                return t
        return ties[0] if ties else best_tile

    def _best_discard_simcat(self, candidates, base_shanten):
        """低向听: NN候选排序 + FlatMC+PP(VHR) + Discard-Twice"""
        # Step 1: NN 候选排序
        ranked = self._nn_rank_candidates(candidates)
        n_eval = min(len(ranked), TOP_K_CANDIDATES)
        to_eval = ranked[:n_eval]

        # Step 2: 检查 NN 置信度 — 如果 NN 非常确信，可以减少 FlatMC 预算
        nn_confident = False
        if len(to_eval) >= 2 and to_eval[0][2] > NN_CONFIDENCE_SKIP:
            nn_confident = True

        remaining_draws = sum(self.tileWall)
        draw_factor = min(1.0, remaining_draws / 40.0)
        budget = int(SIM_TOTAL_BUDGET * draw_factor)
        if nn_confident:
            budget = max(600, budget // 2)  # NN 高置信度时减半预算
        else:
            budget = max(budget, 600)
        max_rounds = min(SIM_MAX_ROUNDS, max(3, remaining_draws // 4))

        if remaining_draws < 8 or budget < 600:
            return self._best_discard_early([{'tile': t, 'shanten': s}
                                             for t, s, _ in to_eval])

        # Step 3: FlatMC+PP (with VHR)
        flatmc_results = self._flatmc_with_pruning(
            [(t, s) for t, s, _ in to_eval],
            budget, max_rounds, use_vhr=True
        )

        # Step 4: NN value 融合
        # 对 FlatMC 结果用 NN value 进行修正
        if self.nn is not None:
            nn_eval = self._get_nn_eval()
            if nn_eval is not None:
                _, base_value, _, _ = nn_eval
                base_wr = _sigmoid(base_value)

                # 对每个候选：编码弃牌后状态 → NN value → 融合
                for tile in flatmc_results:
                    tmp_hand = list(self.hand)
                    tmp_hand.remove(tile)
                    obs_after = StateEncoder.encode_for_value_only(
                        tmp_hand, self.packs, self.shownTiles, self.tileWall,
                        self.seatWind, self.prevalentWind, self.oppDiscards
                    )
                    nn_val = self.nn.evaluate_value_only(obs_after[np.newaxis])[0]
                    nn_wr = _sigmoid(nn_val)

                    wr, tr, cnt = flatmc_results[tile]
                    # 自适应混合: 模拟次数越多, FlatMC越可靠
                    beta = min(NN_BETA_MAX,
                              NN_BETA_BASE + 0.5 * cnt / max(1, SIM_TOTAL_BUDGET))
                    fused_wr = beta * wr + (1 - beta) * nn_wr
                    flatmc_results[tile] = (fused_wr, tr, cnt)

        # Step 5: Discard-Twice
        best_tile = self._apply_discard_twice(
            [(t, s) for t, s, _ in to_eval],
            flatmc_results, base_shanten
        )
        return best_tile

    def _best_discard_early(self, candidates):
        """高向听 (>=3): NN policy + 受入枚数 + 安全牌"""
        h = self.hand
        pack = list(self.packs[0])
        best_s = candidates[0]['shanten']
        top = candidates[:min(len(candidates), TOP_K_CANDIDATES)]

        def acceptance(hand, pack):
            current_s = self.shanten(hand, pack)
            if current_s <= 0:
                return 0, 0
            pool_counter = Counter(self._remaining_pool())
            for t in hand:
                pool_counter[t] = pool_counter.get(t, 0) + 1
            useful_types = 0
            useful_copies = 0
            saved = list(hand)
            for tile in TILE_TYPES:
                copies = pool_counter.get(tile, 0)
                if copies == 0:
                    continue
                saved.append(tile)
                try:
                    s = MahjongShanten(hand=tuple(saved), pack=tuple(pack))
                except Exception:
                    s = 999
                saved.pop()
                if s < current_s:
                    useful_types += 1
                    useful_copies += copies
            return useful_types, useful_copies

        # NN policy 辅助排序
        nn_probs = {}
        nn_eval = self._get_nn_eval()
        if nn_eval is not None:
            probs, _, _, _ = nn_eval
            for tile in set(h):
                idx = OFFSET_ACT['Play'] + OFFSET_TILE[tile]
                nn_probs[tile] = float(probs[idx])

        scored = []
        for c in top:
            t = c['tile']
            saved_h = list(h)
            saved_h.remove(t)
            acc_types, acc_copies = acceptance(saved_h, pack)
            safety = self._safety_score(t)
            nn_score = nn_probs.get(t, 0.0) * 10
            score = -acc_types * 10 - acc_copies * 0.5 - safety * 2 + nn_score
            scored.append({'tile': t, 'score': score, 'acc_types': acc_types,
                           'acc_copies': acc_copies, 'safety': safety})

        scored.sort(key=lambda x: x['score'])

        if len(scored) >= 2 and scored[0]['acc_types'] == scored[1]['acc_types']:
            for s in scored:
                t = s['tile']
                if t[0] in 'FJ' and h.count(t) == 1:
                    return t
            for s in scored:
                t = s['tile']
                if t[1] in '19' and h.count(t) == 1:
                    return t

        return scored[0]['tile']

    # NN-guided 鸣牌决策

    def should_peng(self, tile, rp):
        """
        判断是否应该碰牌（NN value 比较 + shanten 保底）

        Args:
            tile: 被碰的牌
            rp: 相对位置

        Returns:
            (ok: bool, discard_tile: str or None) — 对齐 tournament.py HumanAdapter 接口
        """
        if self.hand.count(tile) < 2:
            return False, None

        s0 = self.shanten()
        sh, sp = list(self.hand), list(self.packs[0])
        for _ in range(2):
            self.hand.remove(tile)
        self.packs[0].append(('PENG', tile, (4 + rp - self.tileFrom) % 4))
        s1 = self.shanten()

        # shanten 不降低 → 直接否决
        if s1 >= s0:
            self.hand, self.packs[0] = sh, sp
            return False, None

        # NN value 评估（如果可用）
        if self.nn is not None:
            # 碰牌前
            obs_before, mask_before = StateEncoder.encode(self)
            _, value_before = self.nn.evaluate(obs_before, mask_before)

            # 碰牌后（模拟状态）
            self._invalidate_nn_cache()
            obs_after, mask_after = StateEncoder.encode(self)
            _, value_after = self.nn.evaluate(obs_after, mask_after)
            self._invalidate_nn_cache()

            # NN 认为碰牌后局面更差 → 否决（即使 shanten 降低了）
            if value_after < value_before - NN_MELD_THRESHOLD:
                self.hand, self.packs[0] = sh, sp
                return False, None

        # 碰牌通过 → 选最优弃牌
        best = self.best_discard()
        self.hand, self.packs[0] = sh, sp
        return True, best

    def should_chi(self, best_chi, curTile, best_s=None):
        """
        判断是否应该吃牌（NN value 比较 + shanten 保底）

        Args:
            best_chi: 吃牌中心牌
            curTile: 当前出的牌
            best_s: 吃后向听数（可选，兼容两种调用方式）

        Returns:
            (ok: bool, discard_tile: str or None) — 对齐 tournament.py HumanAdapter 接口
        """
        # 计算吃前向听
        s0 = self.shanten()

        # 如果提供了 best_s，使用它；否则计算吃后向听
        if best_s is not None and best_s >= s0:
            return False, None

        # NN value 评估
        if self.nn is not None:
            obs_before, mask_before = StateEncoder.encode(self)
            _, value_before = self.nn.evaluate(obs_before, mask_before)

            # 模拟吃牌后
            saved_hand, saved_packs = list(self.hand), list(self.packs[0])
            c = best_chi[0]
            n = int(best_chi[1])
            self.hand.append(curTile)
            for dn in (-1, 0, 1):
                card = c + str(n + dn)
                if card in self.hand and card != curTile:
                    self.hand.remove(card)
            self.packs[0].append(('CHI', best_chi, int(curTile[1]) - n + 2))
            self._invalidate_nn_cache()
            obs_after, mask_after = StateEncoder.encode(self)
            _, value_after = self.nn.evaluate(obs_after, mask_after)

            if value_after < value_before - NN_MELD_THRESHOLD:
                self.hand, self.packs[0] = saved_hand, saved_packs
                self._invalidate_nn_cache()
                return False, None

            # 吃牌通过 → 选最优弃牌
            best = self.best_discard()
            self.hand, self.packs[0] = saved_hand, saved_packs
            self._invalidate_nn_cache()
            return True, best

        # 无 NN 时仅靠 shanten 判断
        if best_s is not None and best_s < s0:
            # 计算吃后最优弃牌
            saved_hand, saved_packs = list(self.hand), list(self.packs[0])
            c = best_chi[0]
            n = int(best_chi[1])
            self.hand.append(curTile)
            for dn in (-1, 0, 1):
                card = c + str(n + dn)
                if card in self.hand and card != curTile:
                    self.hand.remove(card)
            self.packs[0].append(('CHI', best_chi, int(curTile[1]) - n + 2))
            best = self.best_discard()
            self.hand, self.packs[0] = saved_hand, saved_packs
            return True, best

        return False, None

    def should_gang(self, tile, rp):
        """判断是否应该明杠"""
        s0 = self.shanten()
        sh, sp = list(self.hand), list(self.packs[0])
        for _ in range(3):
            self.hand.remove(tile)
        self.packs[0].append(('GANG', tile, (4 + rp - self.tileFrom) % 4))
        s1 = self.shanten()
        self.hand, self.packs[0] = sh, sp
        return s1 < s0

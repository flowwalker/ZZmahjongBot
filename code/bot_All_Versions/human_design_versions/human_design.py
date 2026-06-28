"""
一个纯手工的复杂规则编码版本
用于看看人类规则的效果如何
实测是连baseline版本的sl都难以打平
"""
import sys
import time
import random
import math
from collections import defaultdict, Counter
from MahjongGB import MahjongShanten, MahjongFanCalculator



TILE_TYPES = [
    *('W%d' % (i + 1) for i in range(9)),
    *('T%d' % (i + 1) for i in range(9)),
    *('B%d' % (i + 1) for i in range(9)),
    *('F%d' % (i + 1) for i in range(4)),
    *('J%d' % (i + 1) for i in range(3)),
]
TOTAL_COPIES = {t: 4 for t in TILE_TYPES}

SIM_TOTAL_BUDGET = 3800       
SIM_INITIAL_PER_ACTION = 60   
SIM_BATCH_SIZE = 40           
PP_RD = 2.0                   
SIM_MAX_ROUNDS = 8            
TOP_K_CANDIDATES = 5          

TIME_LIMIT = 4.5              
TIME_CHECK_INTERVAL = 25      


# 辅助函数

def classify_tatsu_breaking(hand, pack, base_shanten):
    tb_set, ntb_set = [], []
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
    if not hand:
        return None, 999
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


# 状态机与算法引擎

class GameState:
    def __init__(self, seatWind):
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

    def _time_left(self, margin=0.10):
        if self._start_time == 0.0:
            return True
        return time.perf_counter() - self._start_time < TIME_LIMIT - margin

    def _remaining_pool(self):
        count = dict(TOTAL_COPIES)
        for t in self.hand:
            if count[t] > 0: count[t] -= 1
        for i in range(4):
            for pk, pt, po in self.packs[i]:
                if pk == 'CHI':
                    c, n = pt[0], int(pt[1])
                    for dn in (-1, 0, 1):
                        tn = c + str(n + dn)
                        if count[tn] > 0: count[tn] -= 1
                elif pk == 'PENG':
                    count[pt] = max(0, count[pt] - 3)
                elif pk == 'GANG':
                    if pt != 'CONCEALED':
                        count[pt] = max(0, count[pt] - 4)
        for i in range(4):
            for t in self.oppDiscards[i]:
                if count[t] > 0: count[t] -= 1
        pool = []
        for t, c in count.items():
            if c > 0:
                pool.extend([t] * c)
        return pool

    def _wait_count(self, hand, pack):
        pool_counter = Counter(self._remaining_pool())
        wait_types, wait_copies = 0, 0
        saved_hand = list(hand)
        saved_pack = tuple(pack)
        for tile in TILE_TYPES:
            copies = pool_counter.get(tile, 0)
            if copies == 0:
                continue
            saved_hand.append(tile)
            try:
                fans = MahjongFanCalculator(
                    pack=saved_pack, hand=tuple(saved_hand),
                    winTile=tile, flowerCount=0, isSelfDrawn=False,
                    is4thTile=(self.shownTiles[tile]) == 4,
                    isAboutKong=False, isWallLast=self.wallLast,
                    seatWind=self.seatWind, prevalentWind=self.prevalentWind,
                    verbose=False)
                if sum(p * c for p, c, _, _ in fans) >= 8:
                    wait_types += 1
                    wait_copies += copies
            except Exception:
                pass
            saved_hand.pop()
        return wait_types, wait_copies

    def _calculate_risk_penalty(self, tile):
        if sum(self.tileWall) > 45:
            return 1.0
        
        seen_count = self.shownTiles.get(tile, 0)
        in_opp_discards = sum(1 for i in range(1, 4) if tile in self.oppDiscards[i])
        
        if in_opp_discards > 0:
            return 1.0 
        elif seen_count == 3:
            return 0.98 
        elif seen_count == 0:
            return 0.85 
        return 0.90 

    def _simulate_win_rate(self, hand, pack, n_sims, max_draws, pool=None):
        if pool is None:
            pool = self._remaining_pool()
        if len(pool) < max_draws * 4:
            max_draws = max(1, len(pool) // 4)

        win_count, tenpai_count = 0, 0
        pack_tuple = tuple(pack)

        for i in range(n_sims):
            if i > 0 and i % TIME_CHECK_INTERVAL == 0:
                if not self._time_left(margin=0.08):
                    n_done = i
                    return (win_count / n_done if n_done > 0 else 0.0,
                            tenpai_count / n_done if n_done > 0 else 0.0)
            
            sim_pool = pool[:]
            random.shuffle(sim_pool)
            sim_hand = hand[:]
            reached_tenpai = False

            for draw_idx in range(max_draws):
                if len(sim_pool) < 4:
                    break
                
                del sim_pool[-3:]
                draw_tile = sim_pool.pop()
                sim_hand.append(draw_tile)

                try:
                    s_quick = MahjongShanten(hand=tuple(sim_hand), pack=pack_tuple)
                    if s_quick <= 0:
                        reached_tenpai = True
                        fans = MahjongFanCalculator(
                            pack=pack_tuple, hand=tuple(sim_hand),
                            winTile=draw_tile, flowerCount=0, isSelfDrawn=True,
                            is4thTile=self.shownTiles[draw_tile] == 4,
                            isAboutKong=False, isWallLast=False,
                            seatWind=self.seatWind, prevalentWind=self.prevalentWind,
                            verbose=False)
                        if sum(p * c for p, c, _, _ in fans) >= 8:
                            win_count += 1
                            tenpai_count += 1
                            break
                except Exception:
                    pass

                best_t, best_s = greedy_discard(sim_hand, list(pack_tuple))
                if best_t:
                    sim_hand.remove(best_t)

            else:
                if not reached_tenpai:
                    try:
                        if MahjongShanten(hand=tuple(sim_hand), pack=pack_tuple) <= 0:
                            tenpai_count += 1
                    except Exception:
                        pass

        win_rate = win_count / n_sims if n_sims > 0 else 0.0
        tenpai_rate = tenpai_count / n_sims if n_sims > 0 else 0.0
        return win_rate, tenpai_rate

    def _flatmc_with_pruning(self, candidates, total_budget, max_rounds):
        if not candidates: return {}
        pool = self._remaining_pool()
        stats = {tile: {'win': 0, 'tenpai': 0, 'count': 0} for tile, _ in candidates}
        actions = [tile for tile, _ in candidates]
        remaining_budget = total_budget

        initial_sims = min(SIM_INITIAL_PER_ACTION, max(1, remaining_budget // len(actions)))

        for tile in actions:
            tmp_hand = list(self.hand)
            tmp_hand.remove(tile)
            wr, tr = self._simulate_win_rate(tmp_hand, self.packs[0], initial_sims, max_rounds, pool)
            
            penalty = self._calculate_risk_penalty(tile)
            wr *= penalty
            
            stats[tile]['win'] += wr * initial_sims
            stats[tile]['tenpai'] += tr * initial_sims
            stats[tile]['count'] += initial_sims
            remaining_budget -= initial_sims

        while remaining_budget > 0 and len(actions) > 1:
            if not self._time_left(margin=0.08): break
            means = {}
            stds = {}
            for tile in actions:
                cnt = stats[tile]['count']
                p = stats[tile]['win'] / cnt if cnt > 0 else 0.0
                means[tile] = p
                stds[tile] = math.sqrt(p * (1 - p) / cnt) if cnt > 0 else 1.0

            best_tile = max(actions, key=lambda t: means[t])
            best_mean = means[best_tile]

            new_actions = []
            for tile in actions:
                upper = means[tile] + PP_RD * stds[tile]
                lower_best = best_mean - PP_RD * stds[best_tile]
                if upper >= lower_best or tile == best_tile:
                    new_actions.append(tile)

            actions = new_actions
            if len(actions) <= 1: break

            batch = min(SIM_BATCH_SIZE * len(actions), remaining_budget)
            per_action = max(1, batch // len(actions))

            for tile in actions:
                if remaining_budget < per_action: break
                tmp_hand = list(self.hand)
                tmp_hand.remove(tile)
                wr, tr = self._simulate_win_rate(tmp_hand, self.packs[0], per_action, max_rounds, pool)
                
                penalty = self._calculate_risk_penalty(tile)
                wr *= penalty

                stats[tile]['win'] += wr * per_action
                stats[tile]['tenpai'] += tr * per_action
                stats[tile]['count'] += per_action
                remaining_budget -= per_action

        return {tile: (s['win']/s['count'] if s['count']>0 else 0.0, 
                       s['tenpai']/s['count'] if s['count']>0 else 0.0, s['count']) 
                for tile, s in stats.items()}

    def _discard_twice_eval(self, tile, pack, sim_count_per_path, max_rounds):
        hand_after = list(self.hand)
        hand_after.remove(tile)
        second_candidates = list(set(hand_after))
        if not second_candidates:
            pool = self._remaining_pool()
            wr, _ = self._simulate_win_rate(hand_after, pack, sim_count_per_path, max_rounds, pool)
            return wr * self._calculate_risk_penalty(tile)

        pool = self._remaining_pool()
        best_second_wr = 0.0
        for t2 in second_candidates:
            if not self._time_left(margin=0.08): break
            hand2 = list(hand_after)
            hand2.remove(t2)
            wr, _ = self._simulate_win_rate(hand2, pack, sim_count_per_path, max_rounds, pool)
            penalty = self._calculate_risk_penalty(t2)
            wr *= penalty
            if wr > best_second_wr: best_second_wr = wr
            
        return best_second_wr * self._calculate_risk_penalty(tile)

    def _apply_discard_twice(self, candidates, flatmc_results, base_shanten):
        if len(candidates) == 1: return candidates[0][0]
        tb_set, ntb_set = classify_tatsu_breaking(self.hand, self.packs[0], base_shanten)
        def get_wr(t): return flatmc_results.get(t, (0, 0, 0))[0]

        if ntb_set:
            ntb_wrs = [(t, get_wr(t)) for t in ntb_set if t in [c[0] for c in candidates]]
            if ntb_wrs:
                best_ntb_tile, best_ntb_wr = max(ntb_wrs, key=lambda x: x[1])
                best_ntb_cnt = flatmc_results.get(best_ntb_tile, (0, 0, 0))[2]
                best_ntb_sigma = math.sqrt(best_ntb_wr * (1 - best_ntb_wr) / max(1, best_ntb_cnt))
                lower_best = best_ntb_wr - PP_RD * best_ntb_sigma

                good_ntb = [t for t, wr in ntb_wrs if wr + PP_RD * math.sqrt(wr * (1 - wr) / max(1, flatmc_results.get(t, (0,0,0))[2])) >= lower_best]
                if not good_ntb: good_ntb = [t for t, _ in ntb_wrs]

                h2_candidates = []
                for t1 in good_ntb:
                    hand1 = list(self.hand)
                    hand1.remove(t1)
                    try: s1 = MahjongShanten(hand=tuple(hand1), pack=tuple(self.packs[0]))
                    except Exception: s1 = 999
                    tb1, _ = classify_tatsu_breaking(hand1, self.packs[0], s1)
                    if tb1: h2_candidates.append(t1)

                if h2_candidates: return max(h2_candidates, key=get_wr)

        top3 = sorted(candidates, key=lambda c: get_wr(c[0]), reverse=True)[:3]
        best_h1_tile = top3[0][0]
        best_h1_score = get_wr(best_h1_tile)
        sim_per_path = max(20, SIM_TOTAL_BUDGET // (len(top3) * 12))
        
        for tile, _ in top3:
            if not self._time_left(margin=0.08): break
            two_ply_wr = self._discard_twice_eval(tile, self.packs[0], sim_per_path, max(3, SIM_MAX_ROUNDS // 2))
            combined = get_wr(tile) * 0.4 + two_ply_wr * 0.6
            if combined > best_h1_score:
                best_h1_score = combined
                best_h1_tile = tile

        tb_in_candidates = [t for t in tb_set if t in [c[0] for c in candidates]]
        ntb_in_candidates = [t for t in ntb_set if t in [c[0] for c in candidates]]
        if tb_in_candidates and ntb_in_candidates:
            if max(get_wr(t) for t in tb_in_candidates) >= max(get_wr(t) for t in ntb_in_candidates):
                return max(tb_in_candidates, key=get_wr)
        return best_h1_tile

    def _safety_score(self, tile):
        score = sum(1 for i in range(1, 4) if tile in self.oppDiscards[i])
        if tile[0] in 'FJ' and self.shownTiles.get(tile, 0) >= 2: score += 1
        return score

    def best_discard(self):
        self._start_time = time.perf_counter()
        h, pack = self.hand, list(self.packs[0])
        base_s = self.shanten()
        
        candidates = []
        for t in set(h):
            saved = list(h)
            saved.remove(t)
            candidates.append({'tile': t, 'shanten': self.shanten(saved, pack)})
            
        candidates.sort(key=lambda x: x['shanten'])
        best_s = candidates[0]['shanten']
        top = [c for c in candidates if c['shanten'] == best_s]

        if best_s <= 0: return self._best_discard_tenpai(top)
        if best_s <= 2: return self._best_discard_simcat(top, base_s)
        return self._best_discard_early(top)

    def _best_discard_tenpai(self, candidates):
        h, pack = self.hand, list(self.packs[0])
        best_tile, best_wt, best_wc = None, -1, -1

        for c in candidates:
            t = c['tile']
            saved = list(h)
            saved.remove(t)
            wt, wc = self._wait_count(saved, pack)
            if wt > best_wt or (wt == best_wt and wc > best_wc):
                best_wt, best_wc, best_tile = wt, wc, t

        ties = [c['tile'] for c in candidates if self._wait_count([x for x in h if x != c['tile']], pack)[0] == best_wt]
        for t in ties:
            if t[0] in 'FJ' and h.count(t) == 1: return t
        for t in ties:
            if t[1] in '19' and h.count(t) == 1: return t
        return ties[0] if ties else best_tile

    def _best_discard_simcat(self, candidates, base_shanten):
        n_eval = min(len(candidates), TOP_K_CANDIDATES)
        to_eval = candidates[:n_eval]
        remaining_draws = sum(self.tileWall)
        draw_factor = min(1.0, remaining_draws / 40.0)
        budget = max(600, int(SIM_TOTAL_BUDGET * draw_factor))
        max_rounds = min(SIM_MAX_ROUNDS, max(3, remaining_draws // 4))

        if remaining_draws < 8 or budget < 600:
            return self._best_discard_early(to_eval)

        flatmc_results = self._flatmc_with_pruning([(c['tile'], c['shanten']) for c in to_eval], budget, max_rounds)
        return self._apply_discard_twice([(c['tile'], c['shanten']) for c in to_eval], flatmc_results, base_shanten)

    def _best_discard_early(self, candidates):
        h, pack = self.hand, list(self.packs[0])
        top = candidates[:min(len(candidates), TOP_K_CANDIDATES)]

        def acceptance(hand):
            cs = self.shanten(hand, pack)
            if cs <= 0: return 0, 0
            pool_counter = Counter(self._remaining_pool())
            ut, uc = 0, 0
            saved = list(hand)
            for tile in TILE_TYPES:
                copies = pool_counter.get(tile, 0)
                if copies == 0: continue
                saved.append(tile)
                if self.shanten(saved, pack) < cs:
                    ut += 1; uc += copies
                saved.pop()
            return ut, uc

        scored = []
        for c in top:
            t = c['tile']
            saved_h = list(h)
            saved_h.remove(t)
            acc_types, acc_copies = acceptance(saved_h)
            safety = self._safety_score(t)
            scored.append({'tile': t, 'score': -acc_types * 10 - acc_copies * 0.5 - safety * 2, 'acc_types': acc_types})

        scored.sort(key=lambda x: x['score'])
        if len(scored) >= 2 and scored[0]['acc_types'] == scored[1]['acc_types']:
            for s in scored:
                if s['tile'][0] in 'FJ' and h.count(s['tile']) == 1: return s['tile']
            for s in scored:
                if s['tile'][1] in '19' and h.count(s['tile']) == 1: return s['tile']
        return scored[0]['tile']

    def evaluate_meld(self, meld_hand, meld_pack):
        self._start_time = time.perf_counter()
        best_t, best_s = greedy_discard(meld_hand, meld_pack)
        if best_t is None: return False, None
        
        sim_hand = list(meld_hand)
        sim_hand.remove(best_t)
        
        wr_meld, _ = self._simulate_win_rate(sim_hand, meld_pack, n_sims=120, max_draws=4)
        wr_meld *= self._calculate_risk_penalty(best_t)
        
        wr_pass, _ = self._simulate_win_rate(self.hand, self.packs[0], n_sims=120, max_draws=4)
        
        return (wr_meld >= wr_pass), best_t

# ============================================================
# 接口主循环
# ============================================================

if __name__ == '__main__':
    gs = None
    seatWind = None
    angang = None
    zimo = False

    try:
        input()  
    except EOFError:
        sys.exit(0)
        
    while True:
        try:
            request = input()
            while not request.strip():
                request = input()
        except EOFError:
            break
        t = request.split()

        if t[0] == '0':
            seatWind = int(t[1])
            gs = GameState(seatWind)
            gs.prevalentWind = int(t[2])
            print('PASS')

        elif t[0] == '1':
            gs.hand = t[5:5 + 13]
            print('PASS')

        elif t[0] == '2':
            tile = t[1]
            gs.tileWall[0] -= 1
            gs.wallLast = gs.tileWall[1] == 0

            gs.hand.append(tile)
            if gs.can_hu(tile, isSelfDrawn=True, isAboutKong=gs.isAboutKong):
                print('HU')
                print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
                sys.stdout.flush()
                break

            gs.isAboutKong = False
            did_angang = False
            
            for t_card in set(gs.hand):
                if gs.hand.count(t_card) == 4 and not gs.wallLast and gs.tileWall[0] > 0:
                    s0 = gs.shanten()
                    saved_h, saved_p = list(gs.hand), list(gs.packs[0])
                    for _ in range(4): gs.hand.remove(t_card)
                    gs.packs[0].append(('GANG', t_card, 0))
                    s1 = gs.shanten()
                    gs.hand, gs.packs[0] = saved_h, saved_p
                    if s1 <= s0:
                        for _ in range(4): gs.hand.remove(t_card)
                        gs.packs[0].append(('GANG', t_card, 0))
                        gs.isAboutKong = True
                        angang = t_card
                        print('GANG %s' % t_card)
                        did_angang = True
                        break

            if did_angang:
                print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
                sys.stdout.flush()
                continue

            did_bugang = False
            if not gs.wallLast and gs.tileWall[0] > 0:
                for pk, pt, po in gs.packs[0]:
                    if pk == 'PENG' and pt in gs.hand:
                        gs.hand.remove(pt)
                        gs.isAboutKong = True
                        print('BUGANG %s' % pt)
                        did_bugang = True
                        break

            if did_bugang:
                print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
                sys.stdout.flush()
                continue

            best = gs.best_discard()
            gs.hand.remove(best)
            print('PLAY %s' % best)

        elif t[0] == '3':
            p = int(t[1])
            rp = (p + 4 - gs.seatWind) % 4

            if t[2] == 'DRAW':
                gs.tileWall[rp] -= 1
                zimo = True
                print('PASS')

            elif t[2] == 'GANG':
                if p == seatWind and angang:
                    gs.packs[0].append(('GANG', angang, 0))
                    angang = None
                elif zimo:
                    gs.packs[rp].append(('GANG', 'CONCEALED', 0))
                else:
                    gs.packs[rp].append(('GANG', gs.curTile, (4 + rp - gs.tileFrom) % 4))
                    gs.shownTiles[gs.curTile] += 3
                print('PASS')

            elif t[2] == 'BUGANG':
                tile = t[3]
                gs.shownTiles[tile] += 1
                for i, (pk, pt, po) in enumerate(gs.packs[rp]):
                    if pt == tile:
                        gs.packs[rp][i] = ('GANG', tile, po)
                        break
                if p == seatWind:
                    print('PASS')
                else:
                    if gs.can_hu(tile, isAboutKong=True):
                        print('HU')
                        print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
                        sys.stdout.flush()
                        break
                    print('PASS')

            else: 
                zimo = False
                if t[2] == 'CHI':
                    tile = t[3]
                    color, num = tile[0], int(tile[1])
                    gs.packs[rp].append(('CHI', tile, int(gs.curTile[1]) - num + 2))
                    gs.shownTiles[gs.curTile] -= 1
                    for i in range(-1, 2): gs.shownTiles[color + str(num + i)] += 1
                elif t[2] == 'PENG':
                    gs.packs[rp].append(('PENG', gs.curTile, (4 + rp - gs.tileFrom) % 4))
                    gs.shownTiles[gs.curTile] += 2

                gs.tileFrom = rp
                gs.curTile = t[-1]
                gs.shownTiles[gs.curTile] += 1
                gs.oppDiscards[rp].append(gs.curTile) 

                if p == seatWind:
                    print('PASS')
                else:
                    if gs.can_hu(gs.curTile):
                        print('HU')
                        print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
                        sys.stdout.flush()
                        break

                    if gs.wallLast:
                        print('PASS')
                        print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
                        sys.stdout.flush()
                        continue

                    if gs.hand.count(gs.curTile) >= 2:
                        s0 = gs.shanten()
                        sh, sp = list(gs.hand), list(gs.packs[0])
                        for _ in range(2): sh.remove(gs.curTile)
                        sp.append(('PENG', gs.curTile, (4 + rp - gs.tileFrom) % 4))
                        s1 = gs.shanten(sh, sp)
                        
                        if s1 < s0:
                            should_meld, best_discard_tile = gs.evaluate_meld(sh, sp)
                            if should_meld and best_discard_tile:
                                for _ in range(2): gs.hand.remove(gs.curTile)
                                gs.packs[0].append(('PENG', gs.curTile, (4 + rp - gs.tileFrom) % 4))
                                print('PENG %s' % best_discard_tile)
                                gs.hand.extend([gs.curTile, gs.curTile, best_discard_tile])
                                gs.packs[0].pop()
                                print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
                                sys.stdout.flush()
                                continue

                    if gs.hand.count(gs.curTile) == 3 and gs.tileWall[0] > 0:
                        s0 = gs.shanten()
                        sh, sp = list(gs.hand), list(gs.packs[0])
                        for _ in range(3): sh.remove(gs.curTile)
                        sp.append(('GANG', gs.curTile, (4 + rp - gs.tileFrom) % 4))
                        if gs.shanten(sh, sp) < s0:
                            print('GANG')
                            print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
                            sys.stdout.flush()
                            continue

                    if rp == 3 and gs.curTile[0] in 'WTB':
                        c, n = gs.curTile[0], int(gs.curTile[1])
                        best_chi, best_s, best_chi_discard = None, 999, None
                        
                        for mn in (n - 1, n, n + 1):
                            if mn < 1 or mn > 9: continue
                            mid = c + str(mn)
                            nd = [c + str(mn + 1), mid] if mn < n else ([c + str(mn - 1), mid] if mn > n else [c + str(n - 1), c + str(n + 1)])
                            
                            if not all(card in gs.hand for card in nd): continue
                            
                            sh, sp = list(gs.hand), list(gs.packs[0])
                            gs.hand.append(gs.curTile)
                            for card in nd: gs.hand.remove(card)
                            gs.packs[0].append(('CHI', mid, int(gs.curTile[1]) - mn + 2))
                            
                            s1 = gs.shanten()
                            if s1 < gs.shanten(sh, sp):
                                should_meld, d_tile = gs.evaluate_meld(gs.hand, gs.packs[0])
                                if should_meld and d_tile:
                                    if s1 < best_s:
                                        best_s, best_chi, best_chi_discard = s1, mid, d_tile
                            
                            gs.hand, gs.packs[0] = sh, sp
                            
                        if best_chi and best_chi_discard:
                            mid = best_chi
                            nd = [c + str(int(mid[1]) + 1), mid] if int(mid[1]) < n else ([c + str(int(mid[1]) - 1), mid] if int(mid[1]) > n else [c + str(n - 1), c + str(n + 1)])
                            gs.hand.append(gs.curTile)
                            for card in nd: gs.hand.remove(card)
                            gs.packs[0].append(('CHI', best_chi, int(gs.curTile[1]) - int(best_chi[1]) + 2))
                            
                            print('CHI %s %s' % (best_chi, best_chi_discard))
                            
                            gs.hand.remove(gs.curTile)
                            for card in nd: gs.hand.append(card)
                            gs.hand.append(best_chi_discard)
                            gs.packs[0].pop()
                            print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
                            sys.stdout.flush()
                            continue

                    print('PASS')

        print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
        sys.stdout.flush()
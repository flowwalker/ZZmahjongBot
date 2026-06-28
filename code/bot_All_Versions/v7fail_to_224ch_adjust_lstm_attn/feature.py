"""国标麻将特征提取器 — 224通道."""

from agent import MahjongGBAgent
from collections import defaultdict
import numpy as np

try:
    from MahjongGB import MahjongFanCalculator
except ImportError:
    print('MahjongGB library required!')
    raise


class FeatureAgent(MahjongGBAgent):
    """V3 国标麻将超级特征提取 Agent"""

    OBS_SIZE = 224
    ACT_SIZE = 235

    OFFSET_ACT = {
        'Pass': 0, 'Hu': 1, 'Play': 2, 'Chi': 36,
        'Peng': 99, 'Gang': 133, 'AnGang': 167, 'BuGang': 201
    }

    TILE_LIST = [
        *('W%d' % (i + 1) for i in range(9)),
        *('T%d' % (i + 1) for i in range(9)),
        *('B%d' % (i + 1) for i in range(9)),
        *('F%d' % (i + 1) for i in range(4)),
        *('J%d' % (i + 1) for i in range(3))
    ]

    OFFSET_TILE = {c: i for i, c in enumerate(TILE_LIST)}

    def __init__(self, seatWind: int):
        self.seatWind = seatWind
        self.packs = [[] for _ in range(4)]
        self.history = [[] for _ in range(4)]
        self.tileWall = [21] * 4
        self.shownTiles = defaultdict(int)
        self.wallLast = False
        self.isAboutKong = False
        self.prevalentWind = 0
        self.hand = []
        self.valid = []
        self.tileFrom = 0
        self.curTile = ''
        self.obs = np.zeros((self.OBS_SIZE, 36), dtype=np.float32)


    def _update_obs(self):
        self.obs.fill(0)
        self._update_base_features()      # Ch 0-147
        self._update_v2_features()        # Ch 148-191
        self._update_v3_features()        # Ch 192-223

    def _update_base_features(self):
        for tile in set(self.hand):
            cnt = self.hand.count(tile)
            idx = self.OFFSET_TILE[tile]
            for i in range(min(cnt, 4)):
                self.obs[0 + i, idx] = 1
        for tile in self.TILE_LIST:
            cnt = self.hand.count(tile) + self.shownTiles.get(tile, 0)
            cnt = min(cnt, 4)
            idx = self.OFFSET_TILE[tile]
            for i in range(cnt):
                self.obs[4 + i, idx] = 1
        for p in range(4):
            base_ch = 8 + p * 4
            for pkt, tile, offer in self.packs[p]:
                if pkt == 'CHI':
                    c, n = tile[0], int(tile[1])
                    for dn in (-1, 0, 1):
                        i2 = self.OFFSET_TILE.get(c + str(n + dn))
                        if i2 is not None:
                            self.obs[base_ch + 0, i2] = 1
                elif pkt == 'PENG':
                    idx = self.OFFSET_TILE.get(tile)
                    if idx is not None:
                        self.obs[base_ch + 1, idx] = 1
                elif pkt == 'GANG':
                    if offer == 0:
                        if tile != 'CONCEALED':
                            idx = self.OFFSET_TILE.get(tile)
                            if idx is not None:
                                self.obs[base_ch + 3, idx] = 1
                    else:
                        idx = self.OFFSET_TILE.get(tile)
                        if idx is not None:
                            self.obs[base_ch + 2, idx] = 1
        self.obs[24 + self.prevalentWind, :] = 1
        self.obs[28 + self.seatWind, :] = 1
        for p in range(4):
            self.obs[32 + p, :] = self.tileWall[p] / 21.0
        for p in range(4):
            for step, tile in enumerate(self.history[p][-28:]):
                idx = self.OFFSET_TILE.get(tile)
                if idx is not None:
                    self.obs[36 + p * 28 + step, idx] = 1


    def _update_v2_features(self):
        ha = self._hand_to_array()
        sa = self._shown_to_array()
        wr = {t: max(0, 4 - sa[self.OFFSET_TILE[t]]) for t in self.TILE_LIST}

        # HAND_COUNT (148-151)
        for tile in self.TILE_LIST:
            idx = self.OFFSET_TILE[tile]
            c = ha[idx]
            if c >= 1: self.obs[148 + 0, idx] = 1
            if c >= 2: self.obs[148 + 1, idx] = 1
            if c >= 3: self.obs[148 + 2, idx] = 1
            if c >= 4: self.obs[148 + 3, idx] = 1

        # EFFICIENCY (152-155)
        eff = self._calc_tile_efficiency(ha, wr)
        for i in range(34):
            self.obs[152, i] = eff[i]
            self.obs[153, i] = 1 if eff[i] > 0.3 else 0
            self.obs[154, i] = 1 if eff[i] > 0.6 else 0
            self.obs[155, i] = 1 if eff[i] > 0.9 else 0

        # SAFETY (156-159)
        safe = self._calc_tile_safety(wr)
        for i in range(34):
            self.obs[156, i] = safe[i]
            self.obs[157, i] = 1 if safe[i] > 0.5 else 0
            self.obs[158, i] = 1 if safe[i] > 0.8 else 0
            self.obs[159, i] = 1 if safe[i] > 0.95 else 0

        # OPP_RISK (160-163)
        risk = self._calc_opponent_risk()
        for i in range(34):
            self.obs[160, i] = risk[i]
            self.obs[161, i] = 1 if risk[i] > 0.5 else 0
            self.obs[162, i] = 1 if risk[i] > 0.8 else 0
            self.obs[163, i] = 1 if risk[i] > 0.95 else 0

        # YAKU_POTENTIAL (164-171)
        yf = self._calc_yaku_potential(ha)
        for j in range(8):
            for idx in range(34):
                self.obs[164 + j, idx] = yf[j][idx]

        # WALL_DENSITY (172-175)
        for tile in self.TILE_LIST:
            idx = self.OFFSET_TILE[tile]
            r = wr[tile]
            self.obs[172, idx] = r / 4.0
            self.obs[173, idx] = 1 if r <= 1 else 0
            self.obs[174, idx] = 1 if r <= 2 else 0
            self.obs[175, idx] = 1 if r == 0 else 0

        # POSITION (176-179)
        ttp = sum(len(h) for h in self.history)
        gp = min(ttp / 60.0, 1.0)
        self.obs[176, :] = gp
        self.obs[177, :] = self.tileWall[0] / 21.0
        self.obs[178, :] = 1 if self.tileWall[0] <= 5 else 0
        self.obs[179, :] = 1 if self.seatWind == 0 else 0

        # TEMPORAL (180-187)
        ds = self._calc_discard_stats()
        for j in range(8):
            for idx in range(34):
                self.obs[180 + j, idx] = ds[j][idx]

        # META (188-191)
        self.obs[188, :] = self._estimate_tenpai_chance(ha)
        self.obs[189, :] = sum(1 for c in ha if c > 0) / 13.0
        self.obs[190, :] = len(self.packs[0]) / 4.0
        self.obs[191, :] = 1.0 if any(p[0] == 'GANG' for p in self.packs[0]) else 0.0


    def _update_v3_features(self):
        ha = self._hand_to_array()
        sa = self._shown_to_array()
        wr = {t: max(0, 4 - sa[self.OFFSET_TILE[t]]) for t in self.TILE_LIST}

        # TILE_DRAW 进张贡献 (192-195)
        di = self._calc_draw_improvement(ha, wr)
        for i in range(34):
            self.obs[192, i] = di[i] / 20.0
            self.obs[193, i] = 1 if di[i] > 5 else 0
            self.obs[194, i] = 1 if di[i] > 10 else 0
            self.obs[195, i] = 1 if di[i] > 15 else 0

        # SHANTEN (196-199)
        si = self._calc_shanten_features(ha)
        for i in range(34):
            self.obs[196, i] = si['current'] / 8.0
            self.obs[197, i] = si['if_discard'][i] / 8.0
            self.obs[198, i] = 1 if si['current'] <= 1 else 0
            self.obs[199, i] = 1 if si['if_discard'][i] <= 1 else 0

        # WAIT_TYPE (200-203)
        wi = self._calc_wait_type(ha, wr)
        for i in range(34):
            self.obs[200, i] = wi['single'][i]
            self.obs[201, i] = wi['double_sided'][i]
            self.obs[202, i] = wi['pair_wait'][i]
            self.obs[203, i] = wi['edge'][i]

        # DANGER (204-207)
        dg = self._calc_precise_danger(ha, wr)
        for i in range(34):
            self.obs[204, i] = dg[i]
            self.obs[205, i] = 1 if dg[i] > 0.3 else 0
            self.obs[206, i] = 1 if dg[i] > 0.6 else 0
            self.obs[207, i] = 1 if dg[i] > 0.8 else 0

        # YAKU_PROG (208-211)
        yp = self._calc_yaku_progress(ha)
        for i in range(34):
            self.obs[208, i] = yp['pinfu'][i]
            self.obs[209, i] = yp['all_simples'][i]
            self.obs[210, i] = yp['half_flush'][i]
            self.obs[211, i] = yp['full_flush'][i]

        # HAND_STRUCT (212-215)
        st = self._calc_hand_structure(ha)
        self.obs[212, :] = st['pairs'] / 7.0
        self.obs[213, :] = st['taatsu'] / 8.0
        self.obs[214, :] = st['koutsu'] / 4.0
        self.obs[215, :] = st['shuntsu'] / 4.0

        # OPP_INFER (216-219)
        oi = self._calc_opponent_inference()
        for i in range(34):
            self.obs[216, i] = oi['likely_need'][i]
            self.obs[217, i] = oi['likely_discard'][i]
            self.obs[218, i] = oi['suji_safe'][i]
            self.obs[219, i] = oi['genbutsu'][i]

        # PHASE (220-223)
        ph = self._calc_game_phase()
        self.obs[220, :] = ph['early']
        self.obs[221, :] = ph['mid']
        self.obs[222, :] = ph['late']
        self.obs[223, :] = ph['all_last']


    def _calc_draw_improvement(self, hand_array, wall_remaining):
        improvement = np.zeros(34, dtype=np.float32)
        total = hand_array.sum()
        if total < 1:
            return improvement
        current_taatsu = self._count_taatsu(hand_array)
        for idx in range(34):
            if hand_array[idx] == 0:
                continue
            sim = hand_array.copy()
            sim[idx] -= 1
            new_taatsu = self._count_taatsu(sim)
            taatsu_diff = new_taatsu - current_taatsu
            draw_count = 0
            for dt in self.TILE_LIST:
                didx = self.OFFSET_TILE[dt]
                if wall_remaining[dt] <= 0:
                    continue
                sd = sim.copy()
                sd[didx] += 1
                if self._count_taatsu(sd) > new_taatsu:
                    draw_count += wall_remaining[dt]
            improvement[idx] = max(0, draw_count + taatsu_diff * 2)
        return improvement

    def _count_taatsu(self, hand_array):
        taatsu = 0
        pairs = sum(max(0, c - 1) for c in hand_array)
        taatsu += min(pairs, 7)
        for suit_idx in range(3):
            base = suit_idx * 9
            arr = hand_array[base:base + 9]
            for i in range(8):
                if arr[i] > 0 and arr[i + 1] > 0:
                    taatsu += 0.5
            for i in range(7):
                if arr[i] > 0 and arr[i + 2] > 0:
                    taatsu += 0.3
        return taatsu

    def _calc_shanten_features(self, hand_array):
        n = hand_array.sum()
        if n == 0:
            return {'current': 8, 'if_discard': np.full(34, 8.0, dtype=np.float32)}
        taatsu = self._count_taatsu(hand_array)
        mentsu = self._count_mentsu(hand_array)
        groups = mentsu + min(taatsu, 5) * 0.5
        current = max(0, 8 - int(groups * 2))
        if_discard = np.full(34, float(current), dtype=np.float32)
        for idx in range(34):
            if hand_array[idx] == 0:
                continue
            sim = hand_array.copy()
            sim[idx] -= 1
            st = self._count_taatsu(sim)
            sm = self._count_mentsu(sim)
            sg = sm + min(st, 5) * 0.5
            if_discard[idx] = max(0, 8 - int(sg * 2))
        return {'current': current, 'if_discard': if_discard}

    def _count_mentsu(self, hand_array):
        m = sum(c // 3 for c in hand_array)
        for s in range(3):
            base = s * 9
            a = hand_array[base:base + 9].copy()
            for i in range(7):
                while a[i] > 0 and a[i+1] > 0 and a[i+2] > 0:
                    m += 1
                    a[i] -= 1; a[i+1] -= 1; a[i+2] -= 1
        return min(m, 4)

    def _calc_wait_type(self, hand_array, wall_remaining):
        single = np.zeros(34, dtype=np.float32)
        double_sided = np.zeros(34, dtype=np.float32)
        pair_wait = np.zeros(34, dtype=np.float32)
        edge = np.zeros(34, dtype=np.float32)
        for idx in range(34):
            tile = self.TILE_LIST[idx]
            if wall_remaining[tile] == 0:
                continue
            if hand_array[idx] >= 2:
                single[idx] = 1.0
            if tile[0] in 'WTB':
                num = int(tile[1])
                suit = tile[0]
                sb = {'W': 0, 'T': 9, 'B': 18}[suit]
                if 2 <= num <= 7:
                    if hand_array[sb + num - 2] > 0 and hand_array[sb + num - 1] > 0:
                        double_sided[idx] = 1.0
                    if hand_array[sb + num] > 0 and hand_array[sb + num + 1] > 0:
                        double_sided[idx] = 1.0
                if num == 3 and hand_array[sb] > 0 and hand_array[sb + 1] > 0:
                    edge[idx] = 1.0
                if num == 7 and hand_array[sb + 7] > 0 and hand_array[sb + 8] > 0:
                    edge[idx] = 1.0
            n_pairs = sum(1 for c in hand_array if c >= 2)
            if n_pairs >= 2 and hand_array[idx] >= 2:
                pair_wait[idx] = 1.0
        return {'single': single, 'double_sided': double_sided, 'pair_wait': pair_wait, 'edge': edge}

    def _calc_precise_danger(self, hand_array, wall_remaining):
        danger = np.zeros(34, dtype=np.float32)
        for idx in range(34):
            tile = self.TILE_LIST[idx]
            seen = self.shownTiles.get(tile, 0) + hand_array[idx]
            if seen >= 4:
                danger[idx] = 0.0
                continue
            suji = 0.0
            if tile[0] in 'WTB':
                num = int(tile[1])
                suit = tile[0]
                st = suit + str(num + 3) if num <= 6 else suit + str(num - 3)
                si = self.OFFSET_TILE.get(st)
                if si is not None:
                    for p in range(1, 4):
                        if st in self.history[p]:
                            suji = 0.3
                            break
            gen = 0.0
            for p in range(1, 4):
                if tile in self.history[p]:
                    gen = max(gen, 0.4)
            ol = 0.0
            for p in range(1, 4):
                if len(self.packs[p]) >= 3:
                    ol = max(ol, 0.3)
            danger[idx] = max(0.0, 1.0 - suji - gen - (seen / 4.0) * 0.3) + ol
            danger[idx] = min(1.0, danger[idx])
        return danger

    def _calc_yaku_progress(self, hand_array):
        pinfu = np.zeros(34, dtype=np.float32)
        all_simples = np.zeros(34, dtype=np.float32)
        half_flush = np.zeros(34, dtype=np.float32)
        full_flush = np.zeros(34, dtype=np.float32)
        yaochu = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
        for idx in range(27):
            if hand_array[idx] > 0 and idx not in yaochu:
                all_simples[idx] = 0.5
            num = (idx % 9) + 1
            if 2 <= num <= 8 and hand_array[idx] > 0:
                pinfu[idx] = 0.3
                if idx % 9 > 0 and hand_array[idx - 1] > 0:
                    pinfu[idx] = 0.7
                if idx % 9 < 8 and hand_array[idx + 1] > 0:
                    pinfu[idx] = max(pinfu[idx], 0.7)
        for s, suit in enumerate(['W', 'T', 'B']):
            si = [self.OFFSET_TILE[suit + str(i)] for i in range(1, 10)]
            sc = sum(hand_array[i] for i in si)
            total = max(hand_array.sum(), 1)
            if sc / total > 0.7:
                for i in si:
                    half_flush[i] = sc / total
                    full_flush[i] = sc / total
        return {'pinfu': pinfu, 'all_simples': all_simples, 'half_flush': half_flush, 'full_flush': full_flush}

    def _calc_hand_structure(self, hand_array):
        pairs = sum(1 for c in hand_array if c >= 2)
        taatsu = 0
        for s in range(3):
            base = s * 9
            a = hand_array[base:base + 9]
            for i in range(8):
                if a[i] > 0 and a[i+1] > 0:
                    taatsu += 1
            for i in range(7):
                if a[i] > 0 and a[i+2] > 0:
                    taatsu += 1
        koutsu = sum(1 for c in hand_array if c >= 3)
        shuntsu = 0
        for s in range(3):
            base = s * 9
            a = hand_array[base:base + 9].copy()
            for i in range(7):
                while a[i] > 0 and a[i+1] > 0 and a[i+2] > 0:
                    shuntsu += 1
                    a[i] -= 1; a[i+1] -= 1; a[i+2] -= 1
        return {'pairs': pairs, 'taatsu': taatsu, 'koutsu': koutsu, 'shuntsu': shuntsu}

    def _calc_opponent_inference(self):
        likely_need = np.zeros(34, dtype=np.float32)
        likely_discard = np.zeros(34, dtype=np.float32)
        suji_safe = np.zeros(34, dtype=np.float32)
        genbutsu = np.zeros(34, dtype=np.float32)
        for p in range(1, 4):
            discards = self.history[p]
            suit_count = defaultdict(int)
            for d in discards:
                suit_count[d[0]] += 1
            for idx in range(34):
                tile = self.TILE_LIST[idx]
                suit = tile[0]
                td = suit_count.get(suit, 0)
                if td <= 2 and len(discards) > 5:
                    likely_need[idx] += 0.15
                if tile in discards:
                    genbutsu[idx] = 1.0
                    if suit in 'WTB':
                        num = int(tile[1])
                        for delta in [-3, 3]:
                            n2 = num + delta
                            if 1 <= n2 <= 9:
                                t2i = self.OFFSET_TILE[suit + str(n2)]
                                suji_safe[t2i] = max(suji_safe[t2i], 0.5)
                if suit in 'WTB':
                    num = int(tile[1])
                    for d in discards:
                        if d[0] == suit and abs(int(d[1]) - num) <= 1:
                            likely_discard[idx] += 0.1
        return {'likely_need': np.clip(likely_need, 0, 1), 'likely_discard': np.clip(likely_discard, 0, 1),
                'suji_safe': np.clip(suji_safe, 0, 1), 'genbutsu': genbutsu}

    def _calc_game_phase(self):
        tp = sum(len(h) for h in self.history)
        if tp < 12:
            return {'early': 1.0, 'mid': 0.0, 'late': 0.0, 'all_last': 0.0}
        elif tp < 36:
            return {'early': 0.0, 'mid': 1.0, 'late': 0.0, 'all_last': 0.0}
        elif tp < 54:
            return {'early': 0.0, 'mid': 0.0, 'late': 1.0, 'all_last': 0.0}
        else:
            return {'early': 0.0, 'mid': 0.0, 'late': 0.0, 'all_last': 1.0}


    def _hand_to_array(self):
        arr = np.zeros(34, dtype=np.int32)
        for t in self.hand:
            arr[self.OFFSET_TILE[t]] += 1
        return arr

    def _shown_to_array(self):
        arr = np.zeros(34, dtype=np.int32)
        for t, c in self.shownTiles.items():
            arr[self.OFFSET_TILE[t]] = c
        return arr

    def _calc_tile_efficiency(self, ha, wr):
        eff = np.zeros(34, dtype=np.float32)
        for idx in range(34):
            tile = self.TILE_LIST[idx]
            cnt = ha[idx]
            if cnt == 0:
                continue
            pp = min(cnt, 3) / 3.0
            suit = tile[0]
            num = int(tile[1])
            sp = 0.0
            ip = 0.0
            if suit in 'WTB':
                nb = []
                for d in range(-2, 3):
                    n = num + d
                    if 1 <= n <= 9:
                        ni = self.OFFSET_TILE[suit + str(n)]
                        nb.append(ha[ni])
                    else:
                        nb.append(0)
                ss = []
                for start in range(max(0, num - 2), min(7, num)):
                    if all(nb[start + k - (num - 2)] > 0 for k in range(3)):
                        ss.append(1.0)
                    elif sum(1 for k in range(3) if nb[start + k - (num - 2)] > 0) >= 2:
                        ss.append(0.5)
                sp = max(ss) if ss else 0.0
                ha_adj = False
                for d in (-1, 1):
                    n = num + d
                    if 1 <= n <= 9:
                        if ha[self.OFFSET_TILE[suit + str(n)]] > 0:
                            ha_adj = True
                            break
                if not ha_adj and cnt == 1:
                    ip = 0.3
            if suit in 'FJ':
                pb = min(cnt, 2) / 2.0
                eff[idx] = pb * 0.8
            else:
                eff[idx] = max(0.0, max(pp, sp) - ip)
            rem = wr[tile]
            af = min(rem / 2.0, 1.0)
            eff[idx] *= (0.3 + 0.7 * af)
        return np.clip(eff, 0, 1)

    def _calc_tile_safety(self, wr):
        s = np.ones(34, dtype=np.float32) * 0.5
        for idx in range(34):
            tile = self.TILE_LIST[idx]
            ts = tile[0]
            tn = int(tile[1])
            if ts in 'FJ':
                dc = sum(1 for p in range(1, 4) for t in self.history[p] if t == tile)
                if dc >= 1:
                    s[idx] = 0.7 + 0.15 * min(dc, 2)
                seen = self.shownTiles.get(tile, 0)
                if seen >= 3:
                    s[idx] = 1.0
            else:
                nd = 0
                for p in range(1, 4):
                    for t in self.history[p]:
                        if t[0] == ts and abs(int(t[1]) - tn) <= 2:
                            nd += 1
                if nd >= 4: s[idx] = 0.8
                elif nd >= 2: s[idx] = 0.65
                rv = 0; tr = 0
                for d in range(-2, 3):
                    n = tn + d
                    if 1 <= n <= 9:
                        tr += 1
                        if self.shownTiles.get(ts + str(n), 0) > 0:
                            rv += 1
                if tr > 0 and rv / tr > 0.6:
                    s[idx] = max(s[idx], 0.75)
            rem = wr[tile]
            if rem == 0: s[idx] = 1.0
            elif rem == 1: s[idx] += 0.1
            ttp = sum(len(h) for h in self.history)
            if ttp > 40: s[idx] = min(1.0, s[idx] + 0.1)
            if ttp > 55: s[idx] = min(1.0, s[idx] + 0.15)
        return np.clip(s, 0, 1)

    def _calc_opponent_risk(self):
        risk = np.zeros(34, dtype=np.float32)
        for p in range(1, 4):
            sdc = {'W': 0, 'T': 0, 'B': 0, 'F': 0, 'J': 0}
            stc = {'W': 9, 'T': 9, 'B': 9, 'F': 4, 'J': 3}
            for t in self.history[p]:
                if t[0] in sdc: sdc[t[0]] += 1
            td = max(len(self.history[p]), 1)
            for idx in range(34):
                tile = self.TILE_LIST[idx]
                suit = tile[0]
                dr = sdc.get(suit, 0) / max(stc[suit], 1)
                if dr < 0.1 and td > 5:
                    risk[idx] += 0.15
                if suit in 'WTB':
                    num = int(tile[1])
                    for t in self.history[p]:
                        if t[0] == suit and abs(int(t[1]) - num) <= 1:
                            risk[idx] += 0.05
                            break
            mc = len(self.packs[p])
            if mc >= 3:
                risk += 0.05 * (mc - 2)
        return np.clip(risk, 0, 1)

    def _calc_yaku_potential(self, ha):
        f = [np.zeros(34, dtype=np.float32) for _ in range(8)]
        for idx in range(27):
            num = (idx % 9) + 1
            if 2 <= num <= 8 and ha[idx] > 0:
                f[0][idx] = 0.5 + 0.5 * min(ha[idx], 2) / 2.0
        for idx in range(34):
            if ha[idx] >= 2: f[1][idx] = min(ha[idx], 3) / 3.0
        for suit in ['W', 'T', 'B']:
            si = [self.OFFSET_TILE[suit + str(i)] for i in range(1, 10)]
            sc = sum(ha[i] for i in si)
            total = max(ha.sum(), 1)
            if sc / total > 0.7:
                for i in si: f[2][i] = 0.8
        for idx in range(27, 34):
            if ha[idx] >= 2: f[3][idx] = min(ha[idx], 3) / 3.0
        for suit in ['W', 'T', 'B']:
            for num in [1, 2, 3, 7, 8, 9]:
                idx = self.OFFSET_TILE[suit + str(num)]
                if ha[idx] > 0: f[4][idx] = 0.6
        for idx in range(34):
            if ha[idx] > 0 and ha[idx] < 2:
                tile = self.TILE_LIST[idx]
                if tile[0] in 'WTB' and 2 <= int(tile[1]) <= 8:
                    f[5][idx] = 0.4
        if len(self.packs[0]) == 0:
            for idx in range(34):
                if ha[idx] > 0: f[6][idx] = 0.3
        for idx in range(34):
            if ha[idx] == 4 or (ha[idx] == 3 and self.shownTiles.get(self.TILE_LIST[idx], 0) == 0):
                f[7][idx] = 0.7
        return f

    def _calc_discard_stats(self):
        f = [np.zeros(34, dtype=np.float32) for _ in range(8)]
        for p in range(4):
            base = p * 2
            r = self.history[p][-7:]
            for t in r:
                idx = self.OFFSET_TILE.get(t)
                if idx is not None: f[base][idx] += 1.0 / max(len(r), 1)
            r14 = self.history[p][-14:]
            for t in r14:
                idx = self.OFFSET_TILE.get(t)
                if idx is not None: f[base + 1][idx] += 1.0 / max(len(r14), 1)
        return f

    def _estimate_tenpai_chance(self, ha):
        total = ha.sum()
        if total < 13: return 0.0
        pairs = sum(1 for c in ha if c >= 2)
        triplets = sum(1 for c in ha if c >= 3)
        ml = triplets + min(pairs, 4)
        if ml >= 4: return min(0.5 + 0.1 * (ml - 4), 1.0)
        elif ml >= 3: return 0.2 + 0.15 * (ml - 3)
        return 0.1


    def _make_obs(self):
        mask = np.zeros(self.ACT_SIZE, dtype=np.float32)
        for a in self.valid: mask[a] = 1
        return {'observation': self.obs.reshape((self.OBS_SIZE, 4, 9)).copy(), 'action_mask': mask}

    def request2obs(self, request: str):
        t = request.split()
        if t[0] == 'Wind':
            self.prevalentWind = int(t[1])
            self._update_obs()
            return None
        if t[0] == 'Deal':
            self.hand = t[1:]
            self._update_obs()
            return None
        if t[0] in ('Huang',):
            self.valid = []
            return self._make_obs()
        if t[0] == 'Draw':
            self.tileWall[0] -= 1
            self.wallLast = self.tileWall[1] == 0
            tile = t[1]
            self.valid = []
            if self._can_hu(tile, isSelfDrawn=True, isAboutKong=self.isAboutKong):
                self.valid.append(self.OFFSET_ACT['Hu'])
            self.isAboutKong = False
            self.hand.append(tile)
            for _t in set(self.hand):
                self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[_t])
                if self.hand.count(_t) == 4 and not self.wallLast and self.tileWall[0] > 0:
                    self.valid.append(self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[_t])
            if not self.wallLast and self.tileWall[0] > 0:
                for packType, ptile, offer in self.packs[0]:
                    if packType == 'PENG' and ptile in self.hand:
                        self.valid.append(self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[ptile])
            self._update_obs()
            return self._make_obs()
        p = (int(t[1]) + 4 - self.seatWind) % 4
        if t[2] == 'Draw':
            self.tileWall[p] -= 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            return None
        if t[2] in ('Invalid', 'Hu'):
            self.valid = []
            return self._make_obs()
        if t[2] == 'Play':
            self.tileFrom = p
            self.curTile = t[3]
            self.shownTiles[self.curTile] += 1
            self.history[p].append(self.curTile)
            if p == 0:
                self.hand.remove(self.curTile)
                self._update_obs()
                return None
            else:
                self.valid = []
                if self._can_hu(self.curTile): self.valid.append(self.OFFSET_ACT['Hu'])
                if not self.wallLast:
                    if self.hand.count(self.curTile) >= 2:
                        self.valid.append(self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[self.curTile])
                        if self.hand.count(self.curTile) == 3 and self.tileWall[0]:
                            self.valid.append(self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[self.curTile])
                    color = self.curTile[0]
                    if p == 3 and color in 'WTB':
                        num = int(self.curTile[1])
                        tmp = [color + str(num + i) for i in range(-2, 3)]
                        if tmp[0] in self.hand and tmp[1] in self.hand:
                            idx = ('WTB'.index(color) * 21 + (num - 3) * 3 + 2)
                            self.valid.append(self.OFFSET_ACT['Chi'] + idx)
                        if tmp[1] in self.hand and tmp[3] in self.hand:
                            idx = ('WTB'.index(color) * 21 + (num - 2) * 3 + 1)
                            self.valid.append(self.OFFSET_ACT['Chi'] + idx)
                        if tmp[3] in self.hand and tmp[4] in self.hand:
                            idx = ('WTB'.index(color) * 21 + (num - 1) * 3)
                            self.valid.append(self.OFFSET_ACT['Chi'] + idx)
                self.valid.append(self.OFFSET_ACT['Pass'])
                self._update_obs()
                return self._make_obs()
        if t[2] == 'Chi':
            tile = t[3]; color = tile[0]; num = int(tile[1])
            self.packs[p].append(('CHI', tile, int(self.curTile[1]) - num + 2))
            self.shownTiles[self.curTile] -= 1
            for i in range(-1, 2): self.shownTiles[color + str(num + i)] += 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                self.valid = []
                self.hand.append(self.curTile)
                for i in range(-1, 2): self.hand.remove(color + str(num + i))
                for _t in set(self.hand): self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[_t])
                self._update_obs(); return self._make_obs()
            return None
        if t[2] == 'UnChi':
            tile = t[3]; color = tile[0]; num = int(tile[1])
            self.packs[p].pop(); self.shownTiles[self.curTile] += 1
            for i in range(-1, 2): self.shownTiles[color + str(num + i)] -= 1
            if p == 0:
                for i in range(-1, 2): self.hand.append(color + str(num + i))
                self.hand.remove(self.curTile); self._update_obs()
            return None
        if t[2] == 'Peng':
            self.packs[p].append(('PENG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 2
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                self.valid = []
                for _ in range(2): self.hand.remove(self.curTile)
                for _t in set(self.hand): self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[_t])
                self._update_obs(); return self._make_obs()
            return None
        if t[2] == 'UnPeng':
            self.packs[p].pop(); self.shownTiles[self.curTile] -= 2
            if p == 0:
                for _ in range(2): self.hand.append(self.curTile)
                self._update_obs()
            return None
        if t[2] == 'Gang':
            self.packs[p].append(('GANG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 3
            if p == 0:
                for _ in range(3): self.hand.remove(self.curTile)
                self.isAboutKong = True; self._update_obs()
            return None
        if t[2] == 'AnGang':
            tile = t[3] if (p == 0 and len(t) > 3) else 'CONCEALED'
            self.packs[p].append(('GANG', tile, 0))
            if p == 0:
                self.isAboutKong = True
                for _ in range(4):
                    self.hand.remove(tile)
            else:
                self.isAboutKong = False
            self._update_obs()
            return None
        if t[2] == 'BuGang':
            tile = t[3]
            for i in range(len(self.packs[p])):
                if tile == self.packs[p][i][1]:
                    self.packs[p][i] = ('GANG', tile, self.packs[p][i][2])
                    break
            self.shownTiles[tile] += 1
            if p == 0:
                self.hand.remove(tile); self.isAboutKong = True
                self._update_obs(); return None
            else:
                self.valid = []
                if self._can_hu(tile, isSelfDrawn=False, isAboutKong=True):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                self.valid.append(self.OFFSET_ACT['Pass'])
                self._update_obs(); return self._make_obs()
        raise NotImplementedError('Unknown request: %s' % request)

    def action2response(self, action: int) -> str:
        if action < self.OFFSET_ACT['Hu']: return 'Pass'
        if action < self.OFFSET_ACT['Play']: return 'Hu'
        if action < self.OFFSET_ACT['Chi']:
            return 'Play ' + self.TILE_LIST[action - self.OFFSET_ACT['Play']]
        if action < self.OFFSET_ACT['Peng']:
            t = (action - self.OFFSET_ACT['Chi']) // 3
            return 'Chi ' + 'WTB'[t // 7] + str(t % 7 + 2)
        if action < self.OFFSET_ACT['Gang']: return 'Peng'
        if action < self.OFFSET_ACT['AnGang']: return 'Gang'
        if action < self.OFFSET_ACT['BuGang']:
            return 'Gang ' + self.TILE_LIST[action - self.OFFSET_ACT['AnGang']]
        return 'BuGang ' + self.TILE_LIST[action - self.OFFSET_ACT['BuGang']]

    def response2action(self, response: str) -> int:
        t = response.split()
        if t[0] == 'Pass': return self.OFFSET_ACT['Pass']
        if t[0] == 'Hu': return self.OFFSET_ACT['Hu']
        if t[0] == 'Play': return self.OFFSET_ACT['Play'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Chi':
            return (self.OFFSET_ACT['Chi'] + 'WTB'.index(t[1][0]) * 7 * 3 +
                   (int(t[2][1]) - 2) * 3 + int(t[1][1]) - int(t[2][1]) + 1)
        if t[0] == 'Peng': return self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Gang': return self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'AnGang': return self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'BuGang': return self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[t[1]]
        return self.OFFSET_ACT['Pass']

    def _can_hu(self, winTile: str, isSelfDrawn: bool = False, isAboutKong: bool = False) -> bool:
        try:
            # FIX: is4thTile must include tiles in our own hand
            total_seen = self.shownTiles.get(winTile, 0) + self.hand.count(winTile)
            fans = MahjongFanCalculator(
                pack=tuple(self.packs[0]), hand=tuple(self.hand), winTile=winTile,
                flowerCount=0, isSelfDrawn=isSelfDrawn,
                is4thTile=(total_seen + isSelfDrawn) == 4,
                isAboutKong=isAboutKong, isWallLast=self.wallLast,
                seatWind=self.seatWind, prevalentWind=self.prevalentWind, verbose=True
            )
            if sum(fanPoint * cnt for fanPoint, cnt, _, _ in fans) < 8:
                raise Exception('Not Enough Fans')
            return True
        except Exception:
            return False

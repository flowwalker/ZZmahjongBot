"""状态编码器 — GameState → NN 观测."""

from collections import defaultdict
import numpy as np


# 常量 (与 baseline-v2/feature.py 完全一致)

OBS_SIZE = 16
ACT_SIZE = 235

OFFSET_OBS = {
    'SEAT_WIND': 0,
    'PREVALENT_WIND': 1,
    'HAND': 2,
    'DISCARD': 6,
    'MELD': 10,
    'SHOWN': 14,
    'WALL': 15,
}

OFFSET_ACT = {
    'Pass': 0,
    'Hu': 1,
    'Play': 2,
    'Chi': 36,
    'Peng': 99,
    'Gang': 133,
    'AnGang': 167,
    'BuGang': 201,
}

TILE_LIST = [
    *('W%d' % (i + 1) for i in range(9)),   # 万: W1-W9
    *('T%d' % (i + 1) for i in range(9)),   # 条: T1-T9
    *('B%d' % (i + 1) for i in range(9)),   # 筒: B1-B9
    *('F%d' % (i + 1) for i in range(4)),   # 风: F1(东)-F4(北)
    *('J%d' % (i + 1) for i in range(3)),   # 箭: J1(中)-J3(白)
]

OFFSET_TILE = {c: i for i, c in enumerate(TILE_LIST)}


class StateEncoder:
    """将 GameState 对象编码为 NN 输入张量"""

    @staticmethod
    def encode(gs) -> tuple:
        """
        完整编码：观测张量 + 动作掩码

        Args:
            gs: GameState 对象 (需含 hand, packs, history/oppDiscards, shownTiles, tileWall, seatWind, prevalentWind, valid)

        Returns:
            (observation: np.ndarray(16,4,9), action_mask: np.ndarray(235,))
        """
        obs = np.zeros((OBS_SIZE, 36), dtype=np.float32)

        # 门风
        obs[OFFSET_OBS['SEAT_WIND']][OFFSET_TILE['F%d' % (gs.seatWind + 1)]] = 1

        # 圈风
        obs[OFFSET_OBS['PREVALENT_WIND']][OFFSET_TILE['F%d' % (gs.prevalentWind + 1)]] = 1

        # 手牌 (ch2-5): 按重复张数堆叠
        d = defaultdict(int)
        for tile in gs.hand:
            d[tile] += 1
        for tile, cnt in d.items():
            obs[OFFSET_OBS['HAND']:OFFSET_OBS['HAND'] + cnt,
                OFFSET_TILE[tile]] = 1

        # 弃牌 (ch6-9): 每人一通道
        # baseline_human_v3 用 oppDiscards，baseline-v2 用 history
        discards = getattr(gs, 'oppDiscards', None) or getattr(gs, 'history', None)
        if discards:
            for pi in range(4):
                ch = OFFSET_OBS['DISCARD'] + pi
                for tile in discards[pi]:
                    idx = OFFSET_TILE.get(tile)
                    if idx is not None:
                        obs[ch, idx] = min(4, obs[ch, idx] + 1)

        # 鸣牌 (ch10-13): 每人一通道
        for pi in range(4):
            ch = OFFSET_OBS['MELD'] + pi
            for pkt, tile, _ in gs.packs[pi]:
                idx = OFFSET_TILE.get(tile)
                if idx is not None:
                    if pkt == 'CHI':
                        c, n = tile[0], int(tile[1])
                        for dn in (-1, 0, 1):
                            i2 = OFFSET_TILE.get(c + str(n + dn))
                            if i2 is not None:
                                obs[ch, i2] = 1
                    else:
                        obs[ch, idx] = 4 if pkt == 'GANG' else 3

        # 已见牌 (ch14)
        for tile, cnt in gs.shownTiles.items():
            idx = OFFSET_TILE.get(tile)
            if idx is not None:
                obs[OFFSET_OBS['SHOWN'], idx] = cnt / 4.0

        # 牌墙 (ch15)
        obs[OFFSET_OBS['WALL'], :] = gs.tileWall[0] / 21.0

        observation = obs.reshape((OBS_SIZE, 4, 9))

        # 动作掩码
        mask = np.zeros(ACT_SIZE, dtype=np.float32)
        if hasattr(gs, 'valid') and gs.valid:
            for a in gs.valid:
                mask[a] = 1
        else:
            # 从 GameState 推断合法动作
            mask = StateEncoder._infer_action_mask(gs)

        return observation, mask

    @staticmethod
    def encode_for_value_only(hand, packs, shownTiles, tileWall, seatWind,
                               prevalentWind, oppDiscards=None) -> np.ndarray:
        """
        轻量编码：仅生成观测张量用于 value head 评估

        不计算 action_mask（value head 评估局面价值不需要），
        用于 FlatMC 模拟中叶节点的批量评估。

        Args:
            hand: 手牌列表
            packs: 鸣牌列表 (4人)
            shownTiles: 已见牌计数字典
            tileWall: 牌墙列表 (4人)
            seatWind: 门风
            prevalentWind: 圈风
            oppDiscards: 弃牌历史 (可选)

        Returns:
            observation: np.ndarray(16, 4, 9)
        """
        obs = np.zeros((OBS_SIZE, 36), dtype=np.float32)

        # 门风
        obs[OFFSET_OBS['SEAT_WIND']][OFFSET_TILE['F%d' % (seatWind + 1)]] = 1

        # 圈风
        obs[OFFSET_OBS['PREVALENT_WIND']][OFFSET_TILE['F%d' % (prevalentWind + 1)]] = 1

        # 手牌
        d = defaultdict(int)
        for tile in hand:
            d[tile] += 1
        for tile, cnt in d.items():
            obs[OFFSET_OBS['HAND']:OFFSET_OBS['HAND'] + cnt,
                OFFSET_TILE[tile]] = 1

        # 弃牌
        if oppDiscards:
            for pi in range(min(4, len(oppDiscards))):
                ch = OFFSET_OBS['DISCARD'] + pi
                for tile in oppDiscards[pi]:
                    idx = OFFSET_TILE.get(tile)
                    if idx is not None:
                        obs[ch, idx] = min(4, obs[ch, idx] + 1)

        # 鸣牌
        for pi in range(min(4, len(packs))):
            ch = OFFSET_OBS['MELD'] + pi
            for pkt, tile, _ in packs[pi]:
                idx = OFFSET_TILE.get(tile)
                if idx is not None:
                    if pkt == 'CHI':
                        c, n = tile[0], int(tile[1])
                        for dn in (-1, 0, 1):
                            i2 = OFFSET_TILE.get(c + str(n + dn))
                            if i2 is not None:
                                obs[ch, i2] = 1
                    else:
                        obs[ch, idx] = 4 if pkt == 'GANG' else 3

        # 已见牌
        for tile, cnt in (shownTiles.items() if isinstance(shownTiles, dict)
                          else shownTiles):
            idx = OFFSET_TILE.get(tile)
            if idx is not None:
                obs[OFFSET_OBS['SHOWN'], idx] = cnt / 4.0

        # 牌墙
        if isinstance(tileWall, (list, tuple)):
            obs[OFFSET_OBS['WALL'], :] = tileWall[0] / 21.0

        return obs.reshape((OBS_SIZE, 4, 9))

    @staticmethod
    def _infer_action_mask(gs) -> np.ndarray:
        """从 GameState 推断合法动作掩码（当 gs.valid 未设置时使用）"""
        mask = np.zeros(ACT_SIZE, dtype=np.float32)
        mask[OFFSET_ACT['Pass']] = 1  # Pass 始终合法

        # 出牌: 手牌中每种牌可打
        for t in set(gs.hand):
            mask[OFFSET_ACT['Play'] + OFFSET_TILE[t]] = 1

        return mask

    @staticmethod
    def encode_batch_for_value(leaf_states: list, base_gs) -> np.ndarray:
        """
        批量编码叶节点状态（用于 FlatMC VHR 批量推理）

        Args:
            leaf_states: [(hand, pack), ...] 列表
            base_gs: 基准 GameState（提供 shownTiles, tileWall, seatWind, prevalentWind, oppDiscards）

        Returns:
            observations: np.ndarray(N, 16, 4, 9)
        """
        obs_list = []
        for hand, pack in leaf_states:
            obs = StateEncoder.encode_for_value_only(
                hand, [pack] + base_gs.packs[1:],
                base_gs.shownTiles, base_gs.tileWall,
                base_gs.seatWind, base_gs.prevalentWind,
                base_gs.oppDiscards
            )
            obs_list.append(obs)
        return np.stack(obs_list) if obs_list else np.empty((0, OBS_SIZE, 4, 9))

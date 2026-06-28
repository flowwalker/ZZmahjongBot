"""国标麻将特征提取器 — 6通道."""

from agent import MahjongGBAgent
from collections import defaultdict
import numpy as np

try:
    from MahjongGB import MahjongFanCalculator
except ImportError:
    print('MahjongGB library required! '
          'Please visit https://github.com/ailab-pku/PyMahjongGB for more information.')
    raise


class FeatureAgent(MahjongGBAgent):
    """国标麻将特征提取 Agent"""

    # 观测和动作空间维度
    OBS_SIZE = 6       # 6个通道
    ACT_SIZE = 235     # 235个离散动作

    # 观测通道偏移
    OFFSET_OBS = {
        'SEAT_WIND': 0,
        'PREVALENT_WIND': 1,
        'HAND': 2,
    }

    # 动作类型偏移
    OFFSET_ACT = {
        'Pass': 0,
        'Hu': 1,
        'Play': 2,
        'Chi': 36,
        'Peng': 99,
        'Gang': 133,
        'AnGang': 167,
        'BuGang': 201
    }

    # 34种牌的完整列表
    TILE_LIST = [
        *('W%d' % (i + 1) for i in range(9)),   # 万: W1-W9
        *('T%d' % (i + 1) for i in range(9)),   # 条: T1-T9
        *('B%d' % (i + 1) for i in range(9)),   # 筒: B1-B9
        *('F%d' % (i + 1) for i in range(4)),   # 风: F1(东)-F4(北)
        *('J%d' % (i + 1) for i in range(3))    # 箭: J1(中)-J3(白)
    ]

    # 牌名到索引的映射
    OFFSET_TILE = {c: i for i, c in enumerate(TILE_LIST)}

    def __init__(self, seatWind: int):
        """
        初始化 Agent 状态

        Args:
            seatWind: 门风 (0=东, 1=南, 2=西, 3=北)
        """
        self.seatWind = seatWind

        # 四人鸣牌记录 (吃/碰/杠)
        self.packs = [[] for _ in range(4)]

        # 四人弃牌历史
        self.history = [[] for _ in range(4)]

        # 四人剩余牌墙数 (初始每人21张, 复式赛制34张)
        self.tileWall = [21] * 4

        # 已见牌计数 (用于判断绝张)
        self.shownTiles = defaultdict(int)

        # 状态标记
        self.wallLast = False       # 是否牌墙最后一张
        self.isAboutKong = False    # 是否刚进行杠操作

        # 观测矩阵: (6, 36) — 6通道 × 34牌+2占位
        self.obs = np.zeros((self.OBS_SIZE, 36))

        # 设置门风通道
        self.obs[self.OFFSET_OBS['SEAT_WIND']][
            self.OFFSET_TILE['F%d' % (self.seatWind + 1)]
        ] = 1

    #  Request 解析 (Botzone → 内部状态 + 观测)

    def request2obs(self, request: str):
        """
        解析 Botzone request 字符串，更新内部状态。

        支持的 request 类型:
          Wind <0-3>            — 圈风信息
          Deal <cards>          — 初始手牌 (13张)
          Draw <card>           — 摸牌 (需要决策)
          Player N Draw         — 其他玩家摸牌
          Player N Play <card>  — 玩家出牌
          Player N Chi <card>   — 玩家吃牌
          Player N Peng         — 玩家碰牌
          Player N Gang         — 玩家明杠
          Player N AnGang <card>— 玩家暗杠
          Player N BuGang <card>— 玩家补杠
          Player N Hu           — 玩家和牌
          Huang                 — 荒庄
          Player N Invalid      — 玩家非法操作

        Returns:
            dict or None: 需要决策时返回观测，否则返回 None
        """
        t = request.split()

        if t[0] == 'Wind':
            self.prevalentWind = int(t[1])
            self.obs[self.OFFSET_OBS['PREVALENT_WIND']][
                self.OFFSET_TILE['F%d' % (self.prevalentWind + 1)]
            ] = 1
            return None

        if t[0] == 'Deal':
            self.hand = t[1:]
            self._update_hand_embedding()
            return None

        if t[0] in ('Huang',):
            self.valid = []
            return self._make_obs()

        if t[0] == 'Draw':
            self.tileWall[0] -= 1
            self.wallLast = self.tileWall[1] == 0
            tile = t[1]
            self.valid = []

            # 检查是否可以胡
            if self._can_hu(tile, isSelfDrawn=True, isAboutKong=self.isAboutKong):
                self.valid.append(self.OFFSET_ACT['Hu'])

            self.isAboutKong = False
            self.hand.append(tile)
            self._update_hand_embedding()

            # 出牌选项 (每种手牌可以打出)
            for t in set(self.hand):
                self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[t])
                # 暗杠: 手中有4张相同牌且牌墙非空
                if (self.hand.count(t) == 4
                        and not self.wallLast
                        and self.tileWall[0] > 0):
                    self.valid.append(self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[t])

            # 补杠: 之前碰过的牌现在手中有第4张
            if not self.wallLast and self.tileWall[0] > 0:
                for packType, ptile, offer in self.packs[0]:
                    if packType == 'PENG' and ptile in self.hand:
                        self.valid.append(
                            self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[ptile]
                        )

            return self._make_obs()

        # 计算相对位置 (0=自己, 1=下家, 2=对家, 3=上家)
        p = (int(t[1]) + 4 - self.seatWind) % 4

        if t[2] == 'Draw':
            self.tileWall[p] -= 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            return None

        if t[2] == 'Invalid':
            self.valid = []
            return self._make_obs()

        if t[2] == 'Hu':
            self.valid = []
            return self._make_obs()

        if t[2] == 'Play':
            self.tileFrom = p
            self.curTile = t[3]
            self.shownTiles[self.curTile] += 1
            self.history[p].append(self.curTile)

            if p == 0:
                # 自己出牌后 (已在摸牌阶段处理)
                self.hand.remove(self.curTile)
                self._update_hand_embedding()
                return None
            else:
                # 其他玩家出牌 → 需要决策: Hu/Gang/Peng/Chi/Pass
                self.valid = []
                if self._can_hu(self.curTile):
                    self.valid.append(self.OFFSET_ACT['Hu'])

                if not self.wallLast:
                    # 碰: 手中有2张以上相同牌
                    if self.hand.count(self.curTile) >= 2:
                        self.valid.append(
                            self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[self.curTile]
                        )
                        # 明杠: 手中有3张相同牌
                        if self.hand.count(self.curTile) == 3 and self.tileWall[0]:
                            self.valid.append(
                                self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[self.curTile]
                            )

                    # 吃: 仅上家(p==3)打出的数牌
                    color = self.curTile[0]
                    if p == 3 and color in 'WTB':
                        num = int(self.curTile[1])
                        tmp = [color + str(num + i) for i in range(-2, 3)]
                        # 三种吃法: curTile在左/中/右
                        if tmp[0] in self.hand and tmp[1] in self.hand:
                            idx = ('WTB'.index(color) * 21
                                   + (num - 3) * 3 + 2)
                            self.valid.append(self.OFFSET_ACT['Chi'] + idx)
                        if tmp[1] in self.hand and tmp[3] in self.hand:
                            idx = ('WTB'.index(color) * 21
                                   + (num - 2) * 3 + 1)
                            self.valid.append(self.OFFSET_ACT['Chi'] + idx)
                        if tmp[3] in self.hand and tmp[4] in self.hand:
                            idx = ('WTB'.index(color) * 21
                                   + (num - 1) * 3)
                            self.valid.append(self.OFFSET_ACT['Chi'] + idx)

                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._make_obs()

        if t[2] == 'Chi':
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].append(('CHI', tile, int(self.curTile[1]) - num + 2))
            self.shownTiles[self.curTile] -= 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] += 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0

            if p == 0:
                # 自己吃牌后 → 需要出牌
                self.valid = []
                self.hand.append(self.curTile)
                for i in range(-1, 2):
                    self.hand.remove(color + str(num + i))
                self._update_hand_embedding()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._make_obs()
            return None

        if t[2] == 'UnChi':
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].pop()
            self.shownTiles[self.curTile] += 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] -= 1
            if p == 0:
                for i in range(-1, 2):
                    self.hand.append(color + str(num + i))
                self.hand.remove(self.curTile)
                self._update_hand_embedding()
            return None

        if t[2] == 'Peng':
            self.packs[p].append(('PENG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 2
            self.wallLast = self.tileWall[(p + 1) % 4] == 0

            if p == 0:
                # 自己碰牌后 → 需要出牌
                self.valid = []
                for _ in range(2):
                    self.hand.remove(self.curTile)
                self._update_hand_embedding()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._make_obs()
            return None

        if t[2] == 'UnPeng':
            self.packs[p].pop()
            self.shownTiles[self.curTile] -= 2
            if p == 0:
                for _ in range(2):
                    self.hand.append(self.curTile)
                self._update_hand_embedding()
            return None

        if t[2] == 'Gang':
            self.packs[p].append(('GANG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 3
            if p == 0:
                for _ in range(3):
                    self.hand.remove(self.curTile)
                self._update_hand_embedding()
                self.isAboutKong = True
            return None

        if t[2] == 'AnGang':
            tile = 'CONCEALED' if p else t[3]
            self.packs[p].append(('GANG', tile, 0))
            if p == 0:
                self.isAboutKong = True
                for _ in range(4):
                    self.hand.remove(tile)
            else:
                self.isAboutKong = False
            return None

        if t[2] == 'BuGang':
            tile = t[3]
            # 更新鸣牌记录: PENG → GANG
            for i in range(len(self.packs[p])):
                if tile == self.packs[p][i][1]:
                    self.packs[p][i] = ('GANG', tile, self.packs[p][i][2])
                    break
            self.shownTiles[tile] += 1

            if p == 0:
                self.hand.remove(tile)
                self._update_hand_embedding()
                self.isAboutKong = True
                return None
            else:
                # 其他玩家补杠 → 可以抢杠胡
                self.valid = []
                if self._can_hu(tile, isSelfDrawn=False, isAboutKong=True):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._make_obs()

        raise NotImplementedError('Unknown request: %s' % request)

    #  Action → Response 编码

    def action2response(self, action: int) -> str:
        """将整数动作转换为 Botzone response 字符串"""
        if action < self.OFFSET_ACT['Hu']:
            return 'Pass'
        if action < self.OFFSET_ACT['Play']:
            return 'Hu'
        if action < self.OFFSET_ACT['Chi']:
            return 'Play ' + self.TILE_LIST[action - self.OFFSET_ACT['Play']]
        if action < self.OFFSET_ACT['Peng']:
            t = (action - self.OFFSET_ACT['Chi']) // 3
            return 'Chi ' + 'WTB'[t // 7] + str(t % 7 + 2)
        if action < self.OFFSET_ACT['Gang']:
            return 'Peng'
        if action < self.OFFSET_ACT['AnGang']:
            return 'Gang'
        if action < self.OFFSET_ACT['BuGang']:
            return 'Gang ' + self.TILE_LIST[action - self.OFFSET_ACT['AnGang']]
        return 'BuGang ' + self.TILE_LIST[action - self.OFFSET_ACT['BuGang']]

    def response2action(self, response: str) -> int:
        """将 Botzone response 字符串转换为整数动作 (用于数据预处理)"""
        t = response.split()
        if t[0] == 'Pass':
            return self.OFFSET_ACT['Pass']
        if t[0] == 'Hu':
            return self.OFFSET_ACT['Hu']
        if t[0] == 'Play':
            return self.OFFSET_ACT['Play'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Chi':
            return (self.OFFSET_ACT['Chi']
                    + 'WTB'.index(t[1][0]) * 7 * 3
                    + (int(t[2][1]) - 2) * 3
                    + int(t[1][1]) - int(t[2][1]) + 1)
        if t[0] == 'Peng':
            return self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Gang':
            return self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'AnGang':
            return self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'BuGang':
            return self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[t[1]]
        return self.OFFSET_ACT['Pass']

    #  内部辅助方法

    def _make_obs(self) -> dict:
        """构建观测字典"""
        mask = np.zeros(self.ACT_SIZE)
        for a in self.valid:
            mask[a] = 1
        return {
            'observation': self.obs.reshape((self.OBS_SIZE, 4, 9)).copy(),
            'action_mask': mask
        }

    def _update_hand_embedding(self):
        """根据当前手牌更新观测中的手牌通道"""
        # 清空手牌通道
        self.obs[self.OFFSET_OBS['HAND']:] = 0
        # 统计每种牌的数量
        d = defaultdict(int)
        for tile in self.hand:
            d[tile] += 1
        # 按数量堆叠编码
        for tile, count in d.items():
            start = self.OFFSET_OBS['HAND']
            self.obs[start:start + count, self.OFFSET_TILE[tile]] = 1

    def _can_hu(self, winTile: str, isSelfDrawn: bool = False,
                isAboutKong: bool = False) -> bool:
        """
        检查当前手牌是否可以和牌 (8番起胡)

        使用 PyMahjongGB 的 MahjongFanCalculator 进行算番。
        """
        try:
            fans = MahjongFanCalculator(
                pack=tuple(self.packs[0]),
                hand=tuple(self.hand),
                winTile=winTile,
                flowerCount=0,
                isSelfDrawn=isSelfDrawn,
                is4thTile=(self.shownTiles[winTile] + isSelfDrawn) == 4,
                isAboutKong=isAboutKong,
                isWallLast=self.wallLast,
                seatWind=self.seatWind,
                prevalentWind=self.prevalentWind,
                verbose=True
            )
            fanCnt = sum(fanPoint * cnt for fanPoint, cnt, _, _ in fans)
            if fanCnt < 8:
                raise Exception('Not Enough Fans')
            return True
        except Exception:
            return False

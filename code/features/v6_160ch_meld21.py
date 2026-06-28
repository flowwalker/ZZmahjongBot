"""国标麻将特征提取器 — 160通道."""

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
    """国标麻将全量特征提取 Agent"""

    OBS_SIZE = 160
    ACT_SIZE = 235

    # 特征通道起始偏移量
    OFFSET_OBS = {
        'HAND': 0,         # 4 channels: 0-3
        'SHOWN': 4,        # 4 channels: 4-7
        'MELD': 8,         # 21 channels: 8-28
        'PREV_WIND': 29,   # 4 channels: 29-32
        'SEAT_WIND': 33,   # 4 channels: 33-36
        'WALL': 37,        # 4 channels: 37-40
        'DISCARD': 41,     # 112 channels: 41-152
        'CUR_TILE': 153,   # 1 channel:  当前待决策牌 one-hot
        'TILE_FROM': 154,  # 1 channel:  牌源方归一化标量
        'WALL_LAST': 155,  # 1 channel:  牌墙最后一张标志
        'IS_ABOUT_KONG': 156,  # 1 channel: 刚杠标志
        'HAND_SIZE': 157,  # 3 channels: 157-159 对手手牌数
    }

    # 动作类型偏移
    OFFSET_ACT = {
        'Pass': 0, 'Hu': 1, 'Play': 2, 'Chi': 36,
        'Peng': 99, 'Gang': 133, 'AnGang': 167, 'BuGang': 201
    }

    # 34种牌列表
    TILE_LIST = [
        *('W%d' % (i + 1) for i in range(9)),   # 万: W1-W9
        *('T%d' % (i + 1) for i in range(9)),   # 条: T1-T9
        *('B%d' % (i + 1) for i in range(9)),   # 筒: B1-B9
        *('F%d' % (i + 1) for i in range(4)),   # 风: F1-F4
        *('J%d' % (i + 1) for i in range(3))    # 箭: J1-J3
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

        # 对手手牌数追踪 (相对位置: 0=self, 1=下家, 2=对家, 3=上家)
        self.hand_sizes = [0, 0, 0, 0]

        # 初始化为 (160, 36) 的全零矩阵
        self.obs = np.zeros((self.OBS_SIZE, 36), dtype=np.float32)

    # 更新观测矩阵
    def _update_obs(self):
        """将所有状态映射到 160 通道"""
        self.obs.fill(0)

        # 1. 手牌编码 (Ch 0-3)
        for tile in set(self.hand):
            cnt = self.hand.count(tile)
            idx = self.OFFSET_TILE[tile]
            for i in range(min(cnt, 4)):
                self.obs[self.OFFSET_OBS['HAND'] + i, idx] = 1

        # 2. 全局可见牌编码 (Ch 4-7) = 手牌 + 场上明牌
        for tile in self.TILE_LIST:
            cnt = self.hand.count(tile) + self.shownTiles.get(tile, 0)
            cnt = min(cnt, 4)
            idx = self.OFFSET_TILE[tile]
            for i in range(cnt):
                self.obs[self.OFFSET_OBS['SHOWN'] + i, idx] = 1

        # 3. 鸣牌精细展开编码 (Ch 8-28)
        # 每人 5 通道: [0:吃左, 1:吃中, 2:吃右, 3:碰, 4:明杠]
        # 再加 1 通道用于自己暗杠
        for p in range(4):
            base_ch = self.OFFSET_OBS['MELD'] + p * 5
            for pkt, tile, offer in self.packs[p]:
                if pkt == 'CHI':
                    c, n = tile[0], int(tile[1])
                    # 注意: 原代码中 offer = called_number - n + 2, 实际取值为 1/2/3
                    # offer=1: 被叫牌是 n-1 (左吃, 序列最小)
                    # offer=2: 被叫牌是 n   (中吃, 序列中间)
                    # offer=3: 被叫牌是 n+1 (右吃, 序列最大)
                    chi_type = offer - 1  # 0=左吃, 1=中吃, 2=右吃
                    for dn in (-1, 0, 1):
                        i2 = self.OFFSET_TILE.get(c + str(n + dn))
                        if i2 is not None:
                            self.obs[base_ch + chi_type, i2] = 1
                elif pkt == 'PENG':
                    idx = self.OFFSET_TILE.get(tile)
                    if idx is not None: self.obs[base_ch + 3, idx] = 1
                elif pkt == 'GANG':
                    if offer == 0:  # 暗杠
                        if tile != 'CONCEALED':  # 别家暗杠对己方不可见，自己暗杠可见
                            idx = self.OFFSET_TILE.get(tile)
                            if idx is not None:
                                # 自己暗杠使用独立通道
                                if p == 0:
                                    self.obs[self.OFFSET_OBS['MELD'] + 20, idx] = 1
                    else:           # 明杠
                        idx = self.OFFSET_TILE.get(tile)
                        if idx is not None: self.obs[base_ch + 4, idx] = 1

        # 4. 风位编码 (Ch 29-36)
        self.obs[self.OFFSET_OBS['PREV_WIND'] + self.prevalentWind, :] = 1
        self.obs[self.OFFSET_OBS['SEAT_WIND'] + self.seatWind, :] = 1

        # 5. 剩余牌墙比例 (Ch 37-40)
        # 四家牌墙各自归一化
        for p in range(4):
            self.obs[self.OFFSET_OBS['WALL'] + p, :] = self.tileWall[p] / 21.0

        # 6. 四家弃牌时序 (Ch 41-152)
        for p in range(4):
            # 截取前28步弃牌记录
            for step, tile in enumerate(self.history[p][-28:]):
                idx = self.OFFSET_TILE.get(tile)
                if idx is not None:
                    # 每个玩家占据连续的28层，步数越小越早打出
                    self.obs[self.OFFSET_OBS['DISCARD'] + p * 28 + step, idx] = 1


        # 7. 当前待决策牌 one-hot (Ch 153)
        if self.curTile:
            idx = self.OFFSET_TILE.get(self.curTile)
            if idx is not None:
                self.obs[self.OFFSET_OBS['CUR_TILE'], idx] = 1.0

        # 8. 牌源方归一化标量 (Ch 154)
        self.obs[self.OFFSET_OBS['TILE_FROM'], :] = self.tileFrom / 3.0

        # 9. 牌墙最后一张标志 (Ch 155)
        self.obs[self.OFFSET_OBS['WALL_LAST'], :] = 1.0 if self.wallLast else 0.0

        # 10. 刚杠标志 (Ch 156)
        self.obs[self.OFFSET_OBS['IS_ABOUT_KONG'], :] = 1.0 if self.isAboutKong else 0.0

        # 11. 对手手牌数归一化 (Ch 157-159, 相对位置 1,2,3)
        for i in range(1, 4):
            self.obs[self.OFFSET_OBS['HAND_SIZE'] + i - 1, :] = self.hand_sizes[i] / 14.0

    def _make_obs(self) -> dict:
        """构建最终输出观测"""
        mask = np.zeros(self.ACT_SIZE, dtype=np.float32)
        for a in self.valid:
            mask[a] = 1
        return {
            'observation': self.obs.reshape((self.OBS_SIZE, 4, 9)).copy(),
            'action_mask': mask
        }

    #  Request 解析 (Botzone → 内部状态)
    def request2obs(self, request: str):
        t = request.split()

        if t[0] == 'Wind':
            self.prevalentWind = int(t[1])
            self._update_obs()
            return None

        if t[0] == 'Deal':
            self.hand = t[1:]
            self.hand_sizes = [len(self.hand), 13, 13, 13]
            self._update_obs()
            return None

        if t[0] in ('Huang',):
            self.valid = []
            return self._make_obs()

        if t[0] == 'Draw':
            self.tileWall[0] -= 1
            self.wallLast = self.tileWall[1] == 0
            tile = t[1]
            self.curTile = tile
            self.valid = []

            if self._can_hu(tile, isSelfDrawn=True, isAboutKong=self.isAboutKong):
                self.valid.append(self.OFFSET_ACT['Hu'])

            self.hand.append(tile)
            self.hand_sizes[0] = len(self.hand)

            for _t in set(self.hand):
                self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[_t])
                if (self.hand.count(_t) == 4 and not self.wallLast and self.tileWall[0] > 0):
                    self.valid.append(self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[_t])

            if not self.wallLast and self.tileWall[0] > 0:
                for packType, ptile, offer in self.packs[0]:
                    if packType == 'PENG' and ptile == self.curTile:
                        self.valid.append(self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[ptile])

            self._update_obs()
            self.isAboutKong = False  # 必须在 _update_obs() 之后清除，确保 ch156 编码正确
            return self._make_obs()

        p = (int(t[1]) + 4 - self.seatWind) % 4

        if t[2] == 'Draw':
            self.tileWall[p] -= 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            self.hand_sizes[p] += 1
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
                self.hand_sizes[0] = len(self.hand)
                self._update_obs()
                return None
            else:
                self.hand_sizes[p] -= 1
                self.valid = []
                if self._can_hu(self.curTile):
                    self.valid.append(self.OFFSET_ACT['Hu'])

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
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].append(('CHI', tile, int(self.curTile[1]) - num + 2))
            self.shownTiles[self.curTile] -= 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] += 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0

            if p == 0:
                self.valid = []
                self.hand.append(self.curTile)
                for i in range(-1, 2):
                    self.hand.remove(color + str(num + i))
                self.hand_sizes[0] = len(self.hand)
                for _t in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[_t])
                self._update_obs()
                return self._make_obs()
            else:
                self.hand_sizes[p] -= 2
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
                self.hand_sizes[0] = len(self.hand)
                self._update_obs()
            else:
                self.hand_sizes[p] += 2
            return None

        if t[2] == 'Peng':
            self.packs[p].append(('PENG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 2
            self.wallLast = self.tileWall[(p + 1) % 4] == 0

            if p == 0:
                self.valid = []
                for _ in range(2):
                    self.hand.remove(self.curTile)
                self.hand_sizes[0] = len(self.hand)
                for _t in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[_t])
                self._update_obs()
                return self._make_obs()
            else:
                self.hand_sizes[p] -= 2
            return None

        if t[2] == 'UnPeng':
            self.packs[p].pop()
            self.shownTiles[self.curTile] -= 2
            if p == 0:
                for _ in range(2):
                    self.hand.append(self.curTile)
                self.hand_sizes[0] = len(self.hand)
                self._update_obs()
            else:
                self.hand_sizes[p] += 2
            return None

        if t[2] == 'Gang':
            self.packs[p].append(('GANG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 3
            if p == 0:
                for _ in range(3):
                    self.hand.remove(self.curTile)
                self.isAboutKong = True
                self.hand_sizes[0] = len(self.hand)
                self._update_obs()
            else:
                self.hand_sizes[p] -= 3
            return None

        if t[2] == 'AnGang':
            tile = t[3] if (p == 0 and len(t) > 3) else 'CONCEALED'
            self.packs[p].append(('GANG', tile, 0))
            if p == 0:
                self.isAboutKong = True
                for _ in range(4):
                    self.hand.remove(tile)
                self.hand_sizes[0] = len(self.hand)
            else:
                self.isAboutKong = False
                self.hand_sizes[p] -= 4
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
                self.hand.remove(tile)
                self.isAboutKong = True
                self.hand_sizes[0] = len(self.hand)
                self._update_obs()
                return None
            else:
                self.hand_sizes[p] -= 1
                # 别人补杠：记录 curTile，但不设置自己的 isAboutKong
                self.curTile = tile
                self.isAboutKong = False  # 自己没有杠，不能污染后续摸牌的杠上开花判定
                self.valid = []
                if self._can_hu(tile, isSelfDrawn=False, isAboutKong=True):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                self.valid.append(self.OFFSET_ACT['Pass'])
                self._update_obs()
                return self._make_obs()

        raise NotImplementedError('Unknown request: %s' % request)

    def action2response(self, action: int) -> str:
        if action < self.OFFSET_ACT['Hu']: return 'Pass'
        if action < self.OFFSET_ACT['Play']: return 'Hu'
        if action < self.OFFSET_ACT['Chi']: return 'Play ' + self.TILE_LIST[action - self.OFFSET_ACT['Play']]
        if action < self.OFFSET_ACT['Peng']:
            t = (action - self.OFFSET_ACT['Chi']) // 3
            return 'Chi ' + 'WTB'[t // 7] + str(t % 7 + 2)
        if action < self.OFFSET_ACT['Gang']: return 'Peng'
        if action < self.OFFSET_ACT['AnGang']: return 'Gang'
        if action < self.OFFSET_ACT['BuGang']: return 'Gang ' + self.TILE_LIST[action - self.OFFSET_ACT['AnGang']]
        return 'BuGang ' + self.TILE_LIST[action - self.OFFSET_ACT['BuGang']]

    def response2action(self, response: str) -> int:
        t = response.split()
        if t[0] == 'Pass': return self.OFFSET_ACT['Pass']
        if t[0] == 'Hu': return self.OFFSET_ACT['Hu']
        if t[0] == 'Play': return self.OFFSET_ACT['Play'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Chi':
            return (self.OFFSET_ACT['Chi'] + 'WTB'.index(t[1][0]) * 7 * 3 + (int(t[2][1]) - 2) * 3 + int(t[1][1]) - int(t[2][1]) + 1)
        if t[0] == 'Peng': return self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Gang': return self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'AnGang': return self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'BuGang': return self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[t[1]]
        return self.OFFSET_ACT['Pass']

    def _can_hu(self, winTile: str, isSelfDrawn: bool = False, isAboutKong: bool = False) -> bool:
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
            if sum(fanPoint * cnt for fanPoint, cnt, _, _ in fans) < 8:
                raise Exception('Not Enough Fans')
            return True
        except Exception:
            return False

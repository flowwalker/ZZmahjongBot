"""国标麻将 Botzone Bot — 神经网络直觉 + FlatMC搜索 混合版 (baseline-v3)"""

import os
# 限制底层的多线程数量（防止 Botzone 沙盒线程竞争导致超时）
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
import torch
torch.set_num_threads(1)

from nn_policy import NNEvaluator
from gamestate import GameState


def main():
    model_path = os.path.join(os.path.dirname(__file__), 'model_cnn_Res_Mish_SE_standard.pt')
    nn = NNEvaluator(model_path=model_path)

    gs = None
    seatWind = None
    angang = None
    zimo = False

    input()  # 丢弃首行 (请求计数)
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
            gs = GameState(seatWind, nn_evaluator=nn)
            gs.prevalentWind = int(t[2])
            print('PASS')

        elif t[0] == '1':
            gs.hand = t[5:5 + 13]
            print('PASS')

        elif t[0] == '2':
            tile = t[1]
            gs.tileWall[0] -= 1
            gs.wallLast = gs.tileWall[1] == 0

            # 检查胡
            gs.hand.append(tile)
            if gs.can_hu(tile, isSelfDrawn=True, isAboutKong=gs.isAboutKong):
                print('HU')
                print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
                sys.stdout.flush()
                break

            gs.isAboutKong = False

            # 暗杠
            did_angang = False
            for t_card in set(gs.hand):
                if (gs.hand.count(t_card) == 4
                        and not gs.wallLast
                        and gs.tileWall[0] > 0):
                    s0 = gs.shanten()
                    saved_h = list(gs.hand)
                    saved_p = list(gs.packs[0])
                    for _ in range(4):
                        gs.hand.remove(t_card)
                    gs.packs[0].append(('GANG', t_card, 0))
                    s1 = gs.shanten()
                    gs.hand = saved_h
                    gs.packs[0] = saved_p
                    if s1 <= s0:
                        for _ in range(4):
                            gs.hand.remove(t_card)
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

            # 补杠
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

            # 正常出牌 (NN+FlatMC best_discard)
            best = gs.best_discard()
            gs.hand.remove(best)
            print('PLAY %s' % best)

        elif t[0] == '3':
            p = int(t[1])
            rp = (p + 4 - gs.seatWind) % 4

            if t[2] == 'DRAW':
                gs.tileWall[rp] -= 1
                gs.wallLast = gs.tileWall[(rp + 1) % 4] == 0
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

            else:  # PLAY / CHI / PENG
                zimo = False

                if t[2] == 'CHI':
                    tile = t[3]
                    color, num = tile[0], int(tile[1])
                    gs.packs[rp].append(('CHI', tile, int(gs.curTile[1]) - num + 2))
                    gs.shownTiles[gs.curTile] -= 1
                    for i in range(-1, 2):
                        gs.shownTiles[color + str(num + i)] += 1
                elif t[2] == 'PENG':
                    gs.packs[rp].append(('PENG', gs.curTile, (4 + rp - gs.tileFrom) % 4))
                    gs.shownTiles[gs.curTile] += 2

                # 处理打出的牌
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

                    # 碰牌 (NN-guided)
                    if gs.hand.count(gs.curTile) >= 2:
                        ok, disc = gs.should_peng(gs.curTile, rp)
                        if ok and disc:
                            for _ in range(2):
                                gs.hand.remove(gs.curTile)
                            gs.packs[0].append(('PENG', gs.curTile,
                                                (4 + rp - gs.tileFrom) % 4))
                            print('PENG %s' % disc)
                            gs.hand.extend([gs.curTile, gs.curTile, disc])
                            gs.packs[0].pop()
                            print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
                            sys.stdout.flush()
                            continue

                    # 明杠
                    if gs.hand.count(gs.curTile) == 3 and gs.tileWall[0] > 0:
                        if gs.should_gang(gs.curTile, rp):
                            print('GANG')
                            print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
                            sys.stdout.flush()
                            continue

                    # 吃牌 (仅上家 rp==3, NN-guided)
                    if rp == 3 and gs.curTile[0] in 'WTB':
                        c, n = gs.curTile[0], int(gs.curTile[1])
                        best_chi, best_s = None, 999
                        for mn in (n - 1, n, n + 1):
                            if mn < 1 or mn > 9:
                                continue
                            mid = c + str(mn)
                            if mn < n:
                                nd = [c + str(mn + 1), mid]
                            elif mn > n:
                                nd = [c + str(mn - 1), mid]
                            else:
                                nd = [c + str(n - 1), c + str(n + 1)]
                            if not all(card in gs.hand for card in nd):
                                continue
                            sh, sp = list(gs.hand), list(gs.packs[0])
                            gs.hand.append(gs.curTile)
                            for card in nd:
                                gs.hand.remove(card)
                            gs.packs[0].append(('CHI', mid, int(gs.curTile[1]) - mn + 2))
                            s = gs.shanten()
                            gs.hand, gs.packs[0] = sh, sp
                            if s < best_s:
                                best_s = s
                                best_chi = mid
                        if best_chi:
                            ok, disc = gs.should_chi(best_chi, gs.curTile, best_s)
                            if ok and disc:
                                gs.hand.append(gs.curTile)
                                for dn in (-1, 0, 1):
                                    card = c + str(int(best_chi[1]) + dn)
                                    if card in gs.hand and card != gs.curTile:
                                        gs.hand.remove(card)
                                gs.packs[0].append(('CHI', best_chi,
                                                    int(gs.curTile[1]) - int(best_chi[1]) + 2))
                                print('CHI %s %s' % (best_chi, disc))
                                gs.hand.remove(gs.curTile)
                                for dn in (-1, 0, 1):
                                    card = c + str(int(best_chi[1]) + dn)
                                    if card != gs.curTile:
                                        gs.hand.append(card)
                                gs.hand.append(disc)
                                gs.packs[0].pop()
                                print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
                                sys.stdout.flush()
                                continue

                    print('PASS')

        print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
        sys.stdout.flush()


if __name__ == '__main__':
    main()

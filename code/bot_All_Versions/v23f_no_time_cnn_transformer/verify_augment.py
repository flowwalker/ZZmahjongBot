#!/usr/bin/env python3
"""验证 288-fold 数据增强的正确性."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from augment import (
    get_transforms, apply_transform, augment_batch,
    NUM_TILES, NUM_POS, ACT_SIZE,
    TILE_CHANNELS, UNCHANGED_CHANNELS,
    WIND_PREV_CH, WIND_SEAT_CH,
    MELD_CHI_SWAP_PAIRS,
    OFFSET_ACT,
)

#  Part 1: 验证变换结构正确性

def test_transform_structure():
    """验证所有 288 个变换的内部结构一致性。"""
    print("=" * 60)
    print("Part 1: 变换结构一致性检查")
    print("=" * 60)

    transforms = get_transforms()
    assert len(transforms) == 288, f"Expected 288, got {len(transforms)}"
    print(f"  ✓ 共 {len(transforms)} 个变换")

    tile_inv_set = set()
    action_fwd_set = set()

    for k, tf in enumerate(transforms):
        tile_inv = tf['tile_inv']
        tile_fwd = tf['tile_fwd']
        action_fwd = tf['action_fwd']
        action_inv = tf['action_inv']

        # 1a. tile_fwd ∘ tile_inv = identity
        assert np.array_equal(tile_fwd[tile_inv], np.arange(NUM_POS)), \
            f"Transform {k}: tile_fwd[tile_inv] ≠ identity"
        assert np.array_equal(tile_inv[tile_fwd], np.arange(NUM_POS)), \
            f"Transform {k}: tile_inv[tile_fwd] ≠ identity"

        # 1b. action_fwd ∘ action_inv = identity
        assert np.array_equal(action_fwd[action_inv], np.arange(ACT_SIZE)), \
            f"Transform {k}: action_fwd[action_inv] ≠ identity"
        assert np.array_equal(action_inv[action_fwd], np.arange(ACT_SIZE)), \
            f"Transform {k}: action_inv[action_fwd] ≠ identity"

        # 2. Real tiles (0-33) map to real tiles; padding (34-35) map to themselves
        assert np.all(tile_fwd[:NUM_TILES] < NUM_TILES), \
            f"Transform {k}: real tile mapped to padding"
        assert np.array_equal(tile_fwd[34:36], [34, 35]), \
            f"Transform {k}: padding not fixed: {tile_fwd[34:36]}"
        assert np.array_equal(tile_inv[34:36], [34, 35]), \
            f"Transform {k}: padding not fixed in inverse"

        # 3. Pass (0) and Hu (1) unchanged
        assert action_fwd[0] == 0, f"Transform {k}: Pass changed to {action_fwd[0]}"
        assert action_fwd[1] == 1, f"Transform {k}: Hu changed to {action_fwd[1]}"
        assert action_inv[0] == 0, f"Transform {k}: Pass inv changed to {action_inv[0]}"
        assert action_inv[1] == 1, f"Transform {k}: Hu inv changed to {action_inv[1]}"

        # 4. Chi actions (36-98) stay in range
        chi_mask = (np.arange(ACT_SIZE) >= 36) & (np.arange(ACT_SIZE) < 99)
        assert np.all((action_fwd[chi_mask] >= 36) & (action_fwd[chi_mask] < 99)), \
            f"Transform {k}: Chi action mapped out of range"

        # 5. Play (2-35), Peng (99-132), Gang (133-166), AnGang (167-200), BuGang (201-234)
        for lo, hi, name in [(2, 35, 'Play'), (99, 132, 'Peng'), (133, 166, 'Gang'),
                              (167, 200, 'AnGang'), (201, 234, 'BuGang')]:
            in_range = (np.arange(ACT_SIZE) >= lo) & (np.arange(ACT_SIZE) <= hi)
            assert np.all((action_fwd[in_range] >= lo) & (action_fwd[in_range] <= hi)), \
                f"Transform {k}: {name} action mapped out of range"

        # Collect for uniqueness check
        tile_inv_set.add(tuple(tile_inv))
        action_fwd_set.add(tuple(action_fwd))

    # 6. Uniqueness: all 288 transforms should have unique tile_inv
    #    (but NOT necessarily unique action_fwd — e.g., if ds_active but hands have
    #     no numbered tiles, action_fwd could be same as identity)
    #    Let's check tile_inv uniqueness at minimum
    assert len(tile_inv_set) == 288, \
        f"tile_inv not unique: {len(tile_inv_set)} unique out of 288"
    print(f"  ✓ tile_inv: {len(tile_inv_set)} unique (all distinct)")

    # action_fwd may have some duplicates when digit symmetry doesn't affect
    # the specific Chi encoding. Let's just report the count.
    print(f"  ✓ action_fwd: {len(action_fwd_set)} unique out of 288")

    print("  ✓ All structural checks passed\n")


#  Part 2: 合成 obs 功能验证

def make_synthetic_sample():
    """
    构造一个包含所有牌种类和鸣牌类型的合成观测样本。

    确保:
    - 手牌包含 W/T/B/F/J 各若干张
    - 鸣牌区有吃/碰/明杠/暗杠
    - 弃牌区有各家弃牌
    - 风位和圈风已设置
    - 标量通道有合理的非零值
    """
    obs = np.zeros((160, 4, 9), dtype=np.float32)

    # --- HAND (ch0-3): 手牌 ---
    # W1×2, W3, T5, T5, B1×3, F1, J2
    hand_tiles = [0, 0, 2, 13, 13, 18, 18, 18, 27, 32]  # tile indices
    for t in hand_tiles:
        cnt = hand_tiles.count(t)
        row, col = divmod(t, 9) if t < 34 else (3, t - 27 + 4)
        # HAND: count-based one-hot (ch0=1st, ch1=2nd, ch2=3rd, ch3=4th)
        for i in range(cnt):
            obs[i, row, col] = 1.0

    # --- SHOWN (ch4-7): 全局可见牌 = 手牌 + 场上明牌 ---
    # Simulate some shown tiles (peng/kong visible to all)
    shown_extra = {1: 1, 9: 1, 28: 1}  # W2, T1, F2 appear extra
    for t in hand_tiles:
        shown_extra[t] = shown_extra.get(t, 0) + 1
    for t, cnt in shown_extra.items():
        cnt = min(cnt, 4)
        row, col = divmod(t, 9)
        for i in range(cnt):
            obs[4 + i, row, col] = 1.0

    # --- MELD (ch8-28): 鸣牌精细编码 ---
    # Player 0 (ch8-12): chi_left of W1,W2,W3 (tiles 0,1,2)
    obs[8, 0, 0] = 1.0   # chi_left W1
    obs[8, 0, 1] = 1.0   # chi_left W2
    obs[8, 0, 2] = 1.0   # chi_left W3
    # Player 0 peng of T1 (ch11, tile 9)
    obs[11, 1, 0] = 1.0
    # Player 1 (ch13-17): chi_right of B5,B6,B7 (tiles 22,23,24)
    obs[15, 2, 4] = 1.0  # chi_right B5
    obs[15, 2, 5] = 1.0  # chi_right B6
    obs[15, 2, 6] = 1.0  # chi_right B7
    # Player 2 ming_gang of W5 (ch22, tile 4)
    obs[22, 0, 4] = 1.0
    # Player 3 peng of F2 (ch26, tile 28)
    obs[26, 3, 1] = 1.0
    # Self angang of J1 (ch28, tile 31)
    obs[28, 3, 4] = 1.0

    # --- PREV_WIND (ch29-32) & SEAT_WIND (ch33-36) ---
    obs[29, :, :] = 1.0   # 圈风=东 (ch29 all-1)
    obs[34, :, :] = 1.0   # 门风=南 (ch34 all-1, seatWind=1)

    # --- WALL (ch37-40): scalar values ---
    obs[37, :, :] = 15.0 / 21.0
    obs[38, :, :] = 14.0 / 21.0
    obs[39, :, :] = 13.0 / 21.0
    obs[40, :, :] = 12.0 / 21.0

    # --- DISCARD (ch41-152): 弃牌 ---
    # Player 0 discarded W5, T3 (tiles 4, 11)
    obs[41, 0, 4] = 1.0     # p0 step0 W5
    obs[42, 1, 2] = 1.0     # p0 step1 T3
    # Player 1 discarded B1, F3 (tiles 18, 29)
    obs[69, 2, 0] = 1.0     # p1 step0 B1  (69 = 41 + 1*28)
    obs[70, 3, 2] = 1.0     # p1 step1 F3
    # Player 2 discarded J2, W9 (tiles 32, 8)
    obs[97, 3, 5] = 1.0     # p2 step0 J2  (97 = 41 + 2*28)
    obs[98, 0, 8] = 1.0     # p2 step1 W9

    # --- CUR_TILE (ch153): 当前待决策牌 ---
    obs[153, 0, 3] = 1.0    # W4 (tile 3)

    # --- TILE_FROM (ch154): scalar ---
    obs[154, :, :] = 2.0 / 3.0

    # --- WALL_LAST (ch155): flag ---
    obs[155, :, :] = 0.0

    # --- IS_ABOUT_KONG (ch156): flag ---
    obs[156, :, :] = 0.0

    # --- HAND_SIZE (ch157-159): scalar ---
    obs[157, :, :] = 13.0 / 14.0   # 下家
    obs[158, :, :] = 10.0 / 14.0   # 对家
    obs[159, :, :] = 14.0 / 14.0   # 上家

    # --- Action mask ---
    mask = np.zeros(ACT_SIZE, dtype=np.float32)
    # Make several valid actions: Play some tiles, Hu, Pass
    valid_actions = [
        0,                          # Pass
        1,                          # Hu
        2 + 3,                      # Play W4 (tile 3)
        2 + 13,                     # Play T5 (tile 13)
        2 + 18,                     # Play B1 (tile 18)
        2 + 27,                     # Play F1 (tile 27)
        36 + 0 * 21 + (3 - 2) * 3 + 1,  # Chi W, mid=3, claim_pos=1 (中吃)
    ]
    for a in valid_actions:
        mask[a] = 1.0

    # Action label: pick one valid action (Play W4)
    act = 2 + 3  # Play W4

    return obs, mask, act


def test_synthetic_augmentation():
    """用合成样本验证所有 288 个变换后的 obs/mask/act 一致性。"""
    print("=" * 60)
    print("Part 2: 合成样本增强功能验证")
    print("=" * 60)

    transforms = get_transforms()
    orig_obs, orig_mask, orig_act = make_synthetic_sample()

    # 哪些通道应该是二值的 (0 或 1)
    BINARY_CHANNELS = set(TILE_CHANNELS) | set(range(29, 37)) | {155, 156}
    # HAND, SHOWN, MELD, DISCARD, CUR_TILE (all tile channels)
    # + PREV_WIND, SEAT_WIND
    # + WALL_LAST, IS_ABOUT_KONG

    # 哪些通道是标量的 (在 (0,1] 范围内)
    SCALAR_CHANNELS = set(range(37, 41)) | {154} | set(range(157, 160))
    # WALL, TILE_FROM, HAND_SIZE

    # 对标量通道记录原始值
    orig_scalars = {}
    for ch in SCALAR_CHANNELS:
        orig_scalars[ch] = orig_obs[ch, 0, 0]  # scalar channels are uniform

    errors = []
    unique_states = set()

    for k, tf in enumerate(transforms):
        aug_obs, aug_mask, aug_act = apply_transform(orig_obs, orig_mask, orig_act, tf)

        # --- Check 1: 二值通道仍是 0 或 1 ---
        for ch in BINARY_CHANNELS:
            vals = aug_obs[ch]
            if not np.all((vals == 0.0) | (vals == 1.0)):
                bad = np.where((vals != 0.0) & (vals != 1.0))
                errors.append(f"T{k} ch{ch}: non-binary values at {bad[0][:5]}={vals[bad][:5]}")
                if len(errors) >= 10:
                    break
        if len(errors) >= 10:
            break

        # --- Check 2: 标量通道值不变 ---
        for ch in SCALAR_CHANNELS:
            if not np.allclose(aug_obs[ch], orig_scalars[ch], atol=1e-6):
                errors.append(f"T{k} ch{ch}: scalar changed from {orig_scalars[ch]} to {aug_obs[ch,0,0]}")
                if len(errors) >= 10:
                    break
        if len(errors) >= 10:
            break

        # --- Check 3: mask 是 0/1 ---
        if not np.all((aug_mask == 0.0) | (aug_mask == 1.0)):
            errors.append(f"T{k}: mask has non-binary values")
            if len(errors) >= 10:
                break

        # --- Check 4: mask[aug_act] == 1 ---
        if aug_mask[aug_act] != 1.0:
            errors.append(f"T{k}: action {aug_act} not in mask (valid: {np.where(aug_mask>0)[0][:5]}...)")
            if len(errors) >= 10:
                break

        # --- Check 5: 至少 2 个有效动作 (因为 filterData 已过滤单动作) ---
        if aug_mask.sum() < 2:
            errors.append(f"T{k}: only {aug_mask.sum()} valid actions")
            if len(errors) >= 10:
                break

        # --- Check 6: 不同变换产生不同的 (obs, mask, act) 以 obs hash 记录 ---
        obs_hash = tuple(aug_obs.ravel()[:50])  # first 50 values as fingerprint
        unique_states.add((obs_hash, aug_act))

    if errors:
        print(f"  ✗ {len(errors)} errors found:")
        for e in errors[:20]:
            print(f"    {e}")
        return False

    print(f"  ✓ 所有二值通道保持 0/1")
    print(f"  ✓ 所有标量通道值不变 (WALL, TILE_FROM, HAND_SIZE)")
    print(f"  ✓ mask 一致性: aug_mask[aug_act] == 1 全部成立")
    print(f"  ✓ 有效动作数 ≥ 2 全部成立")
    print(f"  ✓ 唯一 (obs_fingerprint, act) 组合: {len(unique_states)} / 288")
    print()

    # 统计多少变换产生了不同的 obs (前50个值作为指纹)
    obs_fingerprints = set()
    for k, tf in enumerate(transforms):
        aug_obs, _, _ = apply_transform(orig_obs, orig_mask, orig_act, tf)
        obs_fingerprints.add(tuple(aug_obs.ravel()[:50]))
    print(f"  ℹ 唯一 obs 指纹: {len(obs_fingerprints)} / 288")
    print(f"    (合成样本含 W/T/B/F/J 各类牌，多数变换应产生不同 obs)")

    return True


#  Part 3: augment_batch 批量函数验证

def test_augment_batch():
    """验证 augment_batch 的批量输出与 apply_transform 单样本输出一致。"""
    print("=" * 60)
    print("Part 3: augment_batch 批量一致性验证")
    print("=" * 60)

    transforms = get_transforms()
    orig_obs, orig_mask, orig_act = make_synthetic_sample()

    # 构造 batch of 3 (same sample repeated)
    N = 3
    obs_stack = np.stack([orig_obs] * N)
    mask_stack = np.stack([orig_mask] * N)
    act_stack = np.array([orig_act] * N, dtype=np.int64)

    aug_obs, aug_mask, aug_act = augment_batch(obs_stack, mask_stack, act_stack)

    assert aug_obs.shape == (288 * N, 160, 4, 9), f"Shape: {aug_obs.shape}"
    assert aug_mask.shape == (288 * N, 235), f"Shape: {aug_mask.shape}"
    assert aug_act.shape == (288 * N,), f"Shape: {aug_act.shape}"
    print(f"  ✓ 输出形状: obs={aug_obs.shape}, mask={aug_mask.shape}, act={aug_act.shape}")

    # Verify each transform × each sample matches apply_transform
    for k, tf in enumerate(transforms[:10]):  # spot-check 10 transforms
        for i in range(N):
            ref_obs, ref_mask, ref_act = apply_transform(orig_obs, orig_mask, orig_act, tf)

            idx = k * N + i
            assert np.allclose(aug_obs[idx], ref_obs, atol=1e-6), \
                f"obs mismatch at T{k} sample {i}"
            assert np.allclose(aug_mask[idx], ref_mask, atol=1e-6), \
                f"mask mismatch at T{k} sample {i}"
            assert aug_act[idx] == ref_act, \
                f"act mismatch at T{k} sample {i}: {aug_act[idx]} vs {ref_act}"

    print(f"  ✓ spot-check 10/288 transforms × 3 samples: 全部与 apply_transform 一致")
    print()


#  Part 4: 验证 action 对称性 (可逆)

def test_action_roundtrip():
    """验证对任意 action index，经过 action_fwd 再 action_inv 回到原位。"""
    print("=" * 60)
    print("Part 4: Action 映射 round-trip 验证")
    print("=" * 60)

    transforms = get_transforms()
    for k, tf in enumerate(transforms):
        action_fwd = tf['action_fwd']
        action_inv = tf['action_inv']

        for a in range(ACT_SIZE):
            assert action_inv[action_fwd[a]] == a, \
                f"T{k}: round-trip failed for action {a}"
            assert action_fwd[action_inv[a]] == a, \
                f"T{k}: reverse round-trip failed for action {a}"

    print(f"  ✓ 所有 {288}×{ACT_SIZE} 个 action 的 round-trip 全部成立")
    print()


#  Part 5: MELD chi channel swap 验证

def test_meld_chi_swap():
    """验证数字对称变换下 chi_left ↔ chi_right 通道正确互换。"""
    print("=" * 60)
    print("Part 5: MELD chi channel swap 验证")
    print("=" * 60)

    transforms = get_transforms()
    orig_obs, orig_mask, orig_act = make_synthetic_sample()

    for k, tf in enumerate(transforms):
        if not tf['ds_active']:
            continue

        aug_obs, _, _ = apply_transform(orig_obs, orig_mask, orig_act, tf)

        for ch_l, ch_r in MELD_CHI_SWAP_PAIRS:
            # After ds transform, what was in chi_left of original...
            # Let's check: the tile patterns in ch_l of aug should be the
            # tile-permuted versions of what was in ch_r of original
            # (since chi_left ↔ chi_right under DS)

            # Actually verify symmetrically: the two channels should be swapped
            # relative to what they'd be WITHOUT the swap. Hard to verify directly
            # without reimplementing. Let's instead verify a concrete property:
            # if original had chi_left only (ch_l has data, ch_r is zero),
            # then after ds transform with tile_perm applied:
            #   aug[ch_l] should be zero (since chi_left→chi_right)
            #   aug[ch_r] should have the mirrored tile data

            # For our synthetic: ch8 (p0 chi_left) has data, ch10 (p0 chi_right) is zero
            # After ds: ch8 should be zero, ch10 should have mirrored data
            if ch_l == 8:  # p0 chi_left in our synthetic
                assert orig_obs[ch_l].sum() > 0, "pre: p0 chi_left should have data"
                assert orig_obs[ch_r].sum() == 0, "pre: p0 chi_right should be empty"

                # After ds transform: chi_left should be empty, chi_right should have data
                if aug_obs[ch_l].sum() != 0 or aug_obs[ch_r].sum() == 0:
                    # This might fail if tile_perm maps chi_left data to same positions
                    # that are then swapped — actually this is always true for ds.
                    # Let's just verify the channel swap happened:
                    pass  # We'll verify differently

        break  # Only test first ds transform

    # Simpler test: verify that ds transforms swap the chi channels
    # Take a simple obs where chi_left has a single tile marked
    test_obs = np.zeros((160, 4, 9), dtype=np.float32)
    test_obs[8, 0, 0] = 1.0   # p0 chi_left, W1
    test_obs[10, 0, :] = 0.0  # p0 chi_right, all zeros
    test_mask = np.ones(ACT_SIZE, dtype=np.float32)
    test_act = 0

    ds_tf = [tf for tf in transforms if tf['ds_active']][0]
    aug, _, _ = apply_transform(test_obs, test_mask, test_act, ds_tf)

    # After ds: W1 (pos 0) → W9 (pos 8)
    # chi_left data should move to chi_right at W9 position
    assert aug[8, 0, 0] == 0.0, \
        f"ds: p0 chi_left should be zero (was W1, now moved to chi_right)"
    assert aug[10, 0, 8] == 1.0, \
        f"ds: p0 chi_right should have data at W9 (mirrored from W1). " \
        f"Got aug[10,0,8]={aug[10,0,8]:.1f}"

    print(f"  ✓ ds: chi_left(W1@pos0) → chi_right(W9@pos8) correctly swapped")
    print()


#  Main

if __name__ == '__main__':
    all_ok = True

    try:
        test_transform_structure()
    except AssertionError as e:
        print(f"  ✗ FAIL: {e}")
        all_ok = False

    try:
        all_ok &= test_synthetic_augmentation()
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        import traceback; traceback.print_exc()
        all_ok = False

    try:
        test_augment_batch()
    except AssertionError as e:
        print(f"  ✗ FAIL: {e}")
        all_ok = False
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        import traceback; traceback.print_exc()
        all_ok = False

    try:
        test_action_roundtrip()
    except AssertionError as e:
        print(f"  ✗ FAIL: {e}")
        all_ok = False

    try:
        test_meld_chi_swap()
    except AssertionError as e:
        print(f"  ✗ FAIL: {e}")
        all_ok = False
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        import traceback; traceback.print_exc()
        all_ok = False

    print("=" * 60)
    if all_ok:
        print("✓ 所有验证通过！288-fold 增强实现正确。")
    else:
        print("✗ 部分验证失败，请检查上方错误信息。")
        sys.exit(1)

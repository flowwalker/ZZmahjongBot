"""288-fold 数据增强 (国标麻将)."""

import numpy as np
from itertools import permutations as iter_permutations

#  Constants
NUM_TILES = 34      # W1-W9, T1-T9, B1-B9, F1-F4, J1-J3
NUM_POS = 36        # 34 tiles + 2 padding positions (row 3, cols 7-8)
ACT_SIZE = 235

# Obs channel groups
TILE_CHANNELS = (
    list(range(0, 4))    # HAND     ch0-3
    + list(range(4, 8))  # SHOWN    ch4-7
    + list(range(8, 29)) # MELD     ch8-28 (21 channels: 4p×5 + self angang ch28)
    + list(range(41, 153)) # DISCARD ch41-152 (4p×28 steps)
    + [153]              # CUR_TILE ch153
)  # 4+4+21+112+1 = 142 channels

UNCHANGED_CHANNELS = sorted(set(range(160)) - set(TILE_CHANNELS))
# ch29-32 PREV_WIND, ch33-36 SEAT_WIND  → rotated separately
# ch37-40 WALL, ch154 TILE_FROM, ch155 WALL_LAST, ch156 IS_ABOUT_KONG,
# ch157-159 HAND_SIZE → untouched scalar channels

WIND_PREV_CH = slice(29, 33)   # 4 channels
WIND_SEAT_CH = slice(33, 37)   # 4 channels

# MELD chi channel pairs to swap under digital symmetry
# ch8-12: p0 chi_L(8), chi_M(9), chi_R(10), peng(11), gang(12)
# ch13-17: p1 ...
# ch18-22: p2 ...
# ch23-27: p3 ...
# ch28: p0 angang (independent)
MELD_CHI_SWAP_PAIRS = [(8 + p * 5, 8 + p * 5 + 2) for p in range(4)]
# [(8,10), (13,15), (18,20), (23,25)]

# Action offsets (matching feature.py OFFSET_ACT)
OFFSET_ACT = {
    'Pass': 0, 'Hu': 1, 'Play': 2, 'Chi': 36,
    'Peng': 99, 'Gang': 133, 'AnGang': 167, 'BuGang': 201,
}

# Cache for all 288 transforms (lazy init)
_ALL_TRANSFORMS = None


#  Base inverse tile maps (tile_map[new_pos] = old_pos)

def _digital_symmetry_maps():
    """2 maps: identity, and mirror about 5 within each numbered suit (W/T/B)."""
    identity = np.arange(NUM_POS, dtype=np.int64)
    mirrored = identity.copy()
    for start in (0, 9, 18):               # W, T, B start positions
        for i in range(9):
            mirrored[start + i] = start + (8 - i)   # 1↔9, 2↔8, 3↔7, 4↔6, 5↔5
    return [identity, mirrored]


def _suit_maps():
    """6 maps: all permutations of the three numbered suits W/T/B."""
    identity = np.arange(NUM_POS, dtype=np.int64)
    results = []
    suit_starts = [0, 9, 18]               # W, T, B start positions
    for perm in iter_permutations(range(3)):
        m = identity.copy()
        for old_suit, new_suit in enumerate(perm):
            m[suit_starts[new_suit]:suit_starts[new_suit] + 9] = \
                np.arange(suit_starts[old_suit], suit_starts[old_suit] + 9)
        results.append(m)
    return results


def _honor_maps():
    """6 maps: all permutations of J1/J2/J3 (中/发/白)."""
    identity = np.arange(NUM_POS, dtype=np.int64)
    results = []
    base = 31                              # J1 at pos 31
    for perm in iter_permutations(range(3)):
        m = identity.copy()
        for old_pos, new_pos in enumerate(perm):
            m[base + new_pos] = base + old_pos
        results.append(m)
    return results


def _wind_maps():
    """4 maps: cyclic rotations of F1/F2/F3/F4 (东南西北) by 0/1/2/3 steps."""
    identity = np.arange(NUM_POS, dtype=np.int64)
    results = []
    base = 27                              # F1 at pos 27
    for shift in range(4):
        m = identity.copy()
        for i in range(4):
            m[base + (i + shift) % 4] = base + i
        results.append(m)
    return results


#  Action permutation builder

def _build_action_fwd(tile_fwd, ds_active, suit_perm):
    """
    Build forward action permutation (235,) from tile forward map and transform flags.

    Args:
        tile_fwd: (36,) forward tile map  tile_fwd[old] = new
        ds_active: bool — digital symmetry active
        suit_perm: (3,) — old_suit → new_suit under suit permutation

    Returns:
        action_fwd: (235,)  action_fwd[old_action] = new_action
    """
    act = np.arange(ACT_SIZE, dtype=np.int64)

    # Pass (0), Hu (1): unchanged

    # Play (2-35), Peng (99-132), Gang (133-166), AnGang (167-200), BuGang (201-234)
    # All encode tile index as  action = base + tile_idx
    for old_tile in range(NUM_TILES):
        new_tile = tile_fwd[old_tile]
        if new_tile >= NUM_TILES:
            # Should not happen: real tiles always map to real tiles
            continue
        for base in (OFFSET_ACT['Play'], OFFSET_ACT['Peng'], OFFSET_ACT['Gang'],
                     OFFSET_ACT['AnGang'], OFFSET_ACT['BuGang']):
            act[base + old_tile] = base + new_tile

    # Chi (36-98): encoded as 36 + color*21 + (middle_num-2)*3 + claim_pos
    # where color ∈ {0,1,2}, middle_num ∈ {2..8}, claim_pos ∈ {0,1,2}
    for old_color in range(3):
        for mid in range(2, 9):             # middle_num 2-8
            for cp in range(3):             # claim_pos 0=左,1=中,2=右
                old_act = 36 + old_color * 21 + (mid - 2) * 3 + cp

                # Apply suit permutation
                new_color = suit_perm[old_color]

                # Apply digital symmetry (if active)
                if ds_active:
                    new_mid = 10 - mid       # 2↔8, 3↔7, 4↔6, 5↔5
                    new_cp = 2 - cp          # 0↔2 (左↔右), 1↔1 (中不变)
                else:
                    new_mid = mid
                    new_cp = cp

                new_act = 36 + new_color * 21 + (new_mid - 2) * 3 + new_cp
                act[old_act] = new_act

    return act


#  Build all 288 transforms

def _build_all_transforms():
    """Build and return list of 288 transform dicts (cached)."""
    ds_maps = _digital_symmetry_maps()       # 2
    sp_maps = _suit_maps()                   # 6
    hp_maps = _honor_maps()                  # 6
    wp_maps = _wind_maps()                   # 4

    transforms = []

    for ds_idx, ds_map in enumerate(ds_maps):
        ds_active = (ds_idx == 1)

        for sp_map in sp_maps:
            # Extract suit permutation (old_suit → new_suit) from forward suit map
            sp_fwd = np.argsort(sp_map)      # forward: old→new
            suit_perm = [sp_fwd[i * 9] // 9 for i in range(3)]

            for hp_map in hp_maps:
                for wp_idx, wp_map in enumerate(wp_maps):
                    wind_shift = wp_idx

                    # Compose inverse tile maps: tile_inv[new] = ds_inv(sp_inv(hp_inv(wp_inv(new))))
                    # Since maps are already inverse (map[new]=old), composition is direct
                    tile_inv = ds_map[sp_map[hp_map[wp_map]]]

                    # Forward map: argsort of inverse
                    tile_fwd = np.argsort(tile_inv)

                    # Action permutation
                    action_fwd = _build_action_fwd(tile_fwd, ds_active, suit_perm)
                    action_inv = np.argsort(action_fwd)

                    transforms.append({
                        'tile_inv': tile_inv,         # (36,)
                        'tile_fwd': tile_fwd,         # (36,)
                        'action_fwd': action_fwd,     # (235,)
                        'action_inv': action_inv,     # (235,)
                        'ds_active': ds_active,        # bool
                        'wind_shift': wind_shift,      # int 0-3
                    })

    assert len(transforms) == 288, f'Expected 288, got {len(transforms)}'
    return transforms


def get_transforms():
    """Return the list of 288 transform dicts (lazy-init, cached)."""
    global _ALL_TRANSFORMS
    if _ALL_TRANSFORMS is None:
        _ALL_TRANSFORMS = _build_all_transforms()
    return _ALL_TRANSFORMS


def get_transforms_by_level(level):
    """
    Return a subset of the 288 transforms at the given augmentation level.

    The 288 transforms are built in nested-loop order:
        ds(2) → sp(6) → hp(6) → wp(4)
        index = ds*144 + sp*24 + hp*4 + wp

    Levels:
        2   — digital symmetry only (1↔9, 2↔8, … in each suit)
               selects sp=identity(0), hp=identity(0), wp=identity(0)
        12  — digital symmetry + suit permutation (W/T/B)
               selects hp=identity(0), wp=identity(0)
        72  — digital symmetry + suit + honor (J1/J2/J3)
               selects wp=identity(0)
        288 — all four factors (full augmentation)
    """
    all_tfs = get_transforms()

    if level == 288:
        return all_tfs

    keep = []
    for k in range(288):
        sp = (k // 24) % 6   # suit perm index (0 = identity)
        hp = (k // 4) % 6    # honor perm index (0 = identity)
        wp = k % 4           # wind rotation index (0 = identity)

        if level == 2:
            if sp == 0 and hp == 0 and wp == 0:
                keep.append(k)
        elif level == 12:
            if hp == 0 and wp == 0:
                keep.append(k)
        elif level == 72:
            if wp == 0:
                keep.append(k)
        else:
            raise ValueError(
                f'Unknown level: {level}. Valid: 2, 12, 72, 288')

    return [all_tfs[k] for k in keep]


#  Apply a single transform to one sample

def apply_transform(obs, mask, act, tf):
    """
    Apply one transform to a single sample.

    Args:
        obs:  (160, 4, 9) float32 — original observation
        mask: (235,)     float32 — original action mask
        act:  int                 — original action label
        tf:   dict               — transform from get_transforms()

    Returns:
        aug_obs:  (160, 4, 9) float32
        aug_mask: (235,)     float32
        aug_act:  int
    """
    tile_inv = tf['tile_inv']
    action_inv = tf['action_inv']
    action_fwd = tf['action_fwd']
    ds_active = tf['ds_active']
    wind_shift = tf['wind_shift']

    aug_obs = obs.copy()

    # Reshape (160,4,9) → flat spatial dimension → permute → reshape back
    obs_flat = aug_obs.reshape(160, NUM_POS)       # (160, 36)
    obs_flat[TILE_CHANNELS] = obs_flat[TILE_CHANNELS][:, tile_inv]


    if ds_active:
        for ch_l, ch_r in MELD_CHI_SWAP_PAIRS:
            aug_obs[ch_l], aug_obs[ch_r] = aug_obs[ch_r].copy(), aug_obs[ch_l].copy()

    if wind_shift != 0:
        aug_obs[WIND_PREV_CH] = np.roll(obs[WIND_PREV_CH], shift=wind_shift, axis=0)
        aug_obs[WIND_SEAT_CH] = np.roll(obs[WIND_SEAT_CH], shift=wind_shift, axis=0)

    # action_inv[new] = old: aug_mask[new] = orig_mask[old]
    aug_mask = mask[action_inv]

    aug_act = action_fwd[act]

    return aug_obs, aug_mask, aug_act


#  Augment a batch of samples → 288× expanded batch

def augment_batch(obs_stack, mask_stack, act_stack):
    """
    Apply all 288 transforms to a batch, returning a 288× expanded batch.

    Args:
        obs_stack:  (N, 160, 4, 9) float32
        mask_stack: (N, 235)        float32
        act_stack:  (N,)            int64

    Returns:
        aug_obs:  (288*N, 160, 4, 9) float32
        aug_mask: (288*N, 235)        float32
        aug_act:  (288*N,)            int64
    """
    N = obs_stack.shape[0]
    transforms = get_transforms()

    aug_obs = np.empty((288 * N, 160, 4, 9), dtype=np.float32)
    aug_mask = np.empty((288 * N, ACT_SIZE), dtype=np.float32)
    aug_act = np.empty((288 * N,), dtype=np.int64)

    # Pre-flatten obs_stack for fast channel-wise access
    obs_flat = obs_stack.reshape(N, 160, NUM_POS)  # (N, 160, 36)

    for k, tf in enumerate(transforms):
        tile_inv = tf['tile_inv']
        action_inv = tf['action_inv']
        action_fwd = tf['action_fwd']
        ds_active = tf['ds_active']
        wind_shift = tf['wind_shift']

        s = k * N
        e = s + N

        # Output flat view for this transform block
        flat_out = aug_obs[s:e].reshape(N, 160, NUM_POS)

        # --- Unchanged channels (WALL, TILE_FROM, WALL_LAST, IS_ABOUT_KONG,
        #     HAND_SIZE, plus wind channels as default) ---
        flat_out[:, UNCHANGED_CHANNELS] = obs_flat[:, UNCHANGED_CHANNELS]

        # --- Tile-permuted channels (HAND, SHOWN, MELD, DISCARD, CUR_TILE) ---
        flat_out[:, TILE_CHANNELS] = obs_flat[:, TILE_CHANNELS][:, :, tile_inv]

        # --- MELD chi channel swap (digital symmetry: left↔right) ---
        if ds_active:
            for ch_l, ch_r in MELD_CHI_SWAP_PAIRS:
                tmp = aug_obs[s:e, ch_l].copy()
                aug_obs[s:e, ch_l] = aug_obs[s:e, ch_r]
                aug_obs[s:e, ch_r] = tmp

        # --- Wind rotation (overwrites default wind channels from step 1) ---
        if wind_shift != 0:
            aug_obs[s:e, WIND_PREV_CH] = np.roll(
                obs_stack[:, WIND_PREV_CH], shift=wind_shift, axis=1)
            aug_obs[s:e, WIND_SEAT_CH] = np.roll(
                obs_stack[:, WIND_SEAT_CH], shift=wind_shift, axis=1)

        # --- Action mask remapping ---
        aug_mask[s:e] = mask_stack[:, action_inv]

        # --- Action label remapping ---
        aug_act[s:e] = action_fwd[act_stack]

    return aug_obs, aug_mask, aug_act

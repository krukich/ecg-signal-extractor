import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


CHANNELS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
KNOWN_PLOT_AREA_1080P = (37, 1913, 116, 1038)
SCAN_MODE_SCREEN = "screen"
SCAN_MODE_SCAN = "scan"
SCAN_DIVIDER_RATIO = 0.4705
SCAN_TRACE_THRESHOLD = 120
SCAN_RHYTHM_BAND_FACTOR = 0.42
SCREEN_REFERENCE_INTERVAL_PX = 226.0
SCREEN_REFERENCE_INTERVAL_MS = 200.0
SCREEN_REFERENCE_PX_PER_MS = SCREEN_REFERENCE_INTERVAL_PX / SCREEN_REFERENCE_INTERVAL_MS
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def is_screen_sized(image):
    h, w = image.shape[:2]
    return (w, h) == (1920, 1080)


def is_probably_screen_capture(image):
    if not is_screen_sized(image):
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x1, x2, y1, y2 = KNOWN_PLOT_AREA_1080P
    roi = gray[y1:y2, x1:x2]

    if roi.size == 0:
        return False

    return float(np.median(roi)) < 55.0 and float(np.percentile(roi, 85)) < 120.0


def collect_image_paths(input_dir):
    return sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def filter_image_paths_for_mode(image_paths, mode):
    if mode == SCAN_MODE_SCAN:
        return image_paths

    filtered = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is not None and is_screen_sized(image):
            filtered.append(image_path)

    return filtered


def get_plot_area(image):
    h, w = image.shape[:2]

    if (w, h) != (1920, 1080):
        raise ValueError(f"Unsupported image size: {(w, h)}. Expected 1920x1080.")

    return KNOWN_PLOT_AREA_1080P


def get_row_geometry(image):
    x1, x2, y1, y2 = get_plot_area(image)
    row_step = (y2 - y1) / 14.0
    centers = [y1 + (idx + 0.5) * row_step for idx in range(len(CHANNELS))]
    return x1, x2, y1, y2, row_step, centers



def extract_column_groups(signal_mask, x1, x2, y1, y2, row_step):
    group_columns = []
    max_group_height = row_step * 2.40

    h, w = signal_mask.shape[:2]

    for x in range(x1, x2):
        column = signal_mask[y1:y2 + 1, x]
        ys = np.where(column > 0)[0]

        if len(ys) == 0:
            group_columns.append([])
            continue

        ys_global = y1 + ys.astype(np.float32)
        groups = split_column_points(ys_global, max_gap=2)

        column_groups = []

        for group in groups:
            y_min = float(np.min(group))
            y_max = float(np.max(group))
            group_height = y_max - y_min + 1.0

            if group_height > max_group_height:
                continue

            yy1 = max(0, int(round(y_min)) - 4)
            yy2 = min(h, int(round(y_max)) + 5)
            xx1 = max(0, x - 3)
            xx2 = min(w, x + 4)

            signal_window = signal_mask[yy1:yy2, xx1:xx2]
            signal_points = np.where(signal_window > 0)

            if len(signal_points[0]) > 0:
                signal_y = yy1 + signal_points[0].astype(np.float32)
                y_med = float(np.median(signal_y))
                y_low = float(np.min(signal_y))
                y_high = float(np.max(signal_y))
                support_count = int(len(signal_y))
            else:
                y_med = float(np.median(group))
                y_low = y_min
                y_high = y_max
                support_count = 0

            column_groups.append(
                {
                    "y": y_med,
                    "y_low": y_low,
                    "y_high": y_high,
                    "height": float(group_height),
                    "support": support_count,
                }
            )

        column_groups.sort(key=lambda item: item["y"])
        group_columns.append(column_groups)

    return group_columns


def select_ordered_subset(groups, reference_centers):
    return select_ordered_by_cost(
        groups=groups,
        targets=reference_centers,
        cost_fn=lambda group, target: abs(float(group["y"]) - float(target)),
    )


def estimate_channel_geometry(group_columns, fallback_centers, fallback_row_step):
    ranked_groups = [[] for _ in CHANNELS]

    exact_columns = [groups for groups in group_columns if len(groups) == len(CHANNELS)]

    if len(exact_columns) >= 32:
        source_columns = exact_columns
    else:
        source_columns = [groups for groups in group_columns if len(groups) >= len(CHANNELS)]

    for groups in source_columns:
        selected_groups = groups

        if len(selected_groups) > len(CHANNELS):
            selected_groups = select_ordered_subset(selected_groups, fallback_centers)

        if len(selected_groups) < len(CHANNELS):
            continue

        for channel_idx in range(len(CHANNELS)):
            ranked_groups[channel_idx].append(float(selected_groups[channel_idx]["y"]))

    centers = []

    for channel_idx in range(len(CHANNELS)):
        if ranked_groups[channel_idx]:
            centers.append(float(np.median(ranked_groups[channel_idx])))
        else:
            centers.append(float(fallback_centers[channel_idx]))

    min_spacing = fallback_row_step * 0.45

    for channel_idx in range(1, len(centers)):
        if centers[channel_idx] <= centers[channel_idx - 1] + min_spacing:
            centers[channel_idx] = centers[channel_idx - 1] + min_spacing

    row_diffs = np.diff(np.array(centers, dtype=np.float32))
    valid_diffs = row_diffs[(row_diffs >= fallback_row_step * 0.45) & (row_diffs <= fallback_row_step * 2.10)]

    if len(valid_diffs) > 0:
        row_step = float(np.median(valid_diffs))
    else:
        row_step = float(fallback_row_step)

    return row_step, centers


def clean_small_components(mask, x1, x2, y1, y2, min_area=2):
    cleaned = np.zeros_like(mask)
    cleaned[y1:y2, x1:x2] = clean_binary_mask(mask[y1:y2, x1:x2], min_area=min_area)
    return cleaned


def remove_left_label_components(mask, x1, x2, y1, y2, row_step):
    roi = mask[y1:y2, x1:x2].copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi, 8)

    label_band_width = int(max(30, min(90, round(row_step * 0.78))))
    max_label_width = int(max(12, round(row_step * 0.42)))
    max_label_height = int(max(10, round(row_step * 0.32)))

    for label_idx in range(1, num_labels):
        comp_x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        comp_w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        comp_right = comp_x + comp_w
        comp_bottom = int(stats[label_idx, cv2.CC_STAT_TOP]) + comp_h

        if (
            comp_x < label_band_width
            and comp_right <= label_band_width
            and comp_w <= max_label_width
            and comp_h <= max_label_height
            and comp_bottom < (y2 - y1)
        ):
            roi[labels == label_idx] = 0

    cleaned = mask.copy()
    cleaned[y1:y2, x1:x2] = roi
    return cleaned


def create_signal_mask(image):
    x1, x2, y1, y2 = get_plot_area(image)
    _, _, _, _, row_step, _ = get_row_geometry(image)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    roi_gray = gray[y1:y2, x1:x2]
    p97 = np.percentile(roi_gray, 97)
    threshold = int(max(24, min(70, p97 * 0.42)))

    light_gray = (gray >= threshold) & (value >= threshold) & (saturation <= 165)

    green_grid = (hue >= 35) & (hue <= 95) & (saturation >= 70) & (value >= 60)
    blue_lines = (hue >= 95) & (hue <= 140) & (saturation >= 60) & (value >= 40)
    red_debug = ((hue <= 8) | (hue >= 170)) & (saturation >= 80) & (value >= 60)

    candidate = light_gray & (~green_grid) & (~blue_lines) & (~red_debug)

    core_mask = np.zeros(gray.shape, dtype=np.uint8)
    core_mask[candidate] = 255

    core_mask[:y1, :] = 0
    core_mask[y2:, :] = 0
    core_mask[:, :x1] = 0
    core_mask[:, x2:] = 0

    roi = core_mask[y1:y2, x1:x2]

    col_activity = np.sum(roi > 0, axis=0)
    bad_cols = np.where(col_activity > 0.30 * (y2 - y1))[0]
    roi[:, bad_cols] = 0

    row_activity = np.sum(roi > 0, axis=1)
    bad_rows = np.where(row_activity > 0.75 * (x2 - x1))[0]
    roi[bad_rows, :] = 0

    core_mask[y1:y2, x1:x2] = roi
    core_mask = remove_left_label_components(core_mask, x1, x2, y1, y2, row_step)
    core_mask = clean_small_components(core_mask, x1, x2, y1, y2, min_area=2)

    return core_mask


def split_column_points(ys, max_gap=2):
    if len(ys) == 0:
        return []

    ys = np.sort(ys)
    groups = []
    current = [ys[0]]

    for value in ys[1:]:
        if value - current[-1] <= max_gap:
            current.append(value)
        else:
            groups.append(np.array(current, dtype=np.float32))
            current = [value]

    groups.append(np.array(current, dtype=np.float32))
    return groups


def update_state(states, key, cost, choices):
    if key not in states or cost < states[key][0]:
        states[key] = (cost, choices)


def select_ordered_by_cost(groups, targets, cost_fn):
    if len(groups) <= len(targets):
        return groups

    states = {-1: (0.0, [])}
    target_count = len(targets)

    for target_idx, target in enumerate(targets):
        next_states = {}
        max_group_idx = len(groups) - (target_count - target_idx - 1) - 1

        for last_idx, (current_cost, choices) in states.items():
            for group_idx in range(last_idx + 1, max_group_idx + 1):
                cost = current_cost + cost_fn(groups[group_idx], target)
                update_state(next_states, group_idx, cost, choices + [group_idx])

        states = next_states

    best_key = min(states.keys(), key=lambda key: states[key][0])
    return [groups[idx] for idx in states[best_key][1]]


def predict_channel_y(prev_positions, have_prev, gap_counts, centers, channel_idx):
    if not have_prev[channel_idx]:
        return float(centers[channel_idx])

    predicted = float(prev_positions[channel_idx])

    if gap_counts[channel_idx] > 0:
        anchor_weight = min(gap_counts[channel_idx] / 8.0, 0.65)
        predicted = (1.0 - anchor_weight) * predicted + anchor_weight * float(centers[channel_idx])

    return float(predicted)


def build_reference_positions(prev_positions, have_prev, gap_counts, centers, row_step):
    references = np.array(
        [predict_channel_y(prev_positions, have_prev, gap_counts, centers, idx) for idx in range(len(centers))],
        dtype=np.float32,
    )
    min_spacing = max(4.0, row_step * 0.22)

    for idx in range(1, len(references)):
        references[idx] = max(references[idx], references[idx - 1] + min_spacing)

    for idx in range(len(references) - 2, -1, -1):
        references[idx] = min(references[idx], references[idx + 1] - min_spacing)

    return references


def group_match_cost(group, reference_y, row_step):
    y = float(group["y"])
    distance = abs(y - float(reference_y)) / row_step
    support_bonus = min(float(group["support"]) / 16.0, 1.0) * 0.12
    height_bonus = min(float(group["height"]) / (row_step * 0.25), 1.0) * 0.08
    return float(distance - support_bonus - height_bonus)


def select_ordered_groups(groups, reference_positions, row_step):
    return select_ordered_by_cost(
        groups=groups,
        targets=reference_positions,
        cost_fn=lambda group, target: group_match_cost(group, target, row_step),
    )


def assign_sparse_groups(groups, reference_positions, row_step):
    n_channels = len(reference_positions)

    if len(groups) == 0:
        return [None] * n_channels

    states = {-1: (0.0, [])}

    for channel_idx in range(n_channels):
        new_states = {}

        for last_group_idx, (current_cost, choices) in states.items():
            missing_cost = current_cost + 0.95
            update_state(new_states, last_group_idx, missing_cost, choices + [-1])

            for group_idx in range(last_group_idx + 1, len(groups)):
                group = groups[group_idx]
                total_cost = current_cost + group_match_cost(group, reference_positions[channel_idx], row_step)
                update_state(new_states, group_idx, total_cost, choices + [group_idx])

        states = new_states

    best_key = min(states.keys(), key=lambda key: states[key][0])
    return [None if group_idx < 0 else groups[group_idx] for group_idx in states[best_key][1]]


def get_label_skip_columns(channel_name, row_step):
    if channel_name == "I":
        factor = 0.04
    elif channel_name == "II":
        factor = 0.10
    elif channel_name == "III":
        factor = 0.18
    elif channel_name.startswith("aV"):
        factor = 0.52
    elif channel_name.startswith("V"):
        factor = 0.33
    else:
        factor = 0.0

    return int(round(row_step * factor))


def trim_leading_label_artifacts(center_tracks, low_tracks, high_tracks, height_tracks, valid, centers, row_step):
    n_channels, width = center_tracks.shape
    startup_band_width = min(width, int(max(30, min(90, round(row_step * 0.90)))))
    stable_height_limit = max(3.0, row_step * 0.07)
    stable_distance_limit = row_step * 0.16
    stable_run_len = 4

    for channel_idx in range(n_channels):
        run_len = 0
        seed_start = None

        for x_idx in range(startup_band_width):
            is_stable = (
                valid[channel_idx, x_idx]
                and not np.isnan(center_tracks[channel_idx, x_idx])
                and not np.isnan(height_tracks[channel_idx, x_idx])
                and float(height_tracks[channel_idx, x_idx]) <= stable_height_limit
                and abs(float(center_tracks[channel_idx, x_idx]) - float(centers[channel_idx])) <= stable_distance_limit
            )

            if is_stable:
                run_len += 1
                if run_len >= stable_run_len:
                    seed_start = x_idx - stable_run_len + 1
                    break
            else:
                run_len = 0

        if seed_start is None or seed_start <= 0:
            continue

        center_tracks[channel_idx, :seed_start] = np.nan
        low_tracks[channel_idx, :seed_start] = np.nan
        high_tracks[channel_idx, :seed_start] = np.nan
        height_tracks[channel_idx, :seed_start] = np.nan
        valid[channel_idx, :seed_start] = False

    return center_tracks, low_tracks, high_tracks, height_tracks, valid


def track_group_columns(group_columns, centers, row_step):
    n_channels = len(centers)
    width = len(group_columns)
    startup_band_width = int(max(30, min(90, round(row_step * 0.78))))
    startup_height_limit = max(4.0, row_step * 0.10)
    startup_distance_limit = row_step * 0.32
    label_skip_cols = [get_label_skip_columns(channel_name, row_step) for channel_name in CHANNELS]

    center_tracks = np.full((n_channels, width), np.nan, dtype=np.float32)
    low_tracks = np.full((n_channels, width), np.nan, dtype=np.float32)
    high_tracks = np.full((n_channels, width), np.nan, dtype=np.float32)
    height_tracks = np.full((n_channels, width), np.nan, dtype=np.float32)
    valid = np.zeros((n_channels, width), dtype=bool)

    prev_positions = np.array(centers, dtype=np.float32)
    have_prev = np.zeros(n_channels, dtype=bool)
    gap_counts = np.zeros(n_channels, dtype=np.int32)

    for x_idx, groups in enumerate(group_columns):
        reference_positions = build_reference_positions(
            prev_positions=prev_positions,
            have_prev=have_prev,
            gap_counts=gap_counts,
            centers=centers,
            row_step=row_step,
        )

        if len(groups) >= n_channels:
            assigned_groups = select_ordered_groups(groups, reference_positions, row_step)[:n_channels]
        else:
            assigned_groups = assign_sparse_groups(groups, reference_positions, row_step)

        for channel_idx in range(n_channels):
            group = assigned_groups[channel_idx] if channel_idx < len(assigned_groups) else None

            if group is not None and not have_prev[channel_idx] and x_idx < label_skip_cols[channel_idx]:
                group = None

            if (
                group is not None
                and not have_prev[channel_idx]
                and x_idx < startup_band_width
                and (
                    float(group["height"]) > startup_height_limit
                    or abs(float(group["y"]) - float(centers[channel_idx])) > startup_distance_limit
                )
            ):
                group = None

            if group is None:
                gap_counts[channel_idx] += 1
                continue

            y = float(group["y"])
            center_tracks[channel_idx, x_idx] = y
            low_tracks[channel_idx, x_idx] = float(group["y_low"])
            high_tracks[channel_idx, x_idx] = float(group["y_high"])
            height_tracks[channel_idx, x_idx] = float(group["height"])
            valid[channel_idx, x_idx] = True
            prev_positions[channel_idx] = y
            have_prev[channel_idx] = True
            gap_counts[channel_idx] = 0

    return trim_leading_label_artifacts(center_tracks, low_tracks, high_tracks, height_tracks, valid, centers, row_step)


def choose_group_point(center_y, low_y, high_y, height, prev_y, next_y, row_step):
    if np.isnan(center_y):
        return np.nan

    if height <= 3.0:
        return float(center_y)

    if float(height) >= max(10.0, 0.10 * float(row_step)):
        return float(center_y)

    slope_threshold = max(2.0, row_step * 0.04)

    if np.isnan(prev_y) and np.isnan(next_y):
        return float(center_y)

    if np.isnan(prev_y):
        diff = float(next_y - center_y)
        if diff < -slope_threshold:
            return float(low_y)
        if diff > slope_threshold:
            return float(high_y)
        return float(center_y)

    if np.isnan(next_y):
        diff = float(center_y - prev_y)
        if diff < -slope_threshold:
            return float(low_y)
        if diff > slope_threshold:
            return float(high_y)
        return float(center_y)

    avg_y = 0.5 * (float(prev_y) + float(next_y))
    excursion_threshold = max(1.5, min(float(height) * 0.18, row_step * 0.10))

    if float(center_y) < avg_y - excursion_threshold:
        return float(low_y)

    if float(center_y) > avg_y + excursion_threshold:
        return float(high_y)

    slope = float(next_y - prev_y)

    if slope < -slope_threshold:
        return float(low_y)

    if slope > slope_threshold:
        return float(high_y)

    return float(center_y)


def refine_tracked_points(center_tracks, low_tracks, high_tracks, height_tracks, valid, row_step):
    n_channels, _ = center_tracks.shape
    refined = center_tracks.astype(np.float32).copy()

    for channel_idx in range(n_channels):
        center_path = center_tracks[channel_idx]
        low_path = low_tracks[channel_idx]
        high_path = high_tracks[channel_idx]
        height_path = height_tracks[channel_idx]
        path_valid = valid[channel_idx]

        valid_indices = np.where(path_valid & ~np.isnan(center_path))[0]

        for offset, x_idx in enumerate(valid_indices):
            prev_y = np.nan
            next_y = np.nan

            if offset > 0:
                prev_y = center_path[valid_indices[offset - 1]]

            if offset + 1 < len(valid_indices):
                next_y = center_path[valid_indices[offset + 1]]

            refined[channel_idx, x_idx] = choose_group_point(
                center_y=center_path[x_idx],
                low_y=low_path[x_idx],
                high_y=high_path[x_idx],
                height=height_path[x_idx],
                prev_y=prev_y,
                next_y=next_y,
                row_step=row_step,
            )

    return refined


def fill_short_gaps(path, valid, row_step, max_gap=7):
    result = path.astype(np.float32).copy()
    result_valid = valid.copy()

    n = len(result)
    idx = 0

    while idx < n:
        if result_valid[idx] and not np.isnan(result[idx]):
            idx += 1
            continue

        start = idx

        while idx < n and (not result_valid[idx] or np.isnan(result[idx])):
            idx += 1

        end = idx - 1
        gap_len = end - start + 1

        left = start - 1
        right = end + 1

        if (
            gap_len <= max_gap
            and left >= 0
            and right < n
            and result_valid[left]
            and result_valid[right]
            and not np.isnan(result[left])
            and not np.isnan(result[right])
        ):
            y_left = float(result[left])
            y_right = float(result[right])

            if abs(y_right - y_left) <= row_step * 0.42:
                for j in range(start, end + 1):
                    alpha = (j - left) / (right - left)
                    result[j] = (1.0 - alpha) * y_left + alpha * y_right
                    result_valid[j] = True

    return result, result_valid


def clean_tracks(tracks, valid, row_step):
    final_tracks = []
    final_valid = []

    for channel_idx in range(len(CHANNELS)):
        path = tracks[channel_idx].astype(np.float32).copy()
        path_valid = valid[channel_idx].copy()

        path, path_valid = fill_short_gaps(path, path_valid, row_step, max_gap=7)

        final_tracks.append(path.astype(np.float32))
        final_valid.append(path_valid)

    return np.array(final_tracks, dtype=np.float32), np.array(final_valid, dtype=bool)


def extract_all_y_values(group_columns, centers, row_step):
    center_tracks, low_tracks, high_tracks, height_tracks, valid = track_group_columns(group_columns, centers, row_step)
    tracks = refine_tracked_points(center_tracks, low_tracks, high_tracks, height_tracks, valid, row_step)
    return clean_tracks(tracks=tracks, valid=valid, row_step=row_step)


def estimate_baseline(y_values, valid, center_y, row_step):
    usable = valid & ~np.isnan(y_values)

    if np.sum(usable) == 0:
        return float(center_y)

    work = y_values.astype(np.float32).copy()
    work_valid = usable.astype(bool).copy()
    work, work_valid = fill_short_gaps(work, work_valid, row_step, max_gap=12)
    usable = work_valid & ~np.isnan(work)

    if np.sum(usable) == 0:
        return float(center_y)

    provisional = float(np.median(work[usable]))
    smooth = moving_average_1d(np.nan_to_num(work, nan=provisional).astype(np.float32), 11)
    residual = np.abs(work - smooth)
    amplitude = np.abs(work - provisional)

    local_jump = np.full(len(work), np.inf, dtype=np.float32)
    for idx in range(len(work)):
        if not usable[idx]:
            continue
        deltas = []
        if idx > 0 and usable[idx - 1]:
            deltas.append(abs(float(work[idx]) - float(work[idx - 1])))
        if idx + 1 < len(work) and usable[idx + 1]:
            deltas.append(abs(float(work[idx]) - float(work[idx + 1])))
        if deltas:
            local_jump[idx] = float(sum(deltas) / len(deltas))

    flat_mask = (
        usable
        & (local_jump <= max(2.0, 0.05 * float(row_step)))
        & (residual <= max(3.0, 0.09 * float(row_step)))
        & (amplitude <= max(6.0, 0.30 * float(row_step)))
    )

    min_flat_points = max(12, int(round(0.10 * float(np.sum(usable)))))
    if int(np.sum(flat_mask)) < min_flat_points:
        flat_mask = (
            usable
            & (residual <= max(4.0, 0.12 * float(row_step)))
            & (amplitude <= max(8.0, 0.38 * float(row_step)))
        )

    candidates = work[flat_mask] if np.sum(flat_mask) >= 8 else work[usable]
    if len(candidates) == 0:
        return float(center_y)

    q_low, q_high = np.percentile(candidates, [20.0, 80.0])
    central = candidates[(candidates >= q_low) & (candidates <= q_high)]

    if len(central) < max(8, int(round(0.25 * len(candidates)))):
        q_low, q_high = np.percentile(candidates, [25.0, 75.0])
        central = candidates[(candidates >= q_low) & (candidates <= q_high)]

    if len(central) == 0:
        central = candidates

    return float(np.median(central))


def fill_for_csv(path, valid, center_y, row_step):
    result = path.astype(np.float32).copy()
    result_valid = valid.copy()

    n = len(result)

    if np.sum(result_valid & ~np.isnan(result)) == 0:
        return np.full(n, float(center_y), dtype=np.float32)

    result, result_valid = fill_short_gaps(result, result_valid, row_step, max_gap=25)

    idx = 0

    while idx < n:
        if result_valid[idx] and not np.isnan(result[idx]):
            idx += 1
            continue

        start = idx

        while idx < n and (not result_valid[idx] or np.isnan(result[idx])):
            idx += 1

        end = idx - 1

        left = start - 1
        right = end + 1

        if left >= 0 and result_valid[left] and not np.isnan(result[left]):
            fill_value = float(result[left])
        elif right < n and result_valid[right] and not np.isnan(result[right]):
            fill_value = float(result[right])
        else:
            fill_value = float(center_y)

        result[start:end + 1] = fill_value
        result_valid[start:end + 1] = True

    result = np.nan_to_num(result, nan=float(center_y))

    return result.astype(np.float32)


def convert_y_to_signal(y_values, valid, center_y, row_step):
    baseline = estimate_baseline(y_values, valid, center_y, row_step)
    y_for_csv = fill_for_csv(y_values, valid, baseline, row_step)
    signal = baseline - y_for_csv
    signal = np.nan_to_num(signal, nan=0.0)
    return signal.astype(np.float32), float(baseline)


def resample_to_1ms(signal, px_per_ms):
    if px_per_ms <= 0:
        px_per_ms = 1.0

    signal = np.nan_to_num(signal.astype(np.float32), nan=0.0)

    x_px = np.arange(len(signal), dtype=np.float32)
    x_ms = x_px / px_per_ms

    duration = int(np.floor(x_ms[-1]))

    if duration <= 0:
        return np.array([0.0], dtype=np.float32), np.array([float(signal[0])], dtype=np.float32)

    target_time = np.arange(0, duration + 1, 1.0, dtype=np.float32)
    target_signal = np.interp(target_time, x_ms, signal)
    target_signal = np.nan_to_num(target_signal.astype(np.float32), nan=0.0)

    return target_time, target_signal


def moving_average_1d(values, window):
    if window <= 1:
        return values.astype(np.float32)

    kernel = np.ones(int(window), dtype=np.float32) / float(window)
    return np.convolve(values.astype(np.float32), kernel, mode="same").astype(np.float32)


def smooth_1d(values, sigma):
    if sigma <= 0:
        return values.astype(np.float32)

    smoothed = cv2.GaussianBlur(values.astype(np.float32).reshape(1, -1), (0, 0), sigmaX=float(sigma))
    return smoothed.ravel().astype(np.float32)


def normalize_to_u8(values):
    if values.dtype == np.uint8:
        return values

    values = values.astype(np.float32)
    low = float(np.min(values))
    high = float(np.max(values))

    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)

    scaled = (values - low) * (255.0 / (high - low))
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)

def detect_r_peaks(signal_1ms):
    signal = np.nan_to_num(signal_1ms.astype(np.float32), nan=0.0)
    smooth = moving_average_1d(signal, 7)
    baseline = moving_average_1d(smooth, 151)
    high_pass = smooth - baseline

    derivative = np.diff(high_pass, prepend=high_pass[0])
    energy = moving_average_1d(derivative * derivative, 35)

    threshold = max(float(np.percentile(energy, 92)), float(np.mean(energy) + 1.2 * np.std(energy)))
    min_distance = 280

    candidates = []

    for idx in range(1, len(energy) - 1):
        if energy[idx] < threshold:
            continue

        if energy[idx] < energy[idx - 1] or energy[idx] <= energy[idx + 1]:
            continue

        if candidates and idx - candidates[-1] < min_distance:
            if energy[idx] > energy[candidates[-1]]:
                candidates[-1] = idx
        else:
            candidates.append(idx)

    peaks = []

    for candidate in candidates:
        left = max(0, candidate - 45)
        right = min(len(high_pass), candidate + 46)
        refined = left + int(np.argmax(np.abs(high_pass[left:right])))

        if peaks and refined - peaks[-1] < 220:
            if abs(high_pass[refined]) > abs(high_pass[peaks[-1]]):
                peaks[-1] = refined
        else:
            peaks.append(refined)

    return np.array(peaks, dtype=np.int32)


def estimate_bpm_from_signal(signal_1ms):
    peaks = detect_r_peaks(signal_1ms)

    if len(peaks) < 2:
        return None, peaks

    rr_intervals = np.diff(peaks).astype(np.float32)
    median_rr = float(np.median(rr_intervals))

    robust_rr = rr_intervals[(rr_intervals >= median_rr * 0.88) & (rr_intervals <= median_rr * 1.12)]

    if len(robust_rr) >= 2:
        rr_used = robust_rr
    else:
        rr_used = rr_intervals

    mean_rr = float(np.mean(rr_used))

    if mean_rr <= 0:
        return None, peaks

    bpm = 60000.0 / mean_rr

    if bpm < 20.0 or bpm > 240.0:
        return None, peaks

    return float(bpm), peaks


def estimate_bpm_from_channels(channels_data):
    preferred_leads = ["II", "V5", "V4", "V3", "I", "aVF", "V2"]
    best_result = None

    for lead_name in preferred_leads:
        if lead_name not in channels_data:
            continue

        signal = channels_data[lead_name].astype(np.float32)
        bpm, peaks = estimate_bpm_from_signal(signal)

        if bpm is None:
            continue

        amplitude_score = float(np.nanpercentile(np.abs(signal), 98))
        score = (len(peaks), amplitude_score)

        if best_result is None or score > best_result[0]:
            best_result = (score, bpm, lead_name)

    if best_result is None:
        return None, None

    return float(best_result[1]), best_result[2]


def group_consecutive_indices(indices, max_gap=2):
    if len(indices) == 0:
        return []

    groups = []
    start = int(indices[0])
    prev = int(indices[0])

    for value in indices[1:]:
        value = int(value)
        if value <= prev + max_gap:
            prev = value
        else:
            groups.append((start, prev))
            start = value
            prev = value

    groups.append((start, prev))
    return groups


def build_output_dataframe(channel_resampled):
    max_duration = 0

    for _, (time_ms, _) in channel_resampled.items():
        if len(time_ms) > 0:
            max_duration = max(max_duration, int(time_ms[-1]))

    common_time = np.arange(0, max_duration + 1, 1.0, dtype=np.float32)
    aligned = {}

    for channel in CHANNELS:
        if channel in channel_resampled:
            time_ms, signal_1ms = channel_resampled[channel]
            aligned_signal = np.full(len(common_time), np.nan, dtype=np.float32)
            valid_mask = (common_time >= float(time_ms[0])) & (common_time <= float(time_ms[-1]))

            if np.any(valid_mask):
                aligned_signal[valid_mask] = np.interp(common_time[valid_mask], time_ms, signal_1ms).astype(np.float32)

            aligned[channel] = aligned_signal.astype(np.float32)
        else:
            aligned[channel] = np.full(len(common_time), np.nan, dtype=np.float32)

    data = {"time_ms": common_time.astype(float)}

    for channel in CHANNELS:
        data[channel] = pd.array(np.rint(aligned[channel]), dtype="Int64")

    return pd.DataFrame(data), common_time, aligned



def build_render_config(channel_names, baselines, x1, x2, px_per_ms):
    return {
        channel_name: {
            "reference_y": float(baselines[idx]),
            "x1": int(x1),
            "x2": int(x2),
            "px_per_ms": float(px_per_ms),
        }
        for idx, channel_name in enumerate(channel_names)
    }

def build_uniform_render_config(reference_levels, x1, px_per_ms):
    return build_render_config(CHANNELS, reference_levels, x1, KNOWN_PLOT_AREA_1080P[1], px_per_ms)


def draw_signal_baselines_on_canvas(canvas, render_config, color):
    h, w = canvas.shape[:2]

    for channel_name in CHANNELS:
        if channel_name not in render_config:
            continue

        channel_cfg = render_config[channel_name]
        x1 = int(np.clip(channel_cfg["x1"], 0, w - 1))
        x2 = int(np.clip(int(channel_cfg["x2"]) - 1, 0, w - 1))
        y = int(np.clip(round(float(channel_cfg["reference_y"])), 0, h - 1))

        if x2 > x1:
            cv2.line(canvas, (x1, y), (x2, y), color, 1)

    return canvas


def draw_signals_on_canvas(canvas, time_ms, channels_data, render_config, color):
    h, w = canvas.shape[:2]

    for channel_name in CHANNELS:
        if channel_name not in channels_data or channel_name not in render_config:
            continue

        signal = channels_data[channel_name].astype(np.float32)
        channel_cfg = render_config[channel_name]
        x1 = int(channel_cfg["x1"])
        x2 = int(channel_cfg["x2"])
        if x2 <= x1:
            continue

        x_values = x1 + np.rint(time_ms * float(channel_cfg["px_per_ms"])).astype(np.int32)
        y_values = float(channel_cfg["reference_y"]) - signal
        finite_mask = np.isfinite(signal) & np.isfinite(y_values) & (x_values >= x1) & (x_values < x2)

        if np.sum(finite_mask) < 2:
            continue

        x_values = x_values[finite_mask]
        y_values = y_values[finite_mask]

        unique_x, start_indices = np.unique(x_values, return_index=True)

        if len(unique_x) < 2:
            continue

        counts = np.diff(np.append(start_indices, len(x_values)))
        summed_y = np.add.reduceat(y_values.astype(np.float32), start_indices)
        y_values = summed_y / np.maximum(counts, 1)
        x_values = unique_x

        for i in range(1, len(x_values)):
            x_prev = int(np.clip(x_values[i - 1], 0, w - 1))
            x_curr = int(np.clip(x_values[i], 0, w - 1))
            y_prev = int(np.clip(round(float(y_values[i - 1])), 0, h - 1))
            y_curr = int(np.clip(round(float(y_values[i])), 0, h - 1))
            cv2.line(canvas, (x_prev, y_prev), (x_curr, y_curr), color, 1)

    return canvas


def save_debug_csv_renders(csv_path, image, render_config, image_path, debug_dir):
    debug_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    time_ms = df["time_ms"].to_numpy(dtype=np.float32)
    channels_data = {channel: df[channel].to_numpy(dtype=np.float32) for channel in CHANNELS if channel in df.columns}

    csv_overlay = draw_signal_baselines_on_canvas(
        canvas=image.copy(),
        render_config=render_config,
        color=(0, 0, 255),
    )

    csv_overlay = draw_signals_on_canvas(
        canvas=csv_overlay,
        time_ms=time_ms,
        channels_data=channels_data,
        render_config=render_config,
        color=(0, 255, 0),
    )

    cv2.imwrite(str(debug_dir / f"{image_path.stem}_csv_overlay.png"), csv_overlay)


def process_screen_image(image_path, output_dir, debug_dir=None):
    image = load_image(image_path)
    px_per_ms = SCREEN_REFERENCE_PX_PER_MS

    x1, x2, y1, y2, fallback_row_step, fallback_centers = get_row_geometry(image)
    signal_mask = create_signal_mask(image)
    group_columns = extract_column_groups(signal_mask=signal_mask, x1=x1, x2=x2, y1=y1, y2=y2, row_step=fallback_row_step)
    row_step, centers = estimate_channel_geometry(group_columns, fallback_centers, fallback_row_step)

    all_y_values, all_y_valid = extract_all_y_values(group_columns=group_columns, centers=centers, row_step=row_step)

    channel_resampled = {}
    baseline_levels = []

    for channel, center_y, y_values, y_valid in zip(CHANNELS, centers, all_y_values, all_y_valid):
        signal, baseline_y = convert_y_to_signal(
            y_values,
            y_valid,
            center_y,
            row_step,
        )
        baseline_levels.append(baseline_y)
        channel_resampled[channel] = resample_to_1ms(signal, px_per_ms)

    df, _, aligned_channels = build_output_dataframe(channel_resampled)
    output_path = output_dir / f"{image_path.stem}.csv"
    df.to_csv(output_path, index=False)

    if debug_dir is not None:
        render_config = build_uniform_render_config(baseline_levels, x1, px_per_ms)
        save_debug_csv_renders(output_path, image, render_config, image_path, debug_dir)

    bpm, bpm_source = estimate_bpm_from_channels(aligned_channels)

    print(f"Using px_per_ms={px_per_ms:.6f} (screen_reference)")
    print(f"Saved: {output_path}")
    if bpm is not None and bpm_source is not None:
        print(f"BPM: {bpm:.1f} ({bpm_source})")
    else:
        print("BPM: unavailable")


def order_quad_points(points):
    points = np.array(points, dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()

    top_left = points[np.argmin(sums)]
    bottom_right = points[np.argmax(sums)]
    top_right = points[np.argmin(diffs)]
    bottom_left = points[np.argmax(diffs)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def warp_to_quad(image, quad):
    ordered = order_quad_points(quad)
    width_top = np.linalg.norm(ordered[1] - ordered[0])
    width_bottom = np.linalg.norm(ordered[2] - ordered[3])
    height_left = np.linalg.norm(ordered[3] - ordered[0])
    height_right = np.linalg.norm(ordered[2] - ordered[1])

    width = int(round(max(width_top, width_bottom)))
    height = int(round(max(height_left, height_right)))

    if width <= 0 or height <= 0:
        return image

    target = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(ordered, target)
    return cv2.warpPerspective(image, transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def rectify_scan_perspective(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.55:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) != 4:
            continue

        warped = warp_to_quad(image, approx.reshape(4, 2))
        if warped.shape[0] >= int(image.shape[0] * 0.8) and warped.shape[1] >= int(image.shape[1] * 0.8):
            return warped

    return image


def rotate_image(image, angle_deg):
    h, w = image.shape[:2]
    center = (w * 0.5, h * 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def rotate_scan_to_horizontal(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    min_length = max(180, image.shape[1] // 4)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=120, minLineLength=min_length, maxLineGap=20)

    if lines is None:
        return image

    angles = []

    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = line
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

        if angle < -90.0:
            angle += 180.0
        if angle > 90.0:
            angle -= 180.0

        if abs(angle) <= 8.0:
            angles.append(angle)

    if not angles:
        return image

    median_angle = float(np.median(np.array(angles, dtype=np.float32)))

    if abs(median_angle) < 0.15:
        return image

    return rotate_image(image, -median_angle)


def estimate_scan_grid_angle(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    red_grid = (((hue <= 28) | (hue >= 165)) & (saturation >= 40) & (value >= 190)).astype(np.uint8) * 255
    red_grid = cv2.medianBlur(red_grid, 3)

    h, w = red_grid.shape
    min_length = max(120, int(round(min(h, w) * 0.18)))
    lines = cv2.HoughLinesP(
        red_grid,
        1,
        np.pi / 180.0,
        threshold=80,
        minLineLength=min_length,
        maxLineGap=18,
    )

    if lines is None:
        return None

    horizontal_angles = []
    vertical_angles = []

    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = line
        dx = float(x2 - x1)
        dy = float(y2 - y1)

        if abs(dx) < 1.0 and abs(dy) < 1.0:
            continue

        angle = math.degrees(math.atan2(dy, dx))

        while angle <= -90.0:
            angle += 180.0
        while angle > 90.0:
            angle -= 180.0

        length = math.hypot(dx, dy)

        if abs(angle) <= 12.0:
            horizontal_angles.extend([angle] * max(1, int(length // 80)))
        elif abs(abs(angle) - 90.0) <= 12.0:
            if angle > 0:
                vertical_as_horizontal = angle - 90.0
            else:
                vertical_as_horizontal = angle + 90.0
            vertical_angles.extend([vertical_as_horizontal] * max(1, int(length // 80)))

    candidates = []

    if horizontal_angles:
        candidates.append(float(np.median(np.array(horizontal_angles, dtype=np.float32))))

    if vertical_angles:
        candidates.append(float(np.median(np.array(vertical_angles, dtype=np.float32))))

    if not candidates:
        return None

    return float(np.median(np.array(candidates, dtype=np.float32)))


def rotate_scan_to_grid(image):
    angle = estimate_scan_grid_angle(image)

    if angle is None or abs(angle) < 0.08:
        return image

    if abs(angle) > 3.0:
        return image

    return rotate_image(image, -angle)


def prepare_scan_image(image):
    rectified = rectify_scan_perspective(image)
    grid_aligned = rotate_scan_to_grid(rectified)
    return rotate_scan_to_horizontal(grid_aligned)


def normalize_scan_gray(gray):
    gray = gray.astype(np.uint8)
    gray_med = cv2.medianBlur(gray, 3)

    background = cv2.GaussianBlur(
        gray_med,
        (0, 0),
        sigmaX=85.0,
        sigmaY=85.0,
    )

    background = np.maximum(background, 1)
    norm = cv2.divide(gray_med, background, scale=245)
    norm = cv2.GaussianBlur(norm, (3, 3), 0)
    return norm.astype(np.uint8)


def estimate_profile_period(profile, lag_min, lag_max):
    centered = profile.astype(np.float32) - float(np.mean(profile))

    if np.allclose(centered, 0.0):
        raise ValueError("Flat profile")

    corr = np.correlate(centered, centered, mode="full")
    corr = corr[len(centered) - 1:len(centered) + lag_max + 1]
    corr[:lag_min] = 0.0

    best_lag = int(np.argmax(corr[lag_min:lag_max + 1])) + lag_min
    return float(best_lag)


def estimate_scan_big_grid_px(norm_gray):
    darkness = (255 - norm_gray).astype(np.float32)
    h, w = darkness.shape

    regions = [
        darkness[int(h * 0.15):int(h * 0.85), int(w * 0.08):int(w * 0.95)],
        darkness[int(h * 0.80):int(h * 0.94), int(w * 0.08):int(w * 0.95)],
    ]

    candidates = []

    for region in regions:
        if region.size == 0:
            continue

        profile = np.mean(region, axis=0)
        try:
            candidates.append(estimate_profile_period(profile, 35, 60))
        except Exception:
            continue

    if not candidates:
        return 47.0

    return float(np.median(np.array(candidates, dtype=np.float32)))


def create_scan_geometry_images(norm_gray):
    blackhat = cv2.morphologyEx(norm_gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    strong_mask = ((norm_gray < 224) & (blackhat > 12)).astype(np.uint8) * 255
    score_image = np.maximum(blackhat.astype(np.float32) - 14.0, 0.0)
    score_image += 0.10 * np.maximum(218.0 - norm_gray.astype(np.float32), 0.0)
    return strong_mask, score_image.astype(np.float32)


def remove_dense_background_components(mask, row_step):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    cleaned = mask.copy()
    min_area = max(140, int(round(float(row_step) * float(row_step) * 0.12)))
    min_width = max(18, int(round(float(row_step) * 0.45)))
    min_height = max(18, int(round(float(row_step) * 0.45)))

    for label_idx in range(1, num_labels):
        x, y, comp_w, comp_h, area = stats[label_idx]
        if area < min_area or comp_w < min_width or comp_h < min_height:
            continue
        density = float(area) / max(1.0, float(comp_w * comp_h))
        if density >= 0.22:
            cleaned[labels == label_idx] = 0

    return cleaned.astype(np.uint8)


def create_scan_trace_images(norm_gray, gray, row_step=None, image=None):
    if row_step is None or row_step <= 0:
        row_step = max(35.0, gray.shape[0] / 9.0)

    norm_u8 = norm_gray.astype(np.uint8)
    gray_u8 = gray.astype(np.uint8)
    gray_med = cv2.medianBlur(gray_u8, 3)
    background = cv2.GaussianBlur(gray_med, (0, 0), sigmaX=23.0, sigmaY=23.0)
    local_dark = np.maximum(background.astype(np.float32) - gray_med.astype(np.float32), 0.0)
    blackhat = cv2.morphologyEx(
        gray_med,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    ).astype(np.float32)

    grid_color = np.zeros_like(gray_u8)
    if image is not None:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        grid_color = (
            ((hue <= 28) | (hue >= 165))
            & (saturation >= 24)
            & (saturation <= 150)
            & (value >= 135)
            & (gray_u8 >= 125)
            & (local_dark <= 38.0)
        ).astype(np.uint8) * 255
        grid_color = cv2.medianBlur(grid_color, 3)

    trace_mask = (
        (((local_dark >= 13.0) | (blackhat >= 12.0)) & (norm_u8 <= 248))
        | ((norm_u8 <= 145) & (local_dark >= 4.0))
    ) & (grid_color == 0)
    trace_mask = trace_mask.astype(np.uint8) * 255
    trace_mask = cv2.morphologyEx(trace_mask, cv2.MORPH_CLOSE, np.ones((2, 2), dtype=np.uint8), iterations=1)
    trace_mask = clean_binary_mask(trace_mask, min_area=max(5, int(round(row_step * 0.05))))
    trace_mask = remove_dense_background_components(trace_mask, row_step)

    dark_score = np.maximum(238.0 - norm_u8.astype(np.float32), 0.0)
    score_image = 1.45 * local_dark + 0.45 * dark_score + 0.90 * np.maximum(blackhat - 6.0, 0.0)
    score_image[grid_color > 0] *= 0.08
    score_image[trace_mask > 0] += 8.0

    return blackhat.astype(np.float32), trace_mask.astype(np.uint8), score_image.astype(np.float32)


def clean_binary_mask(mask, min_area=5):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)

    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        height = int(stats[label_idx, cv2.CC_STAT_HEIGHT])

        if area < int(min_area):
            continue
        if width <= 1 and height <= 1:
            continue

        cleaned[labels == label_idx] = 255

    return cleaned


def estimate_calibration_baseline_from_component(component_mask, comp_y, comp_w):
    row_counts = np.sum(component_mask > 0, axis=1).astype(np.float32)

    if len(row_counts) == 0 or float(np.max(row_counts)) <= 0.0:
        return float(comp_y)

    min_count = max(8.0, min(float(comp_w) * 0.38, float(np.max(row_counts)) * 0.60))
    candidate_rows = np.where(row_counts >= min_count)[0]

    if len(candidate_rows) == 0:
        return float(comp_y + int(np.argmax(row_counts)))

    groups = group_consecutive_indices(candidate_rows, max_gap=2)
    best_group = None
    best_score = None

    for start, end in groups:
        rows = np.arange(start, end + 1, dtype=np.int32)
        support = float(np.sum(row_counts[rows]))
        lower_bonus = 0.08 * float(end)
        score = support + lower_bonus

        if best_group is None or score > best_score:
            best_group = (start, end)
            best_score = score

    start, end = best_group
    rows = np.arange(start, end + 1, dtype=np.float32)
    weights = row_counts[start:end + 1]

    if float(np.sum(weights)) <= 0.0:
        return float(comp_y + 0.5 * (start + end))

    return float(comp_y + np.sum(rows * weights) / np.sum(weights))


def choose_regular_scan_centers(candidates, supports, expected_count, image_h):
    candidates = [float(value) for value in candidates]
    supports = [float(value) for value in supports]

    if len(candidates) == 0:
        top = float(image_h) * 0.12
        bottom = float(image_h) * 0.88
        return list(np.linspace(top, bottom, expected_count, dtype=np.float32))

    pairs = sorted(zip(candidates, supports), key=lambda item: item[0])
    candidates = [item[0] for item in pairs]
    supports = [item[1] for item in pairs]

    if len(candidates) <= expected_count:
        centers = list(candidates)
    else:
        ranked = sorted(range(len(candidates)), key=lambda idx: supports[idx], reverse=True)[:max(expected_count, 18)]
        ranked = sorted(ranked, key=lambda idx: candidates[idx])
        centers = [candidates[idx] for idx in ranked]

        while len(centers) > expected_count:
            best_remove = None
            best_score = None

            for remove_idx in range(len(centers)):
                trial = centers[:remove_idx] + centers[remove_idx + 1:]
                diffs = np.diff(np.array(trial, dtype=np.float32))

                if len(diffs) == 0:
                    score = 0.0
                else:
                    median_diff = float(np.median(diffs))
                    score = float(np.std(diffs)) / max(1.0, median_diff)

                if best_score is None or score < best_score:
                    best_score = score
                    best_remove = remove_idx

            centers.pop(best_remove)

    if len(centers) >= 2:
        row_step = float(np.median(np.diff(np.array(centers, dtype=np.float32))))
    else:
        row_step = float(image_h) * 0.11

    row_step = max(18.0, row_step)

    while len(centers) < expected_count:
        can_add_top = centers[0] - row_step > image_h * 0.03
        can_add_bottom = centers[-1] + row_step < image_h * 0.97

        if can_add_bottom:
            centers.append(centers[-1] + row_step)
        elif can_add_top:
            centers.insert(0, centers[0] - row_step)
        else:
            top = max(image_h * 0.06, centers[0])
            bottom = min(image_h * 0.94, centers[-1])
            centers = list(np.linspace(top, bottom, expected_count, dtype=np.float32))
            break

    return [float(value) for value in centers[:expected_count]]


def estimate_scan_signal_start_x_from_projection(strong_mask, centers, row_step):
    h, w = strong_mask.shape[:2]
    if w <= 1:
        return 0

    row_mask = np.zeros(h, dtype=bool)
    for top, bottom in build_scan_lane_bounds(centers[:min(6, len(centers))], h):
        y1 = max(0, int(round(top)))
        y2 = min(h, int(round(bottom)) + 1)
        if y2 > y1:
            row_mask[y1:y2] = True

    if not np.any(row_mask):
        row_mask[:] = True

    col_profile = np.sum(strong_mask[row_mask, :] > 0, axis=0).astype(np.float32)
    col_profile = smooth_1d(col_profile, 2.0)

    search_x1 = int(round(w * 0.02))
    search_x2 = int(round(w * 0.35))
    search_x2 = max(search_x1 + 1, min(search_x2, w))
    search = col_profile[search_x1:search_x2]

    if len(search) == 0 or float(np.max(search)) <= 0.0:
        return int(round(w * 0.05))

    threshold = max(
        float(np.percentile(search, 82.0)) * 0.35,
        float(np.mean(search) + 0.25 * np.std(search)),
        1.0,
    )

    active = np.where(search >= threshold)[0]
    if len(active) == 0:
        return int(round(w * 0.05))

    groups = group_consecutive_indices(active, max_gap=3)
    min_width = max(2, int(round(float(row_step) * 0.03)))

    for start, end in groups:
        if end - start + 1 >= min_width:
            return int(np.clip(search_x1 + start - 6, 0, w - 1))

    return int(np.clip(search_x1 + int(active[0]) - 6, 0, w - 1))


def detect_scan_row_centers_from_projection(strong_mask, expected_count=7):
    h, w = strong_mask.shape[:2]
    if h <= 1 or w <= 1:
        centers = [0.0 for _ in range(expected_count)]
        return centers, 0, 1.0

    x1 = int(round(w * 0.04))
    x2 = int(round(w * 0.96))
    if x2 <= x1:
        x1, x2 = 0, w

    roi = strong_mask[:, x1:x2]
    profile = np.sum(roi > 0, axis=1).astype(np.float32)
    profile = smooth_1d(profile, max(2.0, h / 420.0))

    y_min = int(round(h * 0.04))
    y_max = int(round(h * 0.96))
    y_max = max(y_min + 1, min(y_max, h))
    work = profile[y_min:y_max]

    candidates = []
    supports = []

    if len(work) > 0 and float(np.max(work)) > 0.0:
        threshold = max(
            float(np.percentile(work, 72.0)),
            float(np.mean(work) + 0.20 * np.std(work)),
            1.0,
        )
        active = np.where(work >= threshold)[0]

        if len(active) > 0:
            groups = group_consecutive_indices(active, max_gap=max(3, int(round(h * 0.006))))
            min_height = max(3, int(round(h * 0.006)))

            for start, end in groups:
                if end - start + 1 < min_height:
                    continue

                rows = np.arange(start, end + 1, dtype=np.float32)
                weights = work[start:end + 1]
                support = float(np.sum(weights))

                if support <= 0.0:
                    center_y = float(y_min + 0.5 * (start + end))
                else:
                    center_y = float(y_min + np.sum(rows * weights) / support)

                candidates.append(center_y)
                supports.append(support)

        if len(candidates) < expected_count:
            min_distance = max(14, int(round(h * 0.065)))
            peak_indices = []

            for idx in range(1, len(work) - 1):
                if work[idx] < threshold:
                    continue
                if work[idx] < work[idx - 1] or work[idx] <= work[idx + 1]:
                    continue
                peak_indices.append(idx)

            peak_indices = sorted(peak_indices, key=lambda idx: float(work[idx]), reverse=True)
            for peak_idx in peak_indices:
                peak_y = float(y_min + peak_idx)
                if all(abs(peak_y - value) >= min_distance for value in candidates):
                    candidates.append(peak_y)
                    supports.append(float(work[peak_idx]))
                if len(candidates) >= expected_count:
                    break

    centers = choose_regular_scan_centers(candidates, supports, expected_count, h)
    row_diffs = np.diff(np.array(centers, dtype=np.float32))
    valid_diffs = row_diffs[row_diffs > 3.0]

    if len(valid_diffs) > 0:
        row_step = float(np.median(valid_diffs))
    else:
        row_step = max(18.0, float(h) * 0.11)

    signal_start_x = estimate_scan_signal_start_x_from_projection(strong_mask, centers, row_step)
    return centers, signal_start_x, row_step


def detect_scan_left_row_centers(strong_mask):
    h, w = strong_mask.shape
    strip_width = max(70, int(round(w * 0.035)))
    roi = strong_mask[:, :strip_width]

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi, 8)

    components = []

    for label_idx in range(1, num_labels):
        x, y, comp_w, comp_h, area = stats[label_idx]
        if area >= 250 and 20 <= comp_w <= strip_width and 55 <= comp_h <= 140:
            component_mask = (labels[y:y + comp_h, x:x + comp_w] == label_idx).astype(np.uint8) * 255
            baseline_y = estimate_calibration_baseline_from_component(component_mask, int(y), int(comp_w))
            components.append(
                {
                    "center_y": float(baseline_y),
                    "component_center_y": float(centroids[label_idx][1]),
                    "right": int(x + comp_w),
                    "area": int(area),
                }
            )

    if len(components) < 7:
        centers, signal_start_x, row_step = detect_scan_row_centers_from_projection(strong_mask, expected_count=7)
        return centers, signal_start_x, row_step, "projection"

    components = sorted(components, key=lambda item: (-item["area"], item["component_center_y"]))[:7]
    components = sorted(components, key=lambda item: item["component_center_y"])

    centers = [float(item["center_y"]) for item in components]
    right_edges = [int(item["right"]) for item in components]
    row_step = float(np.median(np.diff(np.array(centers, dtype=np.float32))))
    signal_start_x = int(round(float(np.median(np.array(right_edges, dtype=np.float32))))) + 10

    return centers, signal_start_x, row_step, "calibration"

def refine_scan_row_centers_locally(expected_centers, support_image, search_x1, search_x2, row_step):
    refined = [float(center_y) for center_y in expected_centers]

    h, w = support_image.shape[:2]
    x1 = int(np.clip(search_x1, 0, max(0, w - 1)))
    x2 = int(np.clip(search_x2, x1 + 1, w))

    if x2 <= x1:
        return refined

    search_radius = int(max(10, round(float(row_step) * 0.18)))
    max_shift = float(row_step) * 0.22

    for idx, expected_center in enumerate(expected_centers):
        y1 = max(0, int(round(float(expected_center) - search_radius)))
        y2 = min(h, int(round(float(expected_center) + search_radius + 1)))

        if y2 <= y1 + 2:
            continue

        band = support_image[y1:y2, x1:x2].astype(np.float32)
        if band.size == 0:
            continue

        profile = smooth_1d(np.sum(band, axis=1).astype(np.float32), 2.0)
        peak_value = float(np.max(profile)) if len(profile) else 0.0
        mean_value = float(np.mean(profile)) if len(profile) else 0.0

        if peak_value <= max(1.0, mean_value * 1.06):
            continue

        peak_y = float(y1 + int(np.argmax(profile)))
        shift = float(np.clip(peak_y - float(expected_center), -max_shift, max_shift))
        refined[idx] = float(expected_center + 0.70 * shift)

    return refined


def refine_scan_trace_points(center_path, low_path, high_path, height_path, valid, row_step):
    refined = center_path.astype(np.float32).copy()
    valid_indices = np.where(valid & ~np.isnan(center_path))[0]
    scan_edge_refine_height_limit = max(4.5, 0.035 * float(row_step))

    for offset, x_idx in enumerate(valid_indices):
        if float(height_path[x_idx]) >= scan_edge_refine_height_limit:
            refined[x_idx] = float(center_path[x_idx])
            continue

        prev_y = np.nan
        next_y = np.nan

        if offset > 0:
            prev_y = center_path[valid_indices[offset - 1]]

        if offset + 1 < len(valid_indices):
            next_y = center_path[valid_indices[offset + 1]]

        refined[x_idx] = choose_group_point(
            center_y=center_path[x_idx],
            low_y=low_path[x_idx],
            high_y=high_path[x_idx],
            height=height_path[x_idx],
            prev_y=prev_y,
            next_y=next_y,
            row_step=row_step,
        )

    return refined


def detect_scan_divider_x(gray, row_step):
    h, w = gray.shape[:2]
    search_x1 = int(round(w * 0.46))
    search_x2 = int(round(w * 0.54))
    fallback_x = int(round(w * SCAN_DIVIDER_RATIO))

    if search_x2 <= search_x1:
        return fallback_x

    y1 = int(round(h * 0.06))
    y2 = int(round(h * 0.84))
    y2 = max(y1 + 1, min(y2, h))

    roi = (gray[y1:y2, search_x1:search_x2] < SCAN_TRACE_THRESHOLD).astype(np.uint8) * 255

    close_height = max(9, int(round(float(row_step) * 0.22)))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, close_height))
    closed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, close_kernel)

    open_height = max(5, int(round(float(row_step) * 0.10)))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, open_height))
    vertical = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)

    col_activity = np.sum(vertical > 0, axis=0).astype(np.float32)

    if len(col_activity) == 0:
        return fallback_x

    col_activity = smooth_1d(col_activity, 2.0)
    best_idx = int(np.argmax(col_activity))
    best_value = float(col_activity[best_idx])

    if best_value < max(30.0, 0.08 * float(y2 - y1)):
        return fallback_x

    threshold = max(best_value * 0.55, best_value - 25.0)
    strong = np.where(col_activity >= threshold)[0]

    if len(strong) == 0:
        return int(search_x1 + best_idx)

    groups = group_consecutive_indices(strong, max_gap=2)
    group = min(groups, key=lambda pair: abs(best_idx - 0.5 * (pair[0] + pair[1])))
    divider_x = int(round(search_x1 + 0.5 * (group[0] + group[1])))

    return int(np.clip(divider_x, search_x1, search_x2 - 1))

def estimate_scan_side_start_x(trace_mask, score_image, centers, row_step, search_x1, search_x2):
    h, w = trace_mask.shape[:2]
    x1 = int(np.clip(search_x1, 0, max(0, w - 1)))
    x2 = int(np.clip(search_x2, x1 + 1, w))

    if x2 <= x1 + 2:
        return x1

    starts = []
    lane_bounds = build_scan_lane_bounds(centers[:min(6, len(centers))], h)
    min_run = max(5, int(round(float(row_step) * 0.07)))

    for lane_top, lane_bottom in lane_bounds:
        center_y = 0.5 * (float(lane_top) + float(lane_bottom))
        half = max(5.0, float(row_step) * 0.20)
        y1 = max(0, int(round(center_y - half)))
        y2 = min(h, int(round(center_y + half)) + 1)

        if y2 <= y1 + 1:
            continue

        mask_roi = trace_mask[y1:y2, x1:x2] > 0
        score_roi = score_image[y1:y2, x1:x2].astype(np.float32)
        profile = np.sum(mask_roi, axis=0).astype(np.float32)
        profile += 0.012 * np.sum(np.maximum(score_roi, 0.0), axis=0).astype(np.float32)
        profile = smooth_1d(profile, 1.5)

        if len(profile) == 0 or float(np.max(profile)) <= 0.0:
            continue

        threshold = max(
            0.65,
            float(np.percentile(profile, 70.0)) * 0.20,
            float(np.mean(profile)) + 0.10 * float(np.std(profile)),
        )
        active = np.where(profile >= threshold)[0]

        if len(active) == 0:
            continue

        groups = group_consecutive_indices(active, max_gap=4)
        for start, end in groups:
            if end - start + 1 >= min_run:
                starts.append(int(x1 + max(0, start - 4)))
                break

    if not starts:
        return x1

    starts = np.array(starts, dtype=np.float32)
    return int(np.clip(round(float(np.percentile(starts, 25.0))), 0, w - 1))


def estimate_scan_left_start_x(trace_mask, score_image, centers, row_step, divider_x, initial_x, geometry_source):
    h, w = trace_mask.shape[:2]
    right_limit = int(max(1, min(int(divider_x) - 8, round(w * 0.58))))
    initial_x = int(np.clip(round(float(initial_x)), 0, max(0, right_limit - 1)))

    if geometry_source == "calibration":
        return initial_x

    search_x1 = int(max(0, round(float(initial_x) - 0.35 * float(row_step))))
    auto_x = estimate_scan_side_start_x(
        trace_mask=trace_mask,
        score_image=score_image,
        centers=centers[:6],
        row_step=row_step,
        search_x1=search_x1,
        search_x2=right_limit,
    )

    max_shift_left = int(round(0.45 * float(row_step)))
    guarded_x = max(int(auto_x), int(initial_x) - max_shift_left)
    return int(np.clip(guarded_x, 0, max(0, right_limit - 1)))

def build_scan_lane_bounds(centers, image_h):
    centers = [float(c) for c in centers]
    bounds = []

    if not centers:
        return bounds

    for idx, center_y in enumerate(centers):
        if idx > 0:
            top = 0.5 * (float(centers[idx - 1]) + center_y)
        elif len(centers) > 1:
            top = center_y - 0.5 * (float(centers[1]) - center_y)
        else:
            top = center_y - 0.5 * image_h

        if idx + 1 < len(centers):
            bottom = 0.5 * (center_y + float(centers[idx + 1]))
        elif len(centers) > 1:
            bottom = center_y + 0.5 * (center_y - float(centers[idx - 1]))
        else:
            bottom = center_y + 0.5 * image_h

        bounds.append((float(max(0.0, top)), float(min(float(image_h - 1), bottom))))

    return bounds


def build_single_scan_band_bounds(center_y, image_h, row_step, half_height_factor=0.45):
    half_height = max(18.0, float(row_step) * float(half_height_factor))
    top = max(0.0, float(center_y) - half_height)
    bottom = min(float(image_h - 1), float(center_y) + half_height)
    return [(top, bottom)]


def fill_scan_path_full(path, valid):
    result = path.astype(np.float32).copy()
    result_valid = valid.astype(bool).copy()
    indices = np.where(result_valid & ~np.isnan(result))[0]

    if len(indices) == 0:
        return result, result_valid

    first = int(indices[0])
    last = int(indices[-1])
    result[:first] = float(result[first])
    result_valid[:first] = True
    result[last + 1:] = float(result[last])
    result_valid[last + 1:] = True

    idx = first
    while idx <= last:
        if result_valid[idx] and not np.isnan(result[idx]):
            idx += 1
            continue

        start = idx
        while idx <= last and (not result_valid[idx] or np.isnan(result[idx])):
            idx += 1
        end = idx - 1

        left = start - 1
        right = end + 1
        if left >= 0 and right < len(result) and result_valid[left] and result_valid[right]:
            y_left = float(result[left])
            y_right = float(result[right])
            for j in range(start, end + 1):
                alpha = (j - left) / max(1, right - left)
                result[j] = (1.0 - alpha) * y_left + alpha * y_right
                result_valid[j] = True

    return result.astype(np.float32), result_valid


def build_scan_side_column_groups(score_roi, trace_mask_roi, x_idx):
    if x_idx < 0 or x_idx >= score_roi.shape[1]:
        return np.array([], dtype=np.float32), []

    score_col = score_roi[:, x_idx].astype(np.float32)
    mask_col = trace_mask_roi[:, x_idx] > 0
    max_score = float(np.max(score_col)) if len(score_col) else 0.0

    if len(score_col) == 0:
        return score_col, []

    dynamic = max(7.0, 0.42 * max_score)
    candidate_mask = mask_col | (score_col >= dynamic)
    ys = np.where(candidate_mask)[0]
    groups = []

    if len(ys) > 0:
        for group in split_column_points(ys.astype(np.float32), max_gap=4):
            low = float(np.min(group))
            high = float(np.max(group))
            idx1 = int(max(0, round(low)))
            idx2 = int(min(len(score_col), round(high) + 1))

            if idx2 <= idx1:
                continue

            segment_score = score_col[idx1:idx2]
            support = float(np.sum(segment_score))
            height = high - low + 1.0

            groups.append(
                {
                    "y": 0.5 * (low + high),
                    "y_low": low,
                    "y_high": high,
                    "height": height,
                    "support": support,
                    "idx1": idx1,
                    "idx2": idx2,
                }
            )

    if not groups and max_score >= 8.0:
        y_best = float(np.argmax(score_col))
        groups.append(
            {
                "y": y_best,
                "y_low": y_best,
                "y_high": y_best,
                "height": 1.0,
                "support": max_score,
                "idx1": int(y_best),
                "idx2": int(y_best) + 1,
            }
        )

    groups = sorted(groups, key=lambda item: float(item["y"]))
    return score_col, groups


def extract_scan_side_group_columns(score_roi, trace_mask_roi):
    group_columns = []

    for x_idx in range(score_roi.shape[1]):
        score_col, groups = build_scan_side_column_groups(score_roi, trace_mask_roi, x_idx)
        column_groups = []

        for group in groups:
            idx1 = int(group["idx1"])
            idx2 = int(group["idx2"])
            segment_score = score_col[idx1:idx2]
            ys_local = np.arange(idx1, idx2, dtype=np.float32)

            if len(segment_score) > 0:
                weights = np.maximum(segment_score - 0.18 * float(np.max(segment_score)), 0.0)
                if float(np.sum(weights)) > 0.0:
                    y_weighted = float(np.sum(ys_local * weights) / np.sum(weights))
                else:
                    y_weighted = float(group["y"])
            else:
                y_weighted = float(group["y"])

            height = float(group["height"])
            support = float(group["support"])
            density = support / max(1.0, height)

            column_groups.append(
                {
                    "y": float(y_weighted),
                    "y_low": float(group["y_low"]),
                    "y_high": float(group["y_high"]),
                    "height": height,
                    "support": support,
                    "density": float(density),
                }
            )

        column_groups.sort(key=lambda item: float(item["y"]))
        group_columns.append(column_groups)

    return group_columns


def smooth_valid_scan_path(path, valid, protected=None, sigma=0.45):
    result = path.astype(np.float32).copy()
    usable = valid.astype(bool) & ~np.isnan(result)

    if np.sum(usable) < 4:
        return result

    filled, _ = fill_scan_path_full(result, usable)
    smoothed = smooth_1d(filled, sigma)
    smooth_mask = usable if protected is None else usable & (~protected.astype(bool))
    result[smooth_mask] = smoothed[smooth_mask]
    return result.astype(np.float32)


def get_scan_group_tracking_y(group, target_y, row_step, allow_edge_switch=True):
    center_y = float(group["y"])
    low_y = float(group["y_low"])
    high_y = float(group["y_high"])
    height = float(group["height"])

    if not allow_edge_switch or height <= max(3.0, 0.04 * float(row_step)):
        return center_y

    edge_switch_margin = max(1.0, min(0.12 * height, 0.04 * float(row_step)))
    center_dist = abs(center_y - float(target_y))
    low_dist = abs(low_y - float(target_y))
    high_dist = abs(high_y - float(target_y))

    if low_dist + edge_switch_margin < center_dist and low_dist <= high_dist:
        return low_y

    if high_dist + edge_switch_margin < center_dist and high_dist < low_dist:
        return high_y

    return center_y


def track_scan_lane_groups(group_columns, center_y, row_step):
    width = len(group_columns)
    center_path = np.full(width, np.nan, dtype=np.float32)
    low_path = np.full(width, np.nan, dtype=np.float32)
    high_path = np.full(width, np.nan, dtype=np.float32)
    height_path = np.full(width, np.nan, dtype=np.float32)
    support_path = np.full(width, np.nan, dtype=np.float32)
    valid = np.zeros(width, dtype=bool)

    prev_y = float(center_y)
    have_prev = False
    gap_count = 0
    startup_skip = int(max(2, min(8, round(0.04 * float(row_step)))))
    startup_band_width = int(max(16, min(48, round(0.40 * float(row_step)))))
    startup_seed_height_limit = max(4.0, 0.055 * float(row_step))

    for x_idx, groups in enumerate(group_columns):
        if x_idx < startup_skip and not have_prev:
            gap_count += 1
            continue

        target_y = prev_y if have_prev else float(center_y)
        missing_cost = 0.95 + 0.04 * min(int(gap_count), 8)
        best_group = None
        best_track_y = np.nan
        best_cost = missing_cost
        startup_support_floor = 0.0
        startup_thin_seed_exists = False

        if not have_prev and x_idx < startup_band_width and len(groups) > 0:
            max_support = max(float(group["support"]) for group in groups)
            startup_support_floor = max(80.0, 0.32 * max_support)
            startup_thin_seed_exists = any(
                float(group["support"]) >= startup_support_floor
                and float(group["height"]) <= startup_seed_height_limit
                for group in groups
            )

        for group in groups:
            if (
                not have_prev
                and x_idx < startup_band_width
                and startup_thin_seed_exists
                and (
                    float(group["support"]) < startup_support_floor
                    or float(group["height"]) > startup_seed_height_limit
                )
            ):
                continue

            y = get_scan_group_tracking_y(
                group,
                target_y,
                row_step,
                allow_edge_switch=(x_idx < startup_band_width),
            )
            continuity = abs(y - target_y) / max(7.0, 0.24 * float(row_step))
            center_pull = abs(y - float(center_y)) / max(18.0, 1.10 * float(row_step))
            height_penalty = max(0.0, float(group["height"]) - 1.05 * float(row_step)) / max(8.0, 0.45 * float(row_step))
            support_bonus = min(float(group["support"]) / max(26.0, 1.4 * float(row_step)), 2.2)
            density_bonus = min(float(group.get("density", 0.0)) / max(4.0, 0.08 * float(row_step)), 1.4)
            cost = continuity + 0.18 * center_pull + 0.04 * height_penalty - 0.26 * support_bonus - 0.10 * density_bonus

            if cost < best_cost:
                best_cost = cost
                best_group = group
                best_track_y = y

        if best_group is None:
            gap_count += 1
            if have_prev and gap_count >= 3:
                prev_y = 0.92 * prev_y + 0.08 * float(center_y)
            continue

        center_path[x_idx] = float(best_track_y)
        low_path[x_idx] = float(best_group["y_low"])
        high_path[x_idx] = float(best_group["y_high"])
        height_path[x_idx] = float(best_group["height"])
        support_path[x_idx] = float(best_group["support"])
        valid[x_idx] = True
        prev_y = float(best_track_y)
        have_prev = True
        gap_count = 0

    return (
        center_path.astype(np.float32),
        low_path.astype(np.float32),
        high_path.astype(np.float32),
        height_path.astype(np.float32),
        support_path.astype(np.float32),
        valid.astype(bool),
    )


def extract_scan_side_signals(
    trace_mask,
    score_image,
    channel_names,
    centers,
    x1,
    x2,
    row_step,
    lane_bounds,
):
    if len(channel_names) != len(centers) or len(centers) != len(lane_bounds):
        raise ValueError("Scan side geometry is inconsistent")

    image_h = trace_mask.shape[0]
    results = []
    padding = max(8, int(round(0.34 * float(row_step))))

    for channel_idx, channel_name in enumerate(channel_names):
        lane_top, lane_bottom = lane_bounds[channel_idx]
        y1 = max(0, int(round(float(lane_top))) - padding)
        y2 = min(image_h, int(round(float(lane_bottom))) + padding + 1)

        if y2 <= y1 + 1:
            y1 = max(0, int(round(float(centers[channel_idx]) - 0.50 * float(row_step))))
            y2 = min(image_h, int(round(float(centers[channel_idx] + 0.50 * float(row_step)))) + 1)

        local_center = float(centers[channel_idx]) - float(y1)
        trace_roi = trace_mask[y1:y2, x1:x2]
        score_roi = score_image[y1:y2, x1:x2].astype(np.float32)
        group_columns = extract_scan_side_group_columns(score_roi, trace_roi)

        center_path, low_path, high_path, height_path, support_path, valid = track_scan_lane_groups(
            group_columns=group_columns,
            center_y=local_center,
            row_step=row_step,
        )

        final_local = refine_scan_trace_points(center_path, low_path, high_path, height_path, valid, row_step)
        final_path = final_local.astype(np.float32) + float(y1)
        guide_path = center_path.astype(np.float32) + float(y1)
        guide_valid = valid.astype(bool).copy()

        if np.sum(guide_valid) > 0:
            guide_path, guide_valid = fill_short_gaps(guide_path, guide_valid, row_step, max_gap=8)
            guide_path, _ = fill_scan_path_full(guide_path, guide_valid)
        else:
            guide_path = np.full(x2 - x1, float(centers[channel_idx]), dtype=np.float32)

        protected = np.zeros(len(final_path), dtype=bool)
        usable = valid & ~np.isnan(height_path) & ~np.isnan(support_path)
        if np.any(usable):
            strong = (
                usable
                & (height_path >= max(3.0, 0.14 * float(row_step)))
                & (support_path >= max(25.0, float(np.percentile(support_path[usable], 40.0))))
            )
            if np.any(strong):
                protected = np.convolve(strong.astype(np.uint8), np.array([1, 1, 1], dtype=np.uint8), mode="same") > 0

        final_path, valid = fill_short_gaps(final_path, valid, row_step, max_gap=6)
        final_path = suppress_narrow_scan_spikes(final_path, valid, row_step, protected=protected)
        final_path, valid = fill_short_gaps(final_path, valid, row_step, max_gap=6)
        final_path = smooth_valid_scan_path(final_path, valid, protected=protected, sigma=0.45)

        signal, baseline_y = convert_y_to_signal(
            y_values=final_path,
            valid=valid,
            center_y=float(centers[channel_idx]),
            row_step=row_step,
        )

        results.append(
            {
                "channel_name": channel_name,
                "guide_path": guide_path.astype(np.float32),
                "final_path": final_path.astype(np.float32),
                "valid": valid.astype(bool),
                "signal": signal.astype(np.float32),
                "baseline_y": float(baseline_y),
            }
        )

    return results


def suppress_narrow_scan_spikes(path, valid, row_step, protected=None):
    result = path.astype(np.float32).copy()
    usable = valid & ~np.isnan(result)
    if protected is None:
        protected = np.zeros(len(result), dtype=bool)
    else:
        protected = protected.astype(bool).copy()
    spike_threshold = max(4.0, row_step * 0.09)
    stable_threshold = max(2.0, row_step * 0.04)

    if np.sum(usable) == 0:
        return result

    for _ in range(2):
        for x_idx in range(1, len(result) - 1):
            if not usable[x_idx - 1] or not usable[x_idx] or not usable[x_idx + 1]:
                continue
            if protected[x_idx]:
                continue

            predicted = 0.5 * (float(result[x_idx - 1]) + float(result[x_idx + 1]))
            if (
                abs(float(result[x_idx]) - predicted) > spike_threshold
                and abs(float(result[x_idx + 1]) - float(result[x_idx - 1])) <= stable_threshold
            ):
                result[x_idx] = predicted

        for x_idx in range(1, len(result) - 2):
            if not usable[x_idx - 1] or not usable[x_idx] or not usable[x_idx + 1] or not usable[x_idx + 2]:
                continue
            if protected[x_idx] or protected[x_idx + 1]:
                continue

            left = float(result[x_idx - 1])
            right = float(result[x_idx + 2])
            if abs(right - left) > stable_threshold:
                continue

            expected_1 = left + (right - left) / 3.0
            expected_2 = left + 2.0 * (right - left) / 3.0
            if (
                abs(float(result[x_idx]) - expected_1) > spike_threshold
                and abs(float(result[x_idx + 1]) - expected_2) > spike_threshold
            ):
                result[x_idx] = expected_1
                result[x_idx + 1] = expected_2

    filled = result.copy()
    valid_indices = np.where(usable)[0]

    for idx in range(len(filled)):
        if usable[idx]:
            continue
        left_candidates = valid_indices[valid_indices < idx]
        right_candidates = valid_indices[valid_indices > idx]
        if len(left_candidates) > 0 and len(right_candidates) > 0:
            left_idx = int(left_candidates[-1])
            right_idx = int(right_candidates[0])
            alpha = (idx - left_idx) / max(right_idx - left_idx, 1)
            filled[idx] = (1.0 - alpha) * float(result[left_idx]) + alpha * float(result[right_idx])
        elif len(left_candidates) > 0:
            filled[idx] = float(result[int(left_candidates[-1])])
        else:
            filled[idx] = float(result[int(right_candidates[0])])

    smoothed = smooth_1d(filled, 0.45)
    smooth_mask = usable & (~protected)
    result[smooth_mask] = smoothed[smooth_mask]
    return result


def build_scan_render_config(left_baselines, right_baselines, left_x1, left_x2, right_x1, right_x2, px_per_ms):
    render_config = build_render_config(CHANNELS[:6], left_baselines, left_x1, left_x2, px_per_ms)
    render_config.update(build_render_config(CHANNELS[6:], right_baselines, right_x1, right_x2, px_per_ms))
    return render_config


def add_scan_results(results, px_per_ms, channel_resampled, baselines):
    for item in results:
        baselines.append(float(item["baseline_y"]))
        channel_resampled[item["channel_name"]] = resample_to_1ms(item["signal"], px_per_ms)


def extract_scan_block(trace_mask, score_image, channel_names, centers, x1, x2, row_step, image_h):
    return extract_scan_side_signals(
        trace_mask=trace_mask,
        score_image=score_image,
        channel_names=channel_names,
        centers=centers,
        x1=x1,
        x2=x2,
        row_step=row_step,
        lane_bounds=build_scan_lane_bounds(centers, image_h),
    )

def process_scan_image(image_path, output_dir, debug_dir=None):
    original_image = load_image(image_path)

    if is_probably_screen_capture(original_image):
        print("Detected WorkMate screen capture in scan mode; switching to screen pipeline.")
        process_screen_image(image_path, output_dir, debug_dir)
        return

    if is_screen_sized(original_image):
        print("Full HD image is not a dark WorkMate screenshot; processing it with scan pipeline.")

    image = prepare_scan_image(original_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    norm_gray = normalize_scan_gray(gray)
    row_mask, geometry_score = create_scan_geometry_images(norm_gray)

    left_centers, signal_start_x, row_step, geometry_source = detect_scan_left_row_centers(row_mask)
    blackhat, trace_mask, score_image = create_scan_trace_images(norm_gray, gray, row_step, image=image)

    big_grid_px = estimate_scan_big_grid_px(norm_gray)
    px_per_ms = float(big_grid_px / 200.0)
    h, w = image.shape[:2]

    divider_x = detect_scan_divider_x(gray, row_step)
    left_x1 = estimate_scan_left_start_x(
        trace_mask=trace_mask,
        score_image=score_image,
        centers=left_centers,
        row_step=row_step,
        divider_x=divider_x,
        initial_x=signal_start_x,
        geometry_source=geometry_source,
    )
    left_x2 = max(left_x1 + 1, int(divider_x) - 2)
    right_x1 = min(w - 2, int(divider_x) + 2)
    right_x2 = w

    row_support = geometry_score + 0.18 * np.maximum(score_image, 0.0)
    right_centers = refine_scan_row_centers_locally(
        expected_centers=left_centers[:6],
        support_image=row_support,
        search_x1=right_x1 + max(20, int(round(float(row_step) * 0.22))),
        search_x2=w - 16,
        row_step=row_step,
    )

    channel_resampled = {}
    left_baselines = []
    right_baselines = []

    left_results = extract_scan_block(trace_mask, score_image, CHANNELS[:6], left_centers[:6], left_x1, left_x2, row_step, h)
    right_results = extract_scan_block(trace_mask, score_image, CHANNELS[6:], right_centers[:6], right_x1, right_x2, row_step, h)

    add_scan_results(left_results, px_per_ms, channel_resampled, left_baselines)
    add_scan_results(right_results, px_per_ms, channel_resampled, right_baselines)

    rhythm_center_y = float(left_centers[6])
    rhythm_bounds = build_single_scan_band_bounds(rhythm_center_y, h, row_step, SCAN_RHYTHM_BAND_FACTOR)
    rhythm_result = extract_scan_side_signals(
        trace_mask=trace_mask,
        score_image=score_image,
        channel_names=["rhythm_II"],
        centers=[rhythm_center_y],
        x1=left_x1,
        x2=w,
        row_step=row_step,
        lane_bounds=rhythm_bounds,
    )[0]
    _, rhythm_signal_1ms = resample_to_1ms(rhythm_result["signal"], px_per_ms)

    df, _, aligned_channels = build_output_dataframe(channel_resampled)
    output_path = output_dir / f"{image_path.stem}.csv"
    df.to_csv(output_path, index=False)

    if debug_dir is not None:
        render_config = build_scan_render_config(left_baselines, right_baselines, left_x1, left_x2, right_x1, right_x2, px_per_ms)
        save_debug_csv_renders(output_path, image, render_config, image_path, debug_dir)

    bpm, _ = estimate_bpm_from_signal(rhythm_signal_1ms)
    bpm_source = "rhythm II"

    if bpm is None:
        bpm, _ = estimate_bpm_from_channels(aligned_channels)
        bpm_source = "lead ensemble"

    print(
        f"Scan geometry: rows={geometry_source}, start_x={left_x1}, divider_x={divider_x}"
    )
    print(f"Using px_per_ms={px_per_ms:.6f} (scan_grid:{big_grid_px:.2f}px)")
    print(f"Saved: {output_path}")
    if bpm is not None:
        print(f"BPM: {bpm:.1f} ({bpm_source})")
    else:
        print("BPM: unavailable")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=[SCAN_MODE_SCREEN, SCAN_MODE_SCAN], default=SCAN_MODE_SCREEN)
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    debug_dir = output_dir / "debug"

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths(input_dir)

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    image_paths = filter_image_paths_for_mode(image_paths, args.mode)

    if not image_paths:
        print(f"No {args.mode} images found in {input_dir}")
        return

    for image_path in image_paths:
        try:
            print(f"Processing: {image_path.name}")
            if args.mode == SCAN_MODE_SCREEN:
                process_screen_image(image_path, output_dir, debug_dir)
            else:
                process_scan_image(image_path, output_dir, debug_dir)
        except Exception as error:
            print(f"Error while processing {image_path.name}: {error}")


if __name__ == "__main__":
    main()
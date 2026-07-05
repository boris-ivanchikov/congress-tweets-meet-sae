import math
import os
import textwrap

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import STRtree
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from shapely.affinity import translate
from shapely.geometry import Point, box
from shapely.ops import polylabel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CMAP = "RdBu_r"
MISSING_KWDS = {"color": "white", "edgecolor": "lightgrey", "hatch": "///"}
HALO = [pe.withStroke(linewidth=1.6, foreground="white")]
LEGEND_COLS = 5
PARTY_COLORS = {"R": "#c0392b", "D": "#2e5fa3"}
LABEL_BOX = (1.6, 0.8)
LEGEND_FONT_SIZE = 12.0
MAP_KEY_FONT_SIZE = 14.0
NAME_WRAP = 27
STAGGER_X = 6.0
TOP_SHIFT = (6.0, 1.5)
MANUAL_OFFSETS = {}
TITLE_MAPS = "Congressional District Partisanship (Cook PVI)"
SUBTITLE_TOP = "Before redistricting"
SUBTITLE_TOP_DATES = "(116th Congress, 2019–2021)"
SUBTITLE_BOTTOM = "After redistricting"
SUBTITLE_BOTTOM_DATES = "(117th Congress, 2021–2023)"
TITLE_LIST = "Incumbent U.S. House Representatives by State"


def infer_district(stateab, cdlabel):
    if cdlabel == stateab:
        cdlabel = "01"
    if len(cdlabel) == 1:
        cdlabel = "0" + cdlabel
    return stateab + "-" + cdlabel


def load_data():
    data = pd.read_csv(os.path.join(ROOT, "data", "representatives.csv"))
    data = data.drop_duplicates(subset=["name"])
    data["pvi_change"] = data["cook_pvi_new"] - data["cook_pvi_old"]
    data = data.sort_values(["state", "pvi_change"]).reset_index(drop=True)
    data["rank"] = range(1, len(data) + 1)
    return data


def load_maps(data):
    def load(cd_dir, st_dir, district_col, pvi_col):
        districts = gpd.read_file(os.path.join(ROOT, "data", "maps", cd_dir, cd_dir + ".shp"))
        states = gpd.read_file(os.path.join(ROOT, "data", "maps", st_dir, st_dir + ".shp"))
        districts[district_col] = districts.apply(lambda x: infer_district(x["STATEAB"], x["CDLABEL"]), axis=1)
        agg = (
            data.groupby(district_col)
            .agg(cook_pvi=(pvi_col, "mean"), ranks=("rank", list))
            .reset_index()
        )
        districts = districts.merge(agg, on=district_col, how="left")
        districts = districts.set_geometry(districts.geometry.simplify(0.02))
        states = states.set_geometry(states.geometry.simplify(0.02))
        return districts, states

    old = load("HexCDv21", "HexSTv20", "district_old", "cook_pvi_old")
    new = load("HexCDv30", "HexSTv30", "district_new", "cook_pvi_new")
    return old, new


def interlock_offset(u_top, u_bottom, margin=1.0, step=0.25, max_steps=200):
    yoff = (u_top.bounds[1] - margin) - u_bottom.bounds[3]
    for _ in range(max_steps):
        candidate = yoff + step
        if translate(u_bottom, yoff=candidate).distance(u_top) > margin:
            yoff = candidate
        else:
            break
    return yoff


def optimize_state_labels(states_list, number_xy, w=1.6, h=0.8, n_angles=48, extras=(0.25, 0.5, 0.8, 1.2, 1.8, 2.5)):
    geoms, abs_, map_is = [], [], []
    for i, states in enumerate(states_list):
        for _, row in states.iterrows():
            geoms.append(row.geometry)
            abs_.append(row["STATEAB"])
            map_is.append(i)
    tree = STRtree(geoms)

    per_state = {}
    for g, ab, i in zip(geoms, abs_, map_is):
        per_state.setdefault(ab, []).append((i, g, g.representative_point(), g.centroid))

    def box_penalties(b, ab, i):
        own, other = 0.0, 0.0
        for j in tree.query(b):
            a = geoms[j].intersection(b).area
            if abs_[j] == ab and map_is[j] == i:
                own += a
            else:
                other += a
        return own, other

    placed = []
    deltas = {}
    order = sorted(per_state, key=lambda ab: per_state[ab][0][1].area)
    for ab in order:
        entries = per_state[ab]
        _, g0, rp0, _ = entries[0]
        best_score, best_delta = None, (0.0, 0.0)
        for k in range(n_angles):
            angle = 2 * math.pi * k / n_angles
            ux, uy = math.cos(angle), math.sin(angle)
            t = 0.0
            while t < 30 and Point(rp0.x + ux * t, rp0.y + uy * t).within(g0):
                t += 0.25
            for extra in extras:
                dx, dy = ux * (t + extra), uy * (t + extra)
                score = -0.8 * max(uy, 0.0)
                for i, own, rp, c in entries:
                    cx, cy = rp.x + dx, rp.y + dy
                    b = box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                    ov_own, ov_other = box_penalties(b, ab, i)
                    score += ov_other * 300 + ov_own * 120

                    nb = box(cx - w / 2 - 1.0, cy - h / 2 - 1.0, cx + w / 2 + 1.0, cy + h / 2 + 1.0)
                    nb_own, nb_other = box_penalties(nb, ab, i)
                    score += nb_other * 16 - nb_own * 6

                    nx, ny = number_xy[i]
                    covered = ((nx > cx - w / 2) & (nx < cx + w / 2) & (ny > cy - h / 2) & (ny < cy + h / 2)).sum()
                    score += covered * 60

                    gap = b.distance(own)
                    score += abs(gap - 0.25) * 30 + max(0.0, gap - 0.5) ** 2 * 25

                    align = min(abs(cx - c.x), abs(cy - c.y))
                    score -= 14.0 * max(0.0, 1.0 - align)

                    for px, py in placed:
                        d = math.hypot(cx - px, cy - py)
                        if d < 3.5:
                            score += (3.5 - d) * 10
                if best_score is None or score < best_score:
                    best_score, best_delta = score, (dx, dy)
        deltas[ab] = best_delta
        for i, own, rp, c in entries:
            placed.append((rp.x + best_delta[0], rp.y + best_delta[1]))
    return deltas


def label_point(geom):
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    try:
        return polylabel(geom, tolerance=0.05)
    except Exception:
        return geom.representative_point()


def district_number_xy(districts):
    xs, ys = [], []
    for _, row in districts.iterrows():
        if not isinstance(row["ranks"], list):
            continue
        if not [r for r in row["ranks"] if pd.notna(r)]:
            continue
        p = label_point(row.geometry)
        xs.append(p.x)
        ys.append(p.y)
    return np.array(xs), np.array(ys)


def draw_map(districts, states, ax, norm):
    districts.plot(
        ax=ax,
        column="cook_pvi",
        cmap=CMAP,
        norm=norm,
        edgecolor="grey",
        linewidth=0.8,
        missing_kwds=MISSING_KWDS,
    )
    states.plot(ax=ax, color="none", edgecolor="dimgrey", linewidth=1.5)

    for _, row in districts.iterrows():
        if not isinstance(row["ranks"], list):
            continue
        ranks = [int(r) for r in row["ranks"] if pd.notna(r)]
        if not ranks:
            continue
        pt = label_point(row.geometry)
        ax.annotate(
            ",".join(str(r) for r in ranks),
            (pt.x, pt.y),
            ha="center",
            va="center",
            fontsize=6.5,
            color="black",
            path_effects=HALO,
        )


def draw_state_labels(states, deltas, ax):
    for _, row in states.iterrows():
        ab = row["STATEAB"]
        rp = row.geometry.representative_point()
        dx, dy = deltas[ab]
        mx, my = MANUAL_OFFSETS.get(ab, (0.0, 0.0))
        ax.annotate(
            ab,
            (rp.x + dx + mx, rp.y + dy + my),
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="0.35",
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
        )


def plot_maps(maps, deltas, ax, norm):
    unions = [states.geometry.union_all() for _, states in maps]
    for districts, states in maps:
        draw_map(districts, states, ax, norm)
    for districts, states in maps:
        draw_state_labels(states, deltas, ax)

    lw, lh = LABEL_BOX
    xs, ys = [], []
    for u in unions:
        xs += [u.bounds[0], u.bounds[2]]
        ys += [u.bounds[1], u.bounds[3]]
    for _, states in maps:
        for _, row in states.iterrows():
            rp = row.geometry.representative_point()
            dx, dy = deltas[row["STATEAB"]]
            mx, my = MANUAL_OFFSETS.get(row["STATEAB"], (0.0, 0.0))
            xs += [rp.x + dx + mx - lw / 2, rp.x + dx + mx + lw / 2]
            ys += [rp.y + dy + my - lh / 2, rp.y + dy + my + lh / 2]
    ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    ax.set_aspect("equal")
    ax.set_axis_off()


def plot_colorbar(ax, fig, norm, x_data, y_bottom_data, y_top_data, swatch_y_fig):
    fig.draw_without_rendering()
    to_ax = ax.transAxes.inverted().transform
    x_ax, y_bot = to_ax(ax.transData.transform((x_data, y_bottom_data)))
    _, y_top = to_ax(ax.transData.transform((x_data, y_top_data)))
    pane_w = ax.get_position().width * fig.get_size_inches()[0]
    pane_h = ax.get_position().height * fig.get_size_inches()[1]
    w, h = 1.6 * LEGEND_FONT_SIZE / 72 / pane_w, y_top - y_bot
    y_ax = (y_bot + y_top) / 2
    cax = ax.inset_axes([x_ax, y_ax - h / 2, w, h])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation="vertical")
    cb.ax.tick_params(labelsize=MAP_KEY_FONT_SIZE)
    cb.outline.set_edgecolor("grey")
    cb.outline.set_linewidth(0.6)
    for text, y, ha in [
        ("Republican lean", y_ax + h / 2, "right"),
        ("Cook PVI", y_ax, "center"),
        ("Democratic lean", y_ax - h / 2, "left"),
    ]:
        ax.text(
            x_ax - 0.011,
            y,
            text,
            transform=ax.transAxes,
            fontsize=MAP_KEY_FONT_SIZE,
            color="0.25",
            va="center",
            ha=ha,
            rotation=90,
            rotation_mode="anchor",
        )

    sq = w
    sq_h = sq * pane_w / pane_h
    y_c = to_ax(fig.transFigure.transform((0, swatch_y_fig)))[1]
    ax.add_patch(Rectangle(
        (x_ax, y_c - sq_h / 2),
        sq,
        sq_h,
        facecolor="white",
        edgecolor="lightgrey",
        hatch="///",
        transform=ax.transAxes,
    ))
    ax.text(
        x_ax + sq + 0.008,
        y_c,
        "No incumbent",
        transform=ax.transAxes,
        fontsize=MAP_KEY_FONT_SIZE,
        va="center",
        ha="left",
    )


def legend_blocks(data, state_names, norm):
    cmap = plt.get_cmap(CMAP)
    blocks = []
    for state, group in data.groupby("state", sort=True):
        block = [{"kind": "header", "text": state_names.get(state, state), "rows": 1.15}]
        for _, row in group.iterrows():
            change = int(row["pvi_change"])
            rank_str = f"{int(row['rank'])}. "
            lines = textwrap.wrap(row["name"], width=max(NAME_WRAP - len(rank_str), 10))
            block.append({
                "kind": "entry",
                "rank_str": rank_str,
                "lines": lines,
                "rows": len(lines),
                "party": row["party"],
                "change": f"{change:+d}" if change else "0",
                "change_color": cmap(norm(row["cook_pvi_new"])),
            })
        block.append({"kind": "spacer", "rows": 0.5})
        blocks.append(block)
    return blocks


def block_rows(block):
    return sum(item.get("rows", 1) for item in block)


def pack_blocks(blocks, ncols):
    lo = max(block_rows(b) for b in blocks)
    hi = sum(block_rows(b) for b in blocks)

    def fits(cap):
        cols, used = 1, 0
        for b in blocks:
            if used + block_rows(b) > cap:
                cols += 1
                used = 0
            used += block_rows(b)
        return cols <= ncols

    for _ in range(50):
        mid = (lo + hi) / 2
        if fits(mid):
            hi = mid
        else:
            lo = mid
    cap = hi

    columns, current, used = [], [], 0
    for b in blocks:
        if used + block_rows(b) > cap:
            columns.append(current)
            current, used = [], 0
        current.extend(b)
        used += block_rows(b)
    columns.append(current)
    return columns, cap


def plot_legend(data, state_names, ax, norm, fig):
    ax.set_axis_off()

    blocks = legend_blocks(data, state_names, norm)
    columns, _ = pack_blocks(blocks, LEGEND_COLS)
    columns = [
        [item for i, item in enumerate(items) if not (item["kind"] == "spacer" and i == len(items) - 1)]
        for items in columns
    ]
    top, bottom = 0.94, 0.06
    dys = [(top - bottom) / max(block_rows(items) - 1, 1) for items in columns]

    fig.draw_without_rendering()
    pane_w = ax.get_position().width * fig.get_size_inches()[0]
    fs_entry = LEGEND_FONT_SIZE
    fs_header = fs_entry + 0.5
    char_frac = fs_entry * 0.52 / 72 / pane_w

    for col, items in enumerate(columns):
        x = col / LEGEND_COLS + 0.008
        y = top
        for item in items:
            if item["kind"] == "header":
                ax.text(
                    x + 4.2 * char_frac,
                    y,
                    item["text"],
                    transform=ax.transAxes,
                    fontsize=fs_header,
                    fontweight="bold",
                    va="top",
                )
            elif item["kind"] == "entry":
                color = PARTY_COLORS.get(item["party"], "0.1")
                num_end = x + 3.2 * char_frac
                name_x = num_end + char_frac
                ax.text(
                    num_end,
                    y,
                    item["change"],
                    transform=ax.transAxes,
                    fontsize=fs_entry,
                    va="top",
                    ha="right",
                    color="0.25",
                )
                indent = len(item["rank_str"]) * char_frac
                for j, line in enumerate(item["lines"]):
                    ax.text(
                        name_x + (indent if j else 0),
                        y - j * dys[col],
                        item["rank_str"] + line if j == 0 else line,
                        transform=ax.transAxes,
                        fontsize=fs_entry,
                        va="top",
                        color=color,
                    )
            y -= item.get("rows", 1) * dys[col]


def add_titles(axd, fig, mn_top_old, mn_top_new):
    ax_maps, ax_list = axd["left"], axd["right"]
    ax_maps.text(0.5, 0.975, TITLE_MAPS, transform=ax_maps.transAxes, ha="center", va="top", fontsize=20, fontweight="bold", color="0.1")
    line_h = 0.017
    t_main = ax_maps.text(0.5, 0.912 + line_h, SUBTITLE_TOP, transform=ax_maps.transAxes, ha="center", va="top", fontsize=16, color="0.35")
    t_date = ax_maps.text(0.5, 0.912, SUBTITLE_TOP_DATES, transform=ax_maps.transAxes, ha="center", va="top", fontsize=12.5, color="0.45")
    ax_list.text(0.5, 0.975, TITLE_LIST, transform=ax_list.transAxes, ha="center", va="top", fontsize=20, fontweight="bold", color="0.1")

    fig.draw_without_rendering()
    to_frac = ax_maps.transAxes.inverted()
    bb_main = t_main.get_window_extent().transformed(to_frac)
    bb_date = t_date.get_window_extent().transformed(to_frac)
    pair_gap = bb_main.y0 - bb_date.y1
    map_gap = bb_date.y0 - mn_top_old

    b_date = ax_maps.text(0.5, mn_top_new + map_gap, SUBTITLE_BOTTOM_DATES, transform=ax_maps.transAxes, ha="center", va="bottom", fontsize=12.5, color="0.45")
    fig.draw_without_rendering()
    bb_b_date = b_date.get_window_extent().transformed(to_frac)
    ax_maps.text(0.5, bb_b_date.y1 + pair_gap, SUBTITLE_BOTTOM, transform=ax_maps.transAxes, ha="center", va="bottom", fontsize=16, color="0.35")


def add_party_legend(ax, fig):
    handles = [
        Patch(facecolor=PARTY_COLORS["D"], edgecolor="none", label="Democrat"),
        Patch(facecolor=PARTY_COLORS["R"], edgecolor="none", label="Republican"),
    ]
    leg = ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.008, 0.016),
        ncols=2,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.0,
        handleheight=1.0,
        columnspacing=1.4,
    )
    fig.draw_without_rendering()
    bb = leg.get_window_extent().transformed(ax.transAxes.inverted())
    ax.text(
        bb.x1 + 0.015,
        (bb.y0 + bb.y1) / 2,
        "Numbers before names match those on the maps; grey numbers give the change in Cook PVI",
        transform=ax.transAxes,
        fontsize=LEGEND_FONT_SIZE,
        color="0.35",
        va="center",
    )
    bb_fig = leg.get_window_extent().transformed(fig.transFigure.inverted())
    return (bb_fig.y0 + bb_fig.y1) / 2


def draw_separator(fig, axd):
    fig.draw_without_rendering()
    left = axd["left"].get_position()
    right = axd["right"].get_position()
    x_mid = (left.x1 + right.x0) / 2
    fig.add_artist(
        Line2D(
            [x_mid, x_mid],
            [min(left.y0, right.y0), max(left.y1, right.y1)],
            transform=fig.transFigure,
            color="0.8",
            linewidth=1,
        )
    )


def main():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

    data = load_data()
    (districts_old, states_old), (districts_new, states_new) = load_maps(data)
    state_names = dict(districts_old[["STATEAB", "STATENAME"]].drop_duplicates().values)

    u_old = states_old.geometry.union_all()
    u_new = translate(states_new.geometry.union_all(), xoff=STAGGER_X - TOP_SHIFT[0])
    yoff = interlock_offset(u_old, u_new)
    districts_new = districts_new.set_geometry(districts_new.geometry.translate(xoff=STAGGER_X, yoff=yoff))
    states_new = states_new.set_geometry(states_new.geometry.translate(xoff=STAGGER_X, yoff=yoff))
    districts_old = districts_old.set_geometry(districts_old.geometry.translate(xoff=TOP_SHIFT[0], yoff=TOP_SHIFT[1]))
    states_old = states_old.set_geometry(states_old.geometry.translate(xoff=TOP_SHIFT[0], yoff=TOP_SHIFT[1]))
    u_old = states_old.geometry.union_all()
    number_xy = [district_number_xy(districts_old), district_number_xy(districts_new)]
    deltas = optimize_state_labels([states_old, states_new], number_xy)

    vmax = max(districts_old["cook_pvi"].abs().max(), districts_new["cook_pvi"].abs().max())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, axd = plt.subplot_mosaic(
        [["left", "right"]],
        width_ratios=[2.85, 2.29],
        figsize=(31, 19),
        layout="constrained",
    )
    fig.get_layout_engine().set(rect=(0, 0, 0.985, 1))

    plot_maps([(districts_old, states_old), (districts_new, states_new)], deltas, axd["left"], norm)
    xl = axd["left"].get_xlim()
    axd["left"].set_xlim(xl[0] - 5.5, xl[1] + 0.0)
    plot_legend(data, state_names, axd["right"], norm, fig)
    swatch_y_fig = add_party_legend(axd["right"], fig)
    hi_bottom = states_new.loc[states_new["STATEAB"] == "HI"].geometry.iloc[0].bounds[1]
    wa_top = states_old.loc[states_old["STATEAB"] == "WA"].geometry.iloc[0].bounds[3]
    cb_x = states_new.geometry.union_all().bounds[0] - STAGGER_X + 1.5
    plot_colorbar(axd["left"], fig, norm, cb_x, hi_bottom, wa_top, swatch_y_fig)
    fig.draw_without_rendering()
    to_frac = axd["left"].transAxes.inverted().transform

    def mn_label_top(states):
        row = states.loc[states["STATEAB"] == "MN"].iloc[0]
        rp = row.geometry.representative_point()
        my = MANUAL_OFFSETS.get("MN", (0.0, 0.0))[1]
        y = rp.y + deltas["MN"][1] + my + LABEL_BOX[1] / 2
        return to_frac(axd["left"].transData.transform((0, y)))[1]

    add_titles(axd, fig, mn_label_top(states_old), mn_label_top(states_new))
    draw_separator(fig, axd)

    out = os.path.join(ROOT, "thesis", "redistricting.png")
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"Saved {out}")

    out_pgf = os.path.join(ROOT, "thesis", "tex", "redistricting.pgf")
    try:
        import matplotlib as mpl

        with mpl.rc_context({
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "axes.unicode_minus": False,
        }):
            fig.savefig(out_pgf, format="pgf", facecolor="white", dpi=100)
        print(f"Saved {out_pgf}")
    except Exception as e:
        print(f"PGF export failed: {e}")


if __name__ == "__main__":
    main()

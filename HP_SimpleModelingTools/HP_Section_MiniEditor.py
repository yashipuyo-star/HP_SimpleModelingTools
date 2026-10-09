
bl_info = {
    "name": "HP Section Mini Editor",
    "author": "OpenAI + yashi",
    "version": (0, 27, 2),
    "blender": (4, 3, 0),
    "location": "3D View > Sidebar > HP Tools",
    "description": "Two fully interactive section views with translucent world-plane editing.",
    "category": "Mesh",
}

import bpy
from .HP_Section_Preview import SectionPreviewMixin
import time
import traceback
import copy
from . import HP_Window_Placement
from bpy.props import EnumProperty, BoolProperty
import bmesh
import blf
import gpu
from gpu_extras.batch import batch_for_shader

# ------------------------------------------------------------------------
# Session memory for View A SECTION orientation.
# Kept in-module so it survives editor reopen / selection changes during the
# Blender session without writing persistent preferences into the .blend.
# ------------------------------------------------------------------------
_LAST_SECTION_DOWN_WORLD = None
_LAST_SECTION_RIGHT_WORLD = None
_LAST_SECTION_ORIENT_MODE = 'NONE'
_LAST_SECTION_FLIP_Y = False

from mathutils import Vector, Quaternion
from math import cos, sin, tau, atan2, ceil
from bpy_extras.view3d_utils import location_3d_to_region_2d

HP_PENFIX_DEBUG = False


def _hp_log(message):
    if HP_PENFIX_DEBUG:
        print(f"[HP_PENFIX] {message}")


def _hp_follow_log(message):
    line = f"[HP_FOLLOW] {message}"
    print(line, flush=True)
    try:
        block = bpy.data.texts.get('HP_Follow_Diagnostics')
        if block is None:
            block = bpy.data.texts.new('HP_Follow_Diagnostics')
        block.write(line + '\n')
    except (AttributeError, ReferenceError, RuntimeError):
        pass


_HP_SETTINGS_DEFAULTS = dict(
    correction=0.0, amount=1.0, shape=False,
    side_a=0, side_b=0, falloff=1.0,
    stop_triangles=True, blend_strength=0.65, blend_radius=2,
    direct_follow=False, follow_strength_a=1.0, follow_strength_b=1.0,
)


def _hp_settings_complete(saved=None):
    settings = dict(_HP_SETTINGS_DEFAULTS)
    if saved:
        settings.update(saved)
    return settings

_RUNNING = False
_ACTIVE_SECTION_EDITOR = None
_PINNED_WORKSPACE_PTR = 0
_SECTION_PEN_ANCHOR_DEFAULT = 'STROKE'
PEN_F_HOLD_SECONDS = 0.45

DEFAULT_PANEL_W = 640
DEFAULT_PANEL_H = 480
PANEL_MIN_W = 280
PANEL_MIN_H = 220
PANEL_MAX_W = 820
PANEL_MAX_H = 620
WINDOW_STAGES = [(320, 240), (480, 360), (640, 480)]
PANEL_X = 18
PANEL_Y = 118
PANEL_PAD = 34


def _selected_chain(context):
    obj = context.edit_object
    if not obj or obj.type != 'MESH' or obj.mode != 'EDIT':
        return None

    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.index_update()
    bm.edges.index_update()

    edges = [e for e in bm.edges if e.select]
    if not edges:
        return None

    adj = {}
    for e in edges:
        a, b = e.verts
        adj.setdefault(a.index, []).append((b.index, e.index))
        adj.setdefault(b.index, []).append((a.index, e.index))

    if any(len(v) > 2 for v in adj.values()):
        return None

    verts_in = set(adj.keys())
    seen = set()
    stack = [next(iter(verts_in))]
    while stack:
        vi = stack.pop()
        if vi in seen:
            continue
        seen.add(vi)
        stack.extend(n for n, _ in adj[vi] if n not in seen)

    if seen != verts_in:
        return None

    endpoints = [vi for vi, links in adj.items() if len(links) == 1]
    closed = len(endpoints) == 0
    if not closed and len(endpoints) != 2:
        return None

    start = min(endpoints) if endpoints else min(verts_in)
    ordered = [start]
    prev = None
    cur = start

    for _ in range(len(verts_in) + 2):
        candidates = [n for n, _ in adj[cur] if n != prev]
        if not candidates:
            break

        if closed and len(ordered) < len(verts_in):
            non_start = [n for n in candidates if n != start]
            if non_start:
                candidates = non_start

        nxt = min(candidates)

        if closed and nxt == start:
            break
        if nxt in ordered:
            break

        ordered.append(nxt)
        prev, cur = cur, nxt

    if len(ordered) != len(verts_in):
        return None

    # Coordinate edits can refresh BMesh edge indices, so edge.index is
    # not a stable identity for detecting selection changes.
    signature = tuple(sorted(
        tuple(sorted((e.verts[0].index, e.verts[1].index)))
        for e in edges
    ))

    return obj, bm, ordered, closed, signature


def _section_basis(bm, ordered, closed):
    pts = [bm.verts[i].co.copy() for i in ordered]
    center = sum(pts, Vector()) / len(pts)

    normal = Vector((0.0, 0.0, 0.0))

    if closed and len(pts) >= 3:
        for i, p in enumerate(pts):
            q = pts[(i + 1) % len(pts)]
            normal += (p - center).cross(q - center)
    elif len(pts) >= 3:
        for i in range(1, len(pts) - 1):
            normal += (pts[i] - pts[i - 1]).cross(pts[i + 1] - pts[i])

    if normal.length < 1e-8:
        direction = (pts[-1] - pts[0]) if len(pts) > 1 else Vector((1, 0, 0))
        if direction.length < 1e-8:
            direction = Vector((1, 0, 0))
        direction.normalize()
        ref = Vector((0, 0, 1))
        if abs(direction.dot(ref)) > 0.95:
            ref = Vector((0, 1, 0))
        normal = direction.cross(ref)

    normal.normalize()

    if closed:
        u = pts[0] - center
        if u.length < 1e-8:
            u = Vector((1, 0, 0))
    else:
        u = pts[-1] - pts[0]
        if u.length < 1e-8:
            u = Vector((1, 0, 0))

    u = u - normal * u.dot(normal)
    if u.length < 1e-8:
        ref = Vector((1, 0, 0))
        if abs(ref.dot(normal)) > 0.95:
            ref = Vector((0, 1, 0))
        u = ref - normal * ref.dot(normal)

    u.normalize()
    v = normal.cross(u).normalized()

    coords = []
    depths = []

    for p in pts:
        d = p - center
        coords.append(Vector((d.dot(u), d.dot(v))))
        depths.append(d.dot(normal))

    return center, u, v, normal, coords, depths


def _smooth_falloff(t):
    t = max(0.0, min(1.0, t))
    s = 3.0 * t * t - 2.0 * t * t * t
    return 1.0 - s


def _resample_spacing_2d(points, spacing=3.0, closed=False):
    pts = [Vector(p) for p in points]
    if len(pts) < 2:
        return pts

    if closed:
        if (pts[-1] - pts[0]).length > 0.5:
            pts = pts + [pts[0].copy()]

    out = [pts[0].copy()]
    carry = 0.0
    cur = pts[0].copy()

    for target in pts[1:]:
        seg = target - cur
        seg_len = seg.length
        if seg_len < 1e-8:
            continue

        direction = seg / seg_len
        dist = seg_len

        while carry + dist >= spacing:
            need = spacing - carry
            cur = cur + direction * need
            out.append(cur.copy())
            dist -= need
            carry = 0.0

            if dist <= 1e-8:
                break

            remain = target - cur
            if remain.length < 1e-8:
                break
            direction = remain.normalized()

        if dist > 1e-8:
            carry += dist
            cur = target.copy()

    if closed:
        if len(out) > 1 and (out[-1] - out[0]).length < 1.0:
            out.pop()
    else:
        if (out[-1] - pts[-1]).length > 0.5:
            out.append(pts[-1].copy())

    return out


def _moving_average_2d(points, radius=2, passes=2, closed=False):
    pts = [Vector(p) for p in points]
    if len(pts) < 3 or radius <= 0 or passes <= 0:
        return pts

    for _ in range(passes):
        src = [p.copy() for p in pts]
        dst = []

        for i in range(len(src)):
            if not closed and i in {0, len(src) - 1}:
                dst.append(src[i].copy())
                continue

            acc = Vector((0.0, 0.0))
            total = 0.0

            for off in range(-radius, radius + 1):
                j = i + off

                if closed:
                    j %= len(src)
                elif j < 0 or j >= len(src):
                    continue

                w = radius + 1 - abs(off)
                acc += src[j] * w
                total += w

            dst.append(acc / max(total, 1e-8))

        pts = dst

    return pts


def _chaikin_2d(points, iterations=1, closed=False):
    pts = [Vector(p) for p in points]
    if len(pts) < 3:
        return pts

    for _ in range(iterations):
        new = []

        if closed:
            for i in range(len(pts)):
                a = pts[i]
                b = pts[(i + 1) % len(pts)]
                new.extend((
                    a * 0.75 + b * 0.25,
                    a * 0.25 + b * 0.75,
                ))
        else:
            new = [pts[0].copy()]
            for a, b in zip(pts[:-1], pts[1:]):
                new.extend((
                    a * 0.75 + b * 0.25,
                    a * 0.25 + b * 0.75,
                ))
            new.append(pts[-1].copy())

        pts = new

    return pts


def _stabilize_2d(points, level=2, closed=False):
    if len(points) < 2:
        return [Vector(p) for p in points]

    # 0-10: stronger levels deliberately add visible lag and smoothing.
    level = max(0, min(10, int(level)))

    if level == 0:
        return [Vector(p) for p in points]
    spacing = max(2.0, 4.8 - level * 0.18)

    pts = _resample_spacing_2d(points, spacing=spacing, closed=closed)

    if level > 0:
        radius = 1 + level // 2
        passes = 1 + level
        pts = _moving_average_2d(
            pts,
            radius=radius,
            passes=passes,
            closed=closed,
        )

    return _chaikin_2d(pts, iterations=1, closed=closed)


def _align_closed_pen_stroke(points, target_points, anchor_mode='STROKE'):
    """Choose cyclic phase and winding without rotating the drawn shape.

    Fit in translation/scale-normalized coordinates. Preserve the target's
    perimeter spacing, so the caller can use its existing path sampler.
    The returned polyline has an explicit matched first point, no duplicate
    closing endpoint. This same result feeds both preview and commit.
    """
    from bisect import bisect_right
    from math import hypot, sqrt

    def clean(values):
        out = []
        for p in values:
            q = (float(p[0]), float(p[1]))
            if not out or hypot(q[0]-out[-1][0], q[1]-out[-1][1]) > 1e-8:
                out.append(q)
        if len(out) > 1 and hypot(out[-1][0]-out[0][0], out[-1][1]-out[0][1]) <= 1e-8:
            out.pop()
        return out

    def path(values):
        lengths = [hypot(b[0]-a[0], b[1]-a[1])
                   for a,b in zip(values, values[1:]+values[:1])]
        total = sum(lengths)
        if total <= 1e-8:
            return None
        cumulative = [0.0]
        for length in lengths:
            cumulative.append(cumulative[-1] + length / total)
        cumulative[-1] = 1.0
        # Perimeter centroid / RMS radius do not depend on point density.
        cx = sum((a[0]+b[0])*0.5*l for a,b,l in
                 zip(values, values[1:]+values[:1], lengths)) / total
        cy = sum((a[1]+b[1])*0.5*l for a,b,l in
                 zip(values, values[1:]+values[:1], lengths)) / total
        variance = 0.0
        for a,b,l in zip(values, values[1:]+values[:1], lengths):
            ax,ay,bx,by = a[0]-cx,a[1]-cy,b[0]-cx,b[1]-cy
            variance += l*(ax*ax+ax*bx+bx*bx+ay*ay+ay*by+by*by)/3.0
        scale = max(1e-8, sqrt(variance/total))
        return cumulative, (cx,cy), scale

    def sample(values, cumulative, t):
        t %= 1.0
        i = min(len(values)-1, bisect_right(cumulative, t)-1)
        fraction = (t-cumulative[i]) / max(1e-12, cumulative[i+1]-cumulative[i])
        a,b = values[i],values[(i+1)%len(values)]
        return (a[0]+(b[0]-a[0])*fraction, a[1]+(b[1]-a[1])*fraction)

    source, target = clean(points), clean(target_points)
    if len(source) < 3 or len(target) < 3:
        return [Vector(p) for p in points]
    source_info, target_info = path(source), path(target)
    if source_info is None or target_info is None:
        return [Vector(p) for p in points]
    target_t, tc, ts = target_info
    target_normal = [((x-tc[0])/ts,(y-tc[1])/ts) for x,y in target]
    best = None
    # Bounded coarse search + continuous refinement avoids dependence on
    # where the user started the stroke, even midway along an edge.
    steps = 128
    for values in (source, list(reversed(source))):
        cumulative, center, scale = path(values)
        def score(phase):
            error = 0.0
            for t,(tx,ty) in zip(target_t, target_normal):
                x,y = sample(values,cumulative,t+phase)
                dx,dy = (x-center[0])/scale-tx,(y-center[1])/scale-ty
                error += dx*dx+dy*dy
            return error
        # Include exact vertices (bounded for long mouse strokes).
        stride = max(1, len(values)//128)
        phases = [i/steps for i in range(steps)] + cumulative[:-1:stride]
        phase = min(phases,key=score)
        error = score(phase)
        width = 1.0/steps
        for _ in range(6):
            candidates = [phase+width*j/4 for j in range(-4,5)]
            phase = min(candidates,key=score)
            error = score(phase)
            width /= 4
        if best is None or error < best[0]:
            best = (error,values,cumulative,phase%1.0)
    _,values,cumulative,phase = best
    start = sample(values,cumulative,phase)
    ordered = [start]
    vertices = sorted((((t-phase)%1.0,p) for t,p in zip(cumulative,values)),key=lambda pair:pair[0])
    for distance,p in vertices:
        if distance > 1e-9 and hypot(p[0]-ordered[-1][0],p[1]-ordered[-1][1]) > 1e-8:
            ordered.append(p)
    result = [Vector(p) for p in ordered]
    if anchor_mode == 'POINT':
        offset = Vector(target_points[0]) - result[0]
        result = [p + offset for p in result]
    return result


def _path_t_values(points, closed=False):
    pts = [Vector(p) for p in points]
    if len(pts) <= 1:
        return [0.0]

    if closed:
        segs = [
            (pts[(i + 1) % len(pts)] - pts[i]).length
            for i in range(len(pts))
        ]
        total = sum(segs)
        if total < 1e-8:
            return [i / len(pts) for i in range(len(pts))]

        out = [0.0]
        run = 0.0
        for i in range(len(pts) - 1):
            run += segs[i]
            out.append(run / total)
        return out

    segs = [(b - a).length for a, b in zip(pts[:-1], pts[1:])]
    total = sum(segs)
    if total < 1e-8:
        return [i / (len(pts) - 1) for i in range(len(pts))]

    out = [0.0]
    run = 0.0
    for l in segs:
        run += l
        out.append(run / total)
    return out


def _sample_polyline_2d(points, t_values, closed=False):
    pts = [Vector(p) for p in points]
    if not pts:
        return []
    if len(pts) == 1:
        return [pts[0].copy() for _ in t_values]

    work = pts + [pts[0].copy()] if closed else pts
    seg_lens = [(b - a).length for a, b in zip(work[:-1], work[1:])]
    total = sum(seg_lens)

    if total < 1e-8:
        return [work[0].copy() for _ in t_values]

    cumulative = [0.0]
    run = 0.0
    for l in seg_lens:
        run += l
        cumulative.append(run / total)

    result = []
    for t in t_values:
        if closed:
            t = t % 1.0
        else:
            t = max(0.0, min(1.0, t))

        if not closed and t >= 1.0:
            result.append(work[-1].copy())
            continue

        for i in range(len(cumulative) - 1):
            a_t = cumulative[i]
            b_t = cumulative[i + 1]
            if a_t <= t <= b_t or (i == len(cumulative) - 2 and t >= a_t):
                f = (t - a_t) / max(b_t - a_t, 1e-8)
                result.append(work[i].lerp(work[i + 1], f))
                break

    return result





def _shape_resample_2d(points, count, closed=False):
    pts = [Vector(p).copy() for p in points]

    if count <= 0 or not pts:
        return []

    if count == 1:
        return [pts[0].copy()]

    work = [p.copy() for p in pts]

    if closed:
        work.append(work[0].copy())

    seg_lengths = []
    cumulative = [0.0]

    for a, b in zip(work[:-1], work[1:]):
        length = (b - a).length
        seg_lengths.append(length)
        cumulative.append(cumulative[-1] + length)

    total = cumulative[-1]

    if total <= 1e-8:
        return [pts[0].copy() for _ in range(count)]

    if closed:
        targets = [
            total * i / count
            for i in range(count)
        ]
    else:
        targets = [
            total * i / (count - 1)
            for i in range(count)
        ]

    out = []
    seg_i = 0

    for target in targets:
        while (
            seg_i < len(seg_lengths) - 1
            and cumulative[seg_i + 1] < target
        ):
            seg_i += 1

        length = seg_lengths[seg_i]

        if length <= 1e-8:
            out.append(work[seg_i].copy())
            continue

        local_t = (
            target - cumulative[seg_i]
        ) / length

        out.append(
            work[seg_i].lerp(
                work[seg_i + 1],
                local_t
            )
        )

    return out


def _shape_targets_2d(
    all_points,
    order,
    tool,
    closed=False
):
    order = list(order or [])

    if len(order) < 2:
        return None

    source = [
        Vector(all_points[i]).copy()
        for i in order
    ]

    result = [
        p.copy()
        for p in source
    ]

    if tool == 'RELAX':
        if len(source) < 3:
            return None

        for k in range(len(source)):
            if not closed and k in {
                0,
                len(source) - 1
            }:
                continue

            prev_p = source[
                (k - 1) % len(source)
            ]
            next_p = source[
                (k + 1) % len(source)
            ]

            # One clearly visible Laplacian relax pass.
            # Open-range endpoints stay fixed; interior points move to the
            # midpoint of their two chain neighbours.
            result[k] = (
                prev_p + next_p
            ) * 0.5

    elif tool == 'EQUAL':
        result = _shape_resample_2d(
            source,
            len(source),
            closed=closed
        )

    elif tool == 'LINE':
        if closed:
            return None

        a = source[0]
        b = source[-1]
        denom = max(
            len(source) - 1,
            1
        )

        result = [
            a.lerp(
                b,
                k / denom
            )
            for k in range(len(source))
        ]

    elif tool == 'ELLIPSE':
        if len(source) < 3:
            return None

        xs = [p.x for p in source]
        ys = [p.y for p in source]

        center = Vector((
            (min(xs) + max(xs)) * 0.5,
            (min(ys) + max(ys)) * 0.5
        ))

        rx = max(
            (max(xs) - min(xs)) * 0.5,
            1e-8
        )
        ry = max(
            (max(ys) - min(ys)) * 0.5,
            1e-8
        )

        result = []

        for k, p in enumerate(source):
            nx = (p.x - center.x) / rx
            ny = (p.y - center.y) / ry

            if abs(nx) + abs(ny) <= 1e-8:
                angle = tau * k / len(source)
            else:
                angle = atan2(ny, nx)

            result.append(Vector((
                center.x + cos(angle) * rx,
                center.y + sin(angle) * ry
            )))

    elif tool == 'SYMMETRY':
        if len(source) < 2:
            return None

        xs = [p.x for p in source]
        center_x = (
            min(xs) + max(xs)
        ) * 0.5

        result = [
            p.copy()
            for p in source
        ]

        remaining = set(
            range(len(source))
        )

        while remaining:
            i = min(remaining)
            remaining.remove(i)

            if not remaining:
                result[i].x = center_x
                continue

            reflected = Vector((
                center_x * 2.0 - source[i].x,
                source[i].y
            ))

            j = min(
                remaining,
                key=lambda idx: (
                    source[idx] - reflected
                ).length
            )
            remaining.remove(j)

            if source[i].x <= source[j].x:
                left_i, right_i = i, j
            else:
                left_i, right_i = j, i

            half_width = (
                abs(source[i].x - center_x)
                + abs(source[j].x - center_x)
            ) * 0.5

            y = (
                source[i].y
                + source[j].y
            ) * 0.5

            result[left_i] = Vector((
                center_x - half_width,
                y
            ))
            result[right_i] = Vector((
                center_x + half_width,
                y
            ))

    else:
        return None

    return {
        original_index: result[k]
        for k, original_index
        in enumerate(order)
    }





def _hp_smooth_debug(message):
    line = f"[HP Smooth] {message}"
    print(line)

    try:
        text = bpy.data.texts.get("HP_Smooth_Debug")

        if text is None:
            text = bpy.data.texts.new("HP_Smooth_Debug")

        text.write(line + "\n")
    except Exception:
        pass


def _hp_relax_once_2d(points, closed=False):
    src = [
        Vector(p).copy()
        for p in points
    ]

    if len(src) < 3:
        return src

    dst = [
        p.copy()
        for p in src
    ]

    for i in range(len(src)):
        if (
            not closed
            and i in {0, len(src) - 1}
        ):
            continue

        prev_p = src[
            (i - 1) % len(src)
        ]
        next_p = src[
            (i + 1) % len(src)
        ]

        dst[i] = (
            prev_p + next_p
        ) * 0.5

    return dst


def _hp_relax_amount_2d(
    points,
    amount,
    closed=False
):
    """
    amount:
      0.0 = original
      1.0 = one full relax pass
      2.0 = two passes
      ...
    Fractional values interpolate between passes.
    """
    base = [
        Vector(p).copy()
        for p in points
    ]

    if len(base) < 3:
        return base

    amount = max(
        0.0,
        min(20.0, float(amount))
    )

    whole = int(amount)
    frac = amount - whole

    current = [
        p.copy()
        for p in base
    ]

    for _ in range(whole):
        current = _hp_relax_once_2d(
            current,
            closed=closed
        )

    if frac > 1e-8:
        nxt = _hp_relax_once_2d(
            current,
            closed=closed
        )

        current = [
            a.lerp(b, frac)
            for a, b in zip(
                current,
                nxt
            )
        ]

    # Explicit endpoint preservation for open selections.
    if not closed and len(current) >= 2:
        current[0] = base[0].copy()
        current[-1] = base[-1].copy()

    return current


def _hp_brush_repel_2d(
    points,
    mouse,
    motion,
    radius=70.0,
    strength=0.55,
    closed=False,
):
    """Continuous swept brush with zero influence at its boundary.

    Integrate along the mouse path rather than stamping only at its endpoint.
    Distance-based steps make sparse and dense mouse events behave alike.
    The legacy name is retained for existing vertex/pen brush callers.
    """
    pts = [Vector(p).copy() for p in points]
    motion = Vector(motion)
    if len(pts) < 2 or radius <= 1.0 or motion.length < 0.01:
        return pts
    radius = max(4.0, float(radius))
    strength = max(0.0, min(1.0, float(strength)))
    steps = max(1, min(512, ceil(motion.length / min(2.0, radius * 0.1))))
    delta = motion / steps
    start = Vector(mouse) - motion
    gain = strength * (0.35 + 1.65 * strength)
    for step in range(steps):
        center = start + delta * (step + 0.5)
        for i, point in enumerate(pts):
            t = max(0.0, 1.0 - (point - center).length / radius)
            # Smoothstep removes the old 25% discontinuity at radius.
            influence = t * t * (3.0 - 2.0 * t)
            pts[i] = point + delta * (gain * influence)
    return pts


class HP_OT_section_shape_choice(bpy.types.Operator):
    bl_idname = "hp.section_shape_choice"
    bl_label = "整形"

    tool: EnumProperty(
        items=(
            ('EQUAL', "等間隔化", "選択点を2D上で等間隔に配置します"),
            ('LINE', "直線化", "選択範囲を両端間の直線上へ配置します"),
            ('ELLIPSE', "楕円フィット", "現在の幅・高さを使って楕円へフィットします"),
            ('SYMMETRY', "左右対称化", "現在の2D表示上で左右対称に整えます"),
        ),
        default='EQUAL',
    )

    def execute(self, context):
        editor = _ACTIVE_SECTION_EDITOR
        if editor is None:
            return {'CANCELLED'}

        if getattr(editor, "_shape_pie_target", 'A') == 'B':
            editor._apply_secondary_shape_tool(context, self.tool)
        else:
            editor._apply_primary_shape_tool(context, self.tool)

        if context.area:
            context.area.tag_redraw()

        return {'FINISHED'}


class HP_MT_section_shape_pie(bpy.types.Menu):
    bl_idname = "HP_MT_section_shape_pie"
    bl_label = "整形"

    def draw(self, context):
        pie = self.layout.menu_pie()

        op = pie.operator(
            HP_OT_section_shape_choice.bl_idname,
            text="等間隔化",
            icon='ALIGN_JUSTIFY'
        )
        op.tool = 'EQUAL'

        op = pie.operator(
            HP_OT_section_shape_choice.bl_idname,
            text="直線化",
            icon='IPO_LINEAR'
        )
        op.tool = 'LINE'

        op = pie.operator(
            HP_OT_section_shape_choice.bl_idname,
            text="楕円フィット",
            icon='MESH_CIRCLE'
        )
        op.tool = 'ELLIPSE'

        op = pie.operator(
            HP_OT_section_shape_choice.bl_idname,
            text="左右対称化",
            icon='MOD_MIRROR'
        )
        op.tool = 'SYMMETRY'



class HP_OT_section_topology_choice(bpy.types.Operator):
    bl_idname = "hp.section_topology_choice"
    bl_label = "細分化"

    tool: EnumProperty(
        items=(
            ('SUBDIVIDE', "細分化", "選択範囲内のエッジを1段細分化します"),
            ('DISSOLVE', "点を溶解", "選択した中間点を溶解して点数を減らします"),
            ('MERGE', "2点を中央マージ", "隣接する2点を中央で1点にまとめます"),
        ),
        default='SUBDIVIDE',
    )

    def execute(self, context):
        editor = _ACTIVE_SECTION_EDITOR

        if editor is None:
            return {'CANCELLED'}

        editor._apply_topology_tool(
            context,
            self.tool
        )

        if context.area:
            context.area.tag_redraw()

        return {'FINISHED'}


class HP_MT_section_topology_pie(bpy.types.Menu):
    bl_idname = "HP_MT_section_topology_pie"
    bl_label = "細分化"

    def draw(self, context):
        pie = self.layout.menu_pie()

        op = pie.operator(
            HP_OT_section_topology_choice.bl_idname,
            text="細分化（1段）"
        )
        op.tool = 'SUBDIVIDE'

        op = pie.operator(
            HP_OT_section_topology_choice.bl_idname,
            text="点を溶解"
        )
        op.tool = 'DISSOLVE'

        op = pie.operator(
            HP_OT_section_topology_choice.bl_idname,
            text="2点を中央マージ"
        )
        op.tool = 'MERGE'


class HP_OT_section_pen_anchor_choice(bpy.types.Operator):
    bl_idname = "hp.section_pen_anchor_choice"
    bl_label = "ペン始点モード"

    mode: EnumProperty(
        items=(
            ('STROKE', "Stroke Start", "First point follows where the pen stroke starts"),
            ('POINT', "Point Start", "Keep the first target point in its original position"),
        ),
        default='STROKE',
    )

    def execute(self, context):
        global _SECTION_PEN_ANCHOR_DEFAULT

        editor = _ACTIVE_SECTION_EDITOR
        if editor is None:
            return {'CANCELLED'}

        _SECTION_PEN_ANCHOR_DEFAULT = self.mode
        editor._pen_anchor_mode = self.mode
        editor._activate_pen_target(editor._pie_target)

        if context.area:
            context.area.tag_redraw()

        return {'FINISHED'}


class HP_MT_section_pen_anchor_pie(bpy.types.Menu):
    bl_idname = "HP_MT_section_pen_anchor_pie"
    bl_label = "ペン始点モード"

    def draw(self, context):
        pie = self.layout.menu_pie()

        op = pie.operator(
            HP_OT_section_pen_anchor_choice.bl_idname,
            text="描き始め位置を優先",
            icon='GREASEPENCIL'
        )
        op.mode = 'STROKE'

        op = pie.operator(
            HP_OT_section_pen_anchor_choice.bl_idname,
            text="元の点位置を固定",
            icon='SNAP_ON'
        )
        op.mode = 'POINT'


def _hp_pen_samples(stroke, target, closed, correction, shape_only, anchor):
    """Independent, baseline-only outline calculation shared by preview/commit."""
    original = [p.copy() for p in stroke]
    if correction > 0 and len(original) >= 3:
        # Uniform perimeter samples give correction a consistent meaning,
        # independent of drawing speed. Restore size after smoothing.
        count = 96
        times = [i/(count if closed else count-1) for i in range(count)]
        original = _hp_sample(stroke, times, closed=closed)
        result = [p.copy() for p in original]
        whole = int(correction)
        for _ in range(whole):
            result = _hp_average(result, radius=2, passes=1, closed=closed)
        fraction = correction-whole
        if fraction > 1e-8:
            nxt = _hp_average(result, radius=2, passes=1, closed=closed)
            result = [a.lerp(b,fraction) for a,b in zip(result,nxt)]
        if closed:
            result = _hp_match_size(result, original)
    else:
        result = original
    if closed:
        result = _align_closed_pen_stroke(result,target,anchor)
    else:
        forward=(result[0]-target[0]).length+(result[-1]-target[-1]).length
        reverse=(result[-1]-target[0]).length+(result[0]-target[-1]).length
        if reverse < forward: result.reverse()
        if anchor == 'POINT':
            offset=target[0]-result[0]
            result=[p+offset for p in result]
    samples=_hp_sample(result,_path_t_values(target,closed),closed=closed)
    if shape_only and closed:
        samples=_hp_match_size(samples,target)
    return samples


def _hp_match_size(points, reference):
    from math import sqrt
    center=sum(points,Vector((0,0)))/len(points)
    ref_center=sum(reference,Vector((0,0)))/len(reference)
    size=sqrt(sum((p-center).length**2 for p in points)/len(points))
    ref_size=sqrt(sum((p-ref_center).length**2 for p in reference)/len(reference))
    if size < 1e-9: return [p.copy() for p in points]
    return [ref_center+(p-center)*(ref_size/size) for p in points]


def _hp_adjacent_rings(ring, faces, max_depth=6, stop_triangles=True,
                       closed=True):
    """Follow unambiguous quad strips beside a closed or open section."""
    if len(ring) < 2:
        return [[], []]
    edge_faces={}
    for fi,face in enumerate(faces):
        for a,b in zip(face,face[1:]+face[:1]):
            edge_faces.setdefault(tuple(sorted((a,b))),[]).append(fi)
    def strip(current,first_face):
        current_set=set(current)
        mapping={}
        used=set()
        pairs = (zip(current, current[1:]+current[:1]) if closed
                 else zip(current, current[1:]))
        for i,(a,b) in enumerate(pairs):
            candidates=[]
            linked=edge_faces.get(tuple(sorted((a,b))),[])
            if len(linked)>2: return None
            for fi in linked:
                face=faces[fi]
                shared = set(face) & current_set
                if shared != {a,b}:
                    continue
                if len(face) == 4:
                    ai,bi=face.index(a),face.index(b)
                    na=next(v for v in (face[(ai-1)%4],face[(ai+1)%4]) if v!=b)
                    nb=next(v for v in (face[(bi-1)%4],face[(bi+1)%4]) if v!=a)
                elif len(face) == 3 and not stop_triangles:
                    # A triangle cannot always define a one-to-one loop.
                    # When it can, use its third vertex as a bridge; duplicate
                    # results are rejected below instead of causing a jump.
                    third=next(v for v in face if v not in {a,b})
                    na=third; nb=third
                else:
                    continue
                if i==0 and fi!=first_face: continue
                if a in mapping and mapping[a]!=na: continue
                if b in mapping and mapping[b]!=nb: continue
                candidates.append((fi,na,nb))
            if len(candidates)!=1: return None
            fi,na,nb=candidates[0]
            mapping[a],mapping[b]=na,nb
            used.add(fi)
        out=[mapping[v] for v in current]
        if len(set(out))!=len(current) or set(out)&current_set: return None
        # Verify the outgoing boundary is a manifold loop.
        out_pairs = (zip(out, out[1:]+out[:1]) if closed
                     else zip(out, out[1:]))
        if any(len(edge_faces.get(tuple(sorted((a,b))),[]))>2
               for a,b in out_pairs): return None
        return out,used
    first=edge_faces.get(tuple(sorted(ring[:2])),[])
    if len(first)>2: return [[],[]]
    sides=[]
    visited=set(ring)
    for seed in first[:2]:
        current=list(ring); last_faces=set(); side=[]
        for depth in range(max_depth):
            candidates=[seed] if depth==0 else [fi for fi in edge_faces.get(tuple(sorted(current[:2])),[]) if fi not in last_faces]
            if len(candidates)!=1: break
            result=strip(current,candidates[0])
            if result is None: break
            nxt,last_faces=result
            if set(nxt)&visited: break
            side.append(nxt); visited.update(nxt); current=nxt
        sides.append(side)
    result=(sides+[[],[]])[:2]
    _hp_log(
        f"adjacent ring={len(ring)} closed={closed} stop_triangles={stop_triangles} "
        f"A={len(result[0])} B={len(result[1])}"
    )
    return result


def _hp_adjacent_spans(ring, faces, max_depth=6, closed=True):
    """Map source points through available quads, without requiring a full ring.

    Each depth contains (source position, neighbor vertex) pairs. A/B use the
    directed edge winding so a gap on the first edge cannot swap both sides.
    """
    if len(ring) < 2:
        return [[], []]
    edge_faces = {}
    for fi, face in enumerate(faces):
        for a, b in zip(face, face[1:] + face[:1]):
            edge_faces.setdefault(tuple(sorted((a, b))), []).append(fi)
    sides = []
    visited = set(ring)
    for direction in (1, -1):
        current = dict(enumerate(ring))
        side = []
        for _ in range(max_depth):
            choices = {}
            conflicts = set()
            pairs = ([(i, (i + 1) % len(ring)) for i in range(len(ring))]
                     if closed else [(i, i + 1) for i in range(len(ring) - 1)])
            for i, j in pairs:
                if i not in current or j not in current:
                    continue
                a, b = current[i], current[j]
                linked = edge_faces.get(tuple(sorted((a, b))), [])
                if len(linked) > 2:
                    continue
                for fi in linked:
                    face = faces[fi]
                    if len(face) != 4:
                        continue
                    winding = 1 if any(x == a and y == b for x, y in
                                       zip(face, face[1:] + face[:1])) else -1
                    if winding != direction:
                        continue
                    # Each quad must touch the current span on exactly this edge.
                    if len(set(face) & set(current.values())) != 2:
                        continue
                    ai, bi = face.index(a), face.index(b)
                    na = next(v for v in (face[(ai - 1) % 4], face[(ai + 1) % 4]) if v != b)
                    nb = next(v for v in (face[(bi - 1) % 4], face[(bi + 1) % 4]) if v != a)
                    for source, target in ((i, na), (j, nb)):
                        if source in choices and choices[source] != target:
                            conflicts.add(source)
                        choices[source] = target
            for index in conflicts:
                choices.pop(index, None)
            inverse = {}
            for source, target in choices.items():
                inverse.setdefault(target, []).append(source)
            current = {source: target for source, target in choices.items()
                       if target not in visited and len(inverse[target]) == 1}
            if len(current) < 2:
                break
            side.append(sorted(current.items()))
            visited.update(current.values())
        sides.append(side)
    return sides


def _hp_ring_frame(points):
    from math import sqrt
    center=sum(points,Vector((0,0,0)))/len(points)
    normal=Vector((0,0,0))
    for a,b in zip(points,points[1:]+points[:1]): normal+=(a-center).cross(b-center)
    if normal.length<1e-9: return None
    normal.normalize()
    axis=points[0]-center
    axis-=normal*axis.dot(normal)
    if axis.length<1e-9: return None
    axis.normalize()
    up=normal.cross(axis).normalized()
    radius=sqrt(sum(((p-center).dot(axis))**2+((p-center).dot(up))**2 for p in points)/len(points))
    if radius<1e-9: return None
    return center,axis,up,normal,radius


def _hp_transfer_ring(source_base, source_goal, neighbor_base, amount):
    """Trace shape in each loop's frame, retaining taper and per-point depth."""
    source=_hp_ring_frame(source_base); dest=_hp_ring_frame(neighbor_base)
    if source is None or dest is None: return [p.copy() for p in neighbor_base]
    sc,sx,sy,sn,sr=source; dc,dx,dy,dn,dr=dest
    out=[]
    for p,goal in zip(neighbor_base,source_goal):
        delta=goal-sc
        traced=dc+dx*(delta.dot(sx)*dr/sr)+dy*(delta.dot(sy)*dr/sr)+dn*(p-dc).dot(dn)
        out.append(p.lerp(traced,amount))
    return out


_hp_sample = _sample_polyline_2d
_hp_average = _moving_average_2d
_hp_stabilize = _stabilize_2d

class HP_OT_section_mini_editor(SectionPreviewMixin, bpy.types.Operator):
    _hp_is_curve = False

    def _hp_follow_sides(self, settings):
        if not self._bmesh_ready() or len(self._ordered) < 2:
            return [[], []]
        self._bm.verts.ensure_lookup_table()
        self._bm.verts.index_update()
        faces = [tuple(v.index for v in f.verts) for f in self._bm.faces]
        return _hp_adjacent_spans(self._ordered, faces, max_depth=6,
                                  closed=self._closed)

    def _hp_enable_follow(self, settings, sides):
        """Make the ON switch useful even if both loop counts were still zero."""
        for key, side in zip(('side_a', 'side_b'), sides):
            if side and int(settings.get(key, 0)) <= 0:
                settings[key] = min(2, len(side))
            elif not side:
                settings[key] = 0
        enabled = any(settings.get(key, 0) for key in ('side_a', 'side_b'))
        settings['direct_follow'] = enabled
        if enabled and all(float(settings.get(k, 1.0)) <= 0.0
                           for k in ('follow_strength_a', 'follow_strength_b')):
            settings['follow_strength_a'] = 1.0
            settings['follow_strength_b'] = 1.0
        return enabled

    def _hp_quick_follow_button(self, x, y):
        """Buttons below View A's header: toggle, side A, side B, log."""
        if self._view_a_collapsed:
            return None
        top = self._panel_y + self._panel_h - 39
        if not (top - 28 <= y <= top):
            return None
        left = self._panel_x + 8
        cell = (self._panel_w - 16) / 4
        if left <= x < left + cell * 4:
            return ('toggle', 'A', 'B', 'log')[int((x - left) / cell)]
        return None

    def _hp_quick_follow_click(self, context, button):
        if button == 'log':
            block = bpy.data.texts.get('HP_Follow_Diagnostics')
            if block is None:
                self.report({'WARNING'}, '追従ログはまだありません')
            else:
                context.window_manager.clipboard = block.as_string()
                self.report({'INFO'}, '追従ログをクリップボードにコピーしました')
            return
        settings = _hp_settings_complete(getattr(self, '_hp_settings', None))
        sides = self._hp_follow_sides(settings)
        _hp_follow_log(
            f"click={button} source={len(self._ordered)} closed={self._closed} "
            f"faces={len(self._bm.faces) if self._bmesh_ready() else 'invalid'} "
            f"available=A:{[len(s) for s in sides[0]]} B:{[len(s) for s in sides[1]]} "
            f"previous={settings}"
        )
        if button == 'toggle':
            if settings.get('direct_follow', False):
                settings['direct_follow'] = False
            elif not self._hp_enable_follow(settings, sides):
                self.report({'WARNING'}, '隣接する四角面のループを検出できません')
        else:
            key = 'follow_strength_a' if button == 'A' else 'follow_strength_b'
            if not settings.get('direct_follow', False):
                self._hp_enable_follow(settings, sides)
            if not sides[0 if button == 'A' else 1]:
                self.report({'WARNING'}, f'隣接{button}のループを検出できません')
            else:
                steps = (0.25, 0.5, 0.75, 1.0)
                current = float(settings.get(key, 1.0))
                settings[key] = next((v for v in steps if v > current + 0.01), steps[0])
        self._hp_settings = settings
        self._hp_refresh_follow_visual()
        _hp_follow_log(f"active={settings.get('direct_follow', False)} "
                       f"count=A:{settings.get('side_a', 0)} B:{settings.get('side_b', 0)} "
                       f"strength=A:{settings.get('follow_strength_a', 1.0):.2f} "
                       f"B:{settings.get('follow_strength_b', 1.0):.2f}")
        context.area.tag_redraw()

    def _hp_refresh_follow_visual(self):
        settings = getattr(self, '_hp_settings', {})
        self._hp_visual_spans = [[], []]
        self._hp_visual_edges = set()
        if not settings.get('direct_follow', False):
            return
        sides = self._hp_follow_sides(settings)
        self._hp_visual_spans = [side[:max(0, int(settings.get(key, 0)))]
                                 for side, key in zip(sides, ('side_a', 'side_b'))]
        self._hp_visual_edges = {
            tuple(sorted((edge.verts[0].index, edge.verts[1].index)))
            for edge in self._bm.edges
        }

    def _hp_capture_direct_follow(self):
        """Freeze neighboring quad strips for one direct drag or transform."""
        settings = getattr(self, '_hp_settings', {})
        if self._hp_is_curve or not settings.get('direct_follow', False):
            return None
        if not self._bmesh_ready():
            _hp_follow_log('capture=SKIPPED bmesh unavailable')
            return None
        counts = [max(0, int(settings.get(k, 0))) for k in ('side_a', 'side_b')]
        strengths = [max(0.0, min(1.0, float(settings.get(k, 1.0))))
                     for k in ('follow_strength_a', 'follow_strength_b')]
        if not any(c and s for c, s in zip(counts, strengths)):
            _hp_follow_log(f'capture=SKIPPED counts={counts} strengths={strengths}')
            return None
        self._bm.verts.ensure_lookup_table()
        self._bm.verts.index_update()
        sides = self._hp_follow_sides(settings)
        sides = [side[:count] if strength else []
                 for side, count, strength in zip(sides, counts, strengths)]
        if not any(sides):
            _hp_follow_log(f"capture=EMPTY counts={counts} strengths={strengths} "
                           f"source={len(self._ordered)}")
            return None
        affected = set(self._ordered)
        for side in sides:
            for span in side:
                affected.update(vi for _, vi in span)
        _hp_follow_log(f"capture=OK counts={counts} "
                       f"spans=A:{[len(s) for s in sides[0]]} "
                       f"B:{[len(s) for s in sides[1]]} affected={len(affected)}")
        return dict(
            ring=list(self._ordered), sides=sides,
            base={i: self._bm.verts[i].co.copy() for i in affected},
            counts=(len(self._bm.verts), len(self._bm.edges), len(self._bm.faces)),
            data=self._obj.data, strengths=strengths,
            falloff=max(0.0, float(settings.get('falloff', 1.0))),
            first_move_logged=False,
        )

    def _hp_apply_direct_follow(self):
        state = getattr(self, '_direct_follow_state', None)
        if state is None or self._obj.data != state['data']:
            return
        if (len(self._bm.verts), len(self._bm.edges), len(self._bm.faces)) != state['counts']:
            self._direct_follow_state = None
            return
        mw = self._obj.matrix_world
        inv = mw.inverted()
        deltas = [mw @ self._bm.verts[vi].co - mw @ state['base'][vi]
                  for vi in state['ring']]
        max_delta = max((delta.length for delta in deltas), default=0.0)
        if max_delta > 1e-7 and not state['first_move_logged']:
            _hp_follow_log(f"first_move source_delta={max_delta:.6f} "
                           f"spans=A:{[len(s) for s in state['sides'][0]]} "
                           f"B:{[len(s) for s in state['sides'][1]]}")
            state['first_move_logged'] = True
        for side, strength in zip(state['sides'], state['strengths']):
            count = len(side)
            for depth, span in enumerate(side, 1):
                weight = strength * ((count - depth + 1.0) / count) ** state['falloff']
                for source, vi in span:
                    old = mw @ state['base'][vi]
                    self._bm.verts[vi].co = inv @ self._apply_world_locks(
                        old, old + deltas[source] * weight)

    def _hp_begin_adjust(self, context, target, show=True, settings_only=False):
        if getattr(self,'_hp_menu',None) is not None: return False
        curve=self._hp_is_curve
        if curve:
            self._refresh_selection_from_blender()
            order,full=self._selected_order()
            stroke=self._pen_stroke; level=self._pen_stabilizer
            points=self._panel_points
        elif target=='B':
            order,full=self._secondary_pen_selected_order()
            stroke=self._secondary_pen_stroke; level=self._secondary_pen_stabilizer
            points=self._secondary_panel_points()
        else:
            order,full=self._pen_selected_order()
            stroke=self._pen_stroke; level=self._pen_stabilizer
            points=self._panel_points
        if order is None or len(order)<2 or (not settings_only and len(stroke)<2):
            self.report({'WARNING'},'描線と連続した2点以上が必要です')
            return False
        closed=bool(full)
        if closed and not settings_only and (len(stroke)<3 or (stroke[-1]-stroke[0]).length>20):
            inferred = self._infer_closed_arc_order(stroke, points)
            if inferred is None or len(inferred) < 2:
                self.report({'WARNING'},'閉じたループ上の近い点を判定できません')
                return False
            order = inferred
            closed = False
            _hp_log(f"adjust inferred closed arc points={len(order)}")
        settings=_hp_settings_complete(getattr(self,'_hp_settings',None))
        if not closed: settings['shape']=False
        state=dict(target=target,order=list(order),closed=closed,
                   settings_only=bool(settings_only),
                   stroke=[] if settings_only else _hp_stabilize(stroke,level=level,closed=closed),
                   panel=[points[i].copy() for i in order],settings=settings,
                   original_settings=dict(settings),drag=None,wait_release=True,
                   data=self._obj.data, sides=[[],[]],direct_sides=[[],[]],show=show)
        if curve:
            state['base']=self._snapshot_all()
            state['source_world']=[self._point_world(i).copy() for i in order]
        else:
            self._bm.verts.ensure_lookup_table(); self._bm.verts.index_update()
            self._bm.edges.ensure_lookup_table()
            state['counts']=(len(self._bm.verts),len(self._bm.edges),len(self._bm.faces))
            state['ring']=(list(self._ordered) if settings_only
                           else [self._ordered[i] for i in order])
            # Follow complete sections through unambiguous quad strips.
            if (settings_only or full) and (show or settings['side_a'] or settings['side_b']):
                faces=[tuple(v.index for v in f.verts) for f in self._bm.faces]
                state['sides']=_hp_adjacent_rings(
                    state['ring'],
                    faces,
                    stop_triangles=bool(settings['stop_triangles']),
                    closed=self._closed,
                )
            state['direct_sides']=self._hp_follow_sides(settings)
            affected=set(self._ordered)
            for side in state['sides']:
                for ring in side: affected.update(ring)
            state['base']={i:self._bm.verts[i].co.copy() for i in affected}
            state['depths']=list(self._depths)
            state['source_world']=[self._obj.matrix_world@state['base'][i] for i in state['ring']]
            self._secondary_capture_map()
        self._hp_menu=state
        self._hp_preview(context)
        return self._hp_menu is not None

    def _hp_restore(self, state):
        if self._obj.data != state['data']: return False
        if self._hp_is_curve:
            return self._restore_snapshot(state['base'])
        if not self._bm.is_valid: return False
        if (len(self._bm.verts),len(self._bm.edges),len(self._bm.faces))!=state['counts']: return False
        for i,co in state['base'].items(): self._bm.verts[i].co=co.copy()
        bmesh.update_edit_mesh(self._obj.data,loop_triangles=False,destructive=False)
        self._sync_primary_from_mesh()
        return True

    def _hp_preview(self, context):
        state=self._hp_menu
        if not self._hp_restore(state):
            self.report({'WARNING'},'編集対象が変わったためプレビューを終了しました')
            self._hp_menu=None
            if not self._hp_is_curve: self._secondary_release_map()
            return
        settings=state['settings']
        if state.get('settings_only'):
            context.area.tag_redraw()
            return
        samples=_hp_pen_samples(state['stroke'],state['panel'],state['closed'],
                                settings['correction'],settings['shape'],self._pen_anchor_mode)
        goals=[]
        for index,p,base_world in zip(state['order'],samples,state['source_world']):
            if self._hp_is_curve:
                goal=self._plane_to_world(self._mouse_to_plane(p.x,p.y),base_world)
            elif state['target']=='B':
                goal=self._secondary_plane_to_world(self._secondary_mouse_to_plane(p.x,p.y),base_world)
            else:
                goal=self._obj.matrix_world@self._primary_local_from_values(self._mouse_to_section(p.x,p.y),state['depths'][index])
            goals.append(goal)
        amount=settings['amount']
        if self._hp_is_curve:
            for i,base,goal in zip(state['order'],state['source_world'],goals):
                self._set_point_world_from_base(i,state['base'][i],base.lerp(goal,amount))
            self._obj.data.update_tag()
        else:
            mw=self._obj.matrix_world; inv=mw.inverted()
            goal_map = {
                vi: base.lerp(goal, amount)
                for vi, base, goal in zip(state['ring'], state['source_world'], goals)
            }

            # The menu path must use the same endpoint blend as direct Pen
            # apply.  Previously only the selected arc was previewed here,
            # which made the blend controls appear to do nothing.
            blend_radius = max(0, min(8, int(settings.get('blend_radius', 2))))
            blend_strength = max(0.0, min(1.0, float(settings.get('blend_strength', 0.65))))
            selected = set(state['order'])
            source_closed = bool(self._closed)
            if blend_radius and blend_strength and len(state['order']) >= 2:
                for endpoint_pos in (0, len(state['order']) - 1):
                    endpoint = state['order'][endpoint_pos]
                    endpoint_vi = self._ordered[endpoint]
                    delta = goal_map[endpoint_vi] - state['source_world'][endpoint_pos]
                    direction = 1 if endpoint_pos == 0 else -1
                    for step in range(1, blend_radius + 1):
                        j = endpoint + direction * step
                        if source_closed:
                            j %= len(self._ordered)
                        elif j < 0 or j >= len(self._ordered):
                            continue
                        if j in selected:
                            continue
                        vi = self._ordered[j]
                        base_world = mw @ state['base'][vi]
                        weight = blend_strength * (1.0 - step / (blend_radius + 1.0)) ** 2
                        goal_map[vi] = base_world + delta * weight
                _hp_log(
                    f"menu endpoint blend points={len(goal_map)-len(state['ring'])} "
                    f"radius={blend_radius} strength={blend_strength:.2f}"
                )

            for vi, base in ((vi, mw @ state['base'][vi]) for vi in goal_map):
                self._bm.verts[vi].co=inv@self._apply_world_locks(base,goal_map[vi])
            for side,key in zip(state['sides'],('side_a','side_b')):
                count=min(int(settings[key]),len(side))
                for depth,ring in enumerate(side[:count],1):
                    weight=(1.0-depth/(count+1.0))**settings['falloff']
                    weight*=float(settings.get('follow_strength_a' if key=='side_a' else 'follow_strength_b',1.0))
                    base=[mw@state['base'][i] for i in ring]
                    moved=_hp_transfer_ring(state['source_world'],goals,base,amount*weight)
                    for vi,old,new in zip(ring,base,moved):
                        self._bm.verts[vi].co=inv@self._apply_world_locks(old,new)
            bmesh.update_edit_mesh(self._obj.data,loop_triangles=False,destructive=False)
            self._sync_primary_from_mesh()
        context.area.tag_redraw()

    def _hp_end_adjust(self, context, commit=False):
        state=getattr(self,'_hp_menu',None)
        if state is None: return
        if commit:
            if state.get('settings_only'):
                self._hp_settings=dict(state['settings'])
            elif self._hp_is_curve:
                self._history.append(state['base'])
            else:
                self._history.append({'hp_pen':True,'base':state['base'],
                                      'counts':state['counts'],'data':state['data']})
            if len(self._history)>self._history_limit: self._history.pop(0)
            self._hp_settings=dict(state['settings'])
            if self._hp_is_curve:
                self._pen_mode=False; self._pen_drawing=False
                self._pen_stroke=[]; self._pen_raw_stroke=[]; self._pen_smooth_level=0
            else:
                self._clear_all_pen_state()
        else:
            self._hp_restore(state)
        self._hp_menu=None
        if not self._hp_is_curve: self._secondary_release_map()
        if not self._hp_is_curve and commit:
            self._hp_refresh_follow_visual()
            _hp_follow_log(f"menu_commit settings={self._hp_settings} "
                           f"available=A:{len(state['direct_sides'][0])} "
                           f"B:{len(state['direct_sides'][1])}")
        if getattr(context,'area',None): context.area.tag_redraw()

    def _hp_menu_rows(self):
        state=self._hp_menu
        rows=[('correction','輪郭補正',0.0,20.0),('amount','反映量',0.0,1.0)]
        if not self._hp_is_curve:
            rows.extend([('side_a','隣接A ループ数',0,max(len(state['sides'][0]),len(state['direct_sides'][0]))),
                         ('side_b','隣接B ループ数',0,max(len(state['sides'][1]),len(state['direct_sides'][1]))),
                         ('direct_follow','点移動で隣接追従',0,1),
                         ('follow_strength_a','隣接A 強さ',0.0,1.0),
                         ('follow_strength_b','隣接B 強さ',0.0,1.0),
                         ('falloff','遠くほど弱くする強さ',0.5,3.0),
                         ('stop_triangles','三角面で停止',0,1),
                         ('blend_strength','端点なじみ強さ',0.0,1.0),
                         ('blend_radius','端点なじみ範囲',0,8)])
        return rows

    def _hp_menu_layout(self, context):
        rows=self._hp_menu_rows()
        width=min(380,max(240,context.region.width-24))
        height=178+len(rows)*38
        x=max(8,min(context.region.width-width-12,context.region.width*0.5-width*0.5))
        y=max(8,min(context.region.height-height-12,context.region.height*0.5-height*0.5))
        return x,y,width,height,rows

    def _hp_menu_value(self, context, key, mx):
        x,y,w,h,rows=self._hp_menu_layout(context)
        row=next(r for r in rows if r[0]==key)
        value=row[2]+max(0.0,min(1.0,(mx-x-18)/(w-36)))*(row[3]-row[2])
        if key.startswith('side_') or key in {'stop_triangles', 'blend_radius', 'direct_follow'}:
            value=int(round(value))
        self._hp_menu['settings'][key]=value
        if key == 'direct_follow' and value and not self._hp_is_curve:
            if not self._hp_enable_follow(self._hp_menu['settings'], self._hp_menu['direct_sides']):
                self.report({'WARNING'}, '隣接する四角面のループを検出できません')
        if key == 'stop_triangles' and not self._hp_is_curve:
            state=self._hp_menu
            self._hp_restore(state)
            faces=[tuple(v.index for v in f.verts) for f in self._bm.faces]
            state['sides']=_hp_adjacent_rings(
                state['ring'],
                faces,
                stop_triangles=bool(value),
                closed=self._closed,
            )
            for side in state['sides']:
                for ring in side:
                    for vi in ring:
                        if vi not in state['base']:
                            state['base'][vi]=self._bm.verts[vi].co.copy()
        self._hp_preview(context)

    def _hp_menu_event(self, context, event):
        state=self._hp_menu
        if event.type=='F' and event.value=='RELEASE':
            state['wait_release']=False
            return {'RUNNING_MODAL'}
        if event.value=='PRESS':
            if event.type in {'ESC','RIGHTMOUSE'} or (event.type=='F' and event.shift):
                self._hp_end_adjust(context,False)
                return {'RUNNING_MODAL'}
            if event.type in {'RET','NUMPAD_ENTER'} or (event.type=='F' and not state['wait_release'] and not getattr(event,'is_repeat',False)):
                self._hp_end_adjust(context,True)
                return {'RUNNING_MODAL'}
        if event.type=='LEFTMOUSE' and event.value=='RELEASE':
            state['drag']=None
            return {'RUNNING_MODAL'}
        if event.type in {'MOUSEMOVE','INBETWEEN_MOUSEMOVE'} and state['drag']:
            self._hp_menu_value(context,state['drag'],event.mouse_region_x)
            return {'RUNNING_MODAL'}
        if event.type=='LEFTMOUSE' and event.value=='PRESS':
            mx,my=event.mouse_region_x,event.mouse_region_y
            x,y,w,h,rows=self._hp_menu_layout(context)
            if x+12<=mx<=x+w-12 and y+18<=my<=y+50:
                self._hp_end_adjust(context, mx<x+w*0.5)
            elif x+12<=mx<=x+w-12 and y+h-82<=my<=y+h-52 and state['closed']:
                state['settings']['shape']=not state['settings']['shape']
                self._hp_preview(context)
            else:
                for n,(key,label,lo,hi) in enumerate(rows):
                    ry=y+h-118-n*38
                    if x+12<=mx<=x+w-12 and ry-10<=my<=ry+18:
                        if key == 'direct_follow':
                            settings = state['settings']
                            if settings.get('direct_follow', False):
                                settings['direct_follow'] = False
                            elif not self._hp_enable_follow(settings, state['direct_sides']):
                                self.report({'WARNING'}, '隣接する四角面のループを検出できません')
                            self._hp_preview(context)
                        else:
                            state['drag']=key
                            self._hp_menu_value(context,key,mx)
                        break
        return {'RUNNING_MODAL'}

    def _hp_pen_key(self, context, event):
        # This dispatcher runs before legacy F handling, but leaves empty-pen
        # anchor selection and E-smoothing unchanged.
        if getattr(self,'_hp_menu',None) is not None:
            return self._hp_menu_event(context,event)
        pressed=getattr(self,'_hp_confirm_hold',None)
        if pressed is not None:
            if event.type=='F' and event.value=='RELEASE':
                self._hp_confirm_hold=None
                show=time.perf_counter()-pressed[0]>=PEN_F_HOLD_SECONDS
                if self._hp_begin_adjust(context,pressed[1],show):
                    if show: self._hp_menu['wait_release']=False
                    else: self._hp_end_adjust(context,True)
                return {'RUNNING_MODAL'}
            if event.type=='TIMER' and time.perf_counter()-pressed[0]>=PEN_F_HOLD_SECONDS:
                self._hp_confirm_hold=None
                self._hp_begin_adjust(context,pressed[1],True)
                return {'RUNNING_MODAL'}
            if event.type in {'ESC','RIGHTMOUSE'} and event.value=='PRESS':
                self._hp_confirm_hold=None
                # Legacy handler below cancels the stroke.
                return None
            return {'RUNNING_MODAL'}
        if getattr(self,'_pen_smooth_dragging',False): return None
        target=None
        if self._pen_mode and self._pen_stroke and not self._pen_drawing: target='A'
        if not self._hp_is_curve and self._secondary_pen_mode and self._secondary_pen_stroke and not self._secondary_pen_drawing: target='B'
        if target is None: return None
        if event.type=='F' and event.value=='PRESS' and not event.ctrl and not event.shift and not event.alt:
            if not getattr(event,'is_repeat',False): self._hp_confirm_hold=(time.perf_counter(),target)
            return {'RUNNING_MODAL'}
        if event.type in {'RET','NUMPAD_ENTER'} and event.value=='PRESS':
            if self._hp_begin_adjust(context,target,False): self._hp_end_adjust(context,True)
            return {'RUNNING_MODAL'}
        return None

    def _hp_draw_menu(self, context):
        if getattr(self,'_hp_menu',None) is None: return
        x,y,w,h,rows=self._hp_menu_layout(context)
        gpu.state.blend_set('ALPHA')
        self._draw_rect(x,y,w,h,(0.035,0.045,0.06,0.97))
        def label(px,py,value,size=13,color=(.92,.94,.97,1)):
            blf.size(0,size); blf.color(0,*color); blf.position(0,px,py,0); blf.draw(0,value)
        title = '共有設定 — 描線なし' if self._hp_menu.get('settings_only') else 'ペン調整 — 立体に仮反映'
        label(x+16,y+h-28,title,16)
        settings=self._hp_menu['settings']
        self._draw_rect(x+12,y+h-82,w-24,30,(.10,.17,.23,1))
        mode='形だけ（中心・大きさを維持）' if settings['shape'] else '描線どおり（位置・大きさも反映）'
        if not self._hp_menu['closed']: mode='描線どおり（開いた線）'
        label(x+20,y+h-72,mode)
        for n,(key,title,lo,hi) in enumerate(rows):
            ry=y+h-118-n*38
            value=settings[key]
            if key in {'stop_triangles', 'direct_follow'}:
                text = 'ON' if value >= 0.5 else 'OFF'
            else:
                text=f'{value*100:.0f}%' if key in {'amount','follow_strength_a','follow_strength_b'} else (str(int(value)) if key.startswith('side_') else f'{value:.1f}')
                if key.startswith('side_'): text+=f' / {hi}'
            label(x+18,ry+11,title+'  '+text,12)
            self._draw_rect(x+18,ry-7,w-36,7,(.15,.18,.22,1))
            fraction=(value-lo)/(hi-lo) if hi>lo else 0
            self._draw_rect(x+18,ry-7,(w-36)*fraction,7,(.25,.75,.62,1))
        footer = 'F / Enter 確定    Esc 閉じる' if self._hp_menu.get('settings_only') else 'F / Enter 確定    Esc 描線へ戻る'
        label(x+16,y+65,footer,12)
        self._draw_rect(x+12,y+18,w*.5-16,32,(.12,.40,.32,1))
        self._draw_rect(x+w*.5+4,y+18,w*.5-16,32,(.23,.25,.29,1))
        label(x+30,y+28,'確定'); label(x+w*.5+20,y+28,'キャンセル')

    bl_idname = "hp.section_mini_editor"
    bl_label = "HP Section Mini Editor"
    bl_options = {'REGISTER', 'UNDO'}

    detached: BoolProperty(default=False, options={'SKIP_SAVE'})

    def _layout_panels(self, context):
        if self._preview_state or self._dragging or self._secondary_dragging or self._brush_mode or self._pen_drawing or self._secondary_pen_drawing or self._transform_mode or self._secondary_transform_mode:
            return
        wanted_w, wanted_h = self._panel_sizes[self._panel_stage]
        width = max(160, min(wanted_w, (context.region.width - 48) / 2))
        height = max(140, min(wanted_h, context.region.height * (0.42 if self.detached else 0.48)))
        x = 18 if self.detached else max(18, context.region.width - width * 2 - 30)
        y = 18 if self.detached else max(18, context.region.height - height - 18)
        layout = (width, height, x, y)
        if layout != (self._panel_w, self._panel_h, self._panel_x, self._panel_y):
            self._panel_w, self._panel_h, self._panel_x, self._panel_y = layout
            self._recompute_fit_scale()
            self._refresh_panel_points()
            self._secondary_invalidate_view()

    def _tag_views(self, context):
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

    def _target_available(self, context):
        if not self._bmesh_ready() or context.edit_object != self._obj:
            return False
        try:
            return bool(self._ordered) and all(0 <= i < len(self._bm.verts) for i in self._ordered)
        except (ReferenceError, RuntimeError):
            return False

    def invoke(self, context, event):
        global _RUNNING, _ACTIVE_SECTION_EDITOR, _PINNED_WORKSPACE_PTR

        if _RUNNING or context.area.type != 'VIEW_3D':
            return {'CANCELLED'}

        current_workspace_ptr = (
            context.window.workspace.as_pointer()
            if (
                context.window is not None
                and context.window.workspace is not None
            )
            else 0
        )

        self._finished = False
        self._viewport_navigation_active = False
        self._direct_follow_state = None
        self._hp_visual_spans = [[], []]
        self._hp_settings = _hp_settings_complete(getattr(self, '_hp_settings', None))
        self._timer_owner = context.window_manager
        self._handle = None
        self._handle_follow = None
        self._timer = None

        # The mini editor belongs only to the View3D/workspace where it started.
        # SpaceView3D draw handlers are global, so without this guard the same
        # overlay can appear in Animation / other workspace tabs.
        self._owner_window_ptr = (
            context.window.as_pointer()
            if context.window is not None
            else 0
        )
        self._owner_workspace_ptr = (
            context.window.workspace.as_pointer()
            if (
                context.window is not None
                and context.window.workspace is not None
            )
            else 0
        )
        self._owner_area_ptr = (
            context.area.as_pointer()
            if context.area is not None
            else 0
        )

        self._signature = None
        self._obj = None
        self._bm = None
        self._ordered = []
        self._closed = False

        self._center = Vector()
        self._u = Vector((1, 0, 0))
        self._v = Vector((0, 1, 0))
        self._normal = Vector((0, 0, 1))
        self._coords = []
        self._depths = []

        self._panel_points = []
        self._fit_center = Vector((0, 0))
        self._fit_scale = 1.0
        self._view_zoom = 1.0
        self._view_zoom_min = 0.35
        self._view_zoom_max = 5.0

        self._panel_sizes = list(WINDOW_STAGES)
        self._panel_stage = 1
        self._panel_w = self._panel_sizes[self._panel_stage][0]
        self._panel_h = self._panel_sizes[self._panel_stage][1]

        self._panel_x = 18
        self._panel_y = 18
        self._idle = True
        self._preview_state = None
        self._hp_menu = None
        self._preview_filter_backup = {}
        self._preview_overlay_backup = None

        self._view_a_collapsed = False
        self._view_b_collapsed = False

        # View A keeps its original SECTION projection by default, but can also
        # use Blender-style named orthographic views.
        self._primary_view = 'SECTION'
        self._primary_world_center = Vector((0.0, 0.0, 0.0))
        self._primary_world_u = Vector((1.0, 0.0, 0.0))
        self._primary_world_v = Vector((0.0, 0.0, 1.0))
        self._primary_world_n = Vector((0.0, -1.0, 0.0))

        self._secondary_view = 'FRONT'
        self._secondary_plane = 'XZ'
        self._secondary_zoom = 1.0
        self._secondary_flip_y = False
        self._xray = False
        self._secondary_occlusion_px = 10.0
        self._primary_occlusion_px = 10.0

        # View B editing state.
        self._secondary_dragging = False
        self._secondary_drag_start_mouse = Vector((0.0, 0.0))
        self._secondary_drag_base_world = []

        self._secondary_box_dragging = False
        self._secondary_box_start = None
        self._secondary_box_end = None
        self._secondary_box_additive = False

        self._secondary_transform_mode = None
        self._secondary_transform_axis = None
        self._secondary_transform_start_mouse = Vector((0.0, 0.0))
        self._secondary_transform_base_world = []
        self._secondary_transform_pivot_world = Vector((0.0, 0.0, 0.0))

        self._secondary_pen_mode = False
        self._secondary_pen_drawing = False
        self._secondary_pen_stroke = []
        self._secondary_pen_stabilizer = 2

        # Freeze View B display mapping during mouse transforms so it does not
        # auto-refit under the cursor.
        self._secondary_map_frozen = False
        self._secondary_frozen_center = Vector((0.0, 0.0))
        self._secondary_frozen_panel_center = Vector((0.0, 0.0))
        self._secondary_frozen_scale = 1.0
        self._secondary_frozen_points = []

        # Persistent View B camera. Unlike the old per-frame auto-fit,
        # this mapping stays fixed while the mesh moves, so translations,
        # scaling and shape changes are actually visible in real time.
        self._secondary_view_valid = False
        self._secondary_view_center = Vector((0.0, 0.0))
        self._secondary_view_panel_center = Vector((0.0, 0.0))
        self._secondary_view_base_scale = 1.0

        self._selected = set()

        self._resize_mode = False
        self._resize_start_x = 0
        self._resize_start_stage = 0

        # Blender-like mini-window transform state.
        self._transform_mode = None   # None / G / S / R
        self._transform_axis = None   # None / X / Y / Z
        self._transform_start_mouse = Vector((0.0, 0.0))
        self._transform_base_coords = []
        self._transform_base_depths = []
        self._transform_base_world = []
        self._transform_base_panel = []
        self._transform_pivot_sec = Vector((0.0, 0.0))
        self._transform_pivot_world = Vector((0.0, 0.0, 0.0))

        self._transform_fit_center = Vector((0.0, 0.0))
        self._transform_fit_scale = 1.0
        self._transform_flip_y = False
        self._transform_flip_x = False
        self._transform_section_rotation = 0.0

        self._dragging = False
        self._drag_start_sec = None
        self._drag_base_coords = []
        self._drag_base_depths = []
        self._drag_base_world = []
        self._drag_base_panel = []

        self._box_mode = False
        self._box_dragging = False
        self._box_start = None
        self._box_end = None
        self._box_additive = False

        self._prop_enabled = False
        self._prop_radius_px = 90.0

        global _LAST_SECTION_FLIP_Y

        self._flip_y = bool(
            _LAST_SECTION_FLIP_Y
        )
        self._flip_x = False

        # View A SECTION-only display rotation.  The remembered orientation is
        # reconstructed from a WORLD direction after each section basis is
        # rebuilt, because raw 2D angle is not transferable between sections.
        self._section_rotation = 0.0

        self._lock_x = False
        self._lock_y = False
        self._lock_z = False

        # Mini-window local undo history.
        self._history = []
        self._history_limit = 64

        # Topology uses Blender's undo stack because coordinate snapshots
        # cannot restore point/edge count changes.
        self._topology_undo_steps = 0
        self._topology_rebuild_pending = False

        # Keep the mini-window point selection separate from Blender's
        # whole-chain edge selection used only to keep the editor alive.
        self._topology_pending_selected = None
        self._topology_selection_history = []

        # Mini-window pen-fit mode.
        self._pen_mode = False
        self._pen_drawing = False
        self._pen_stroke = []
        self._pen_stabilizer = 2

        # Pen start mode:
        # STROKE = first target point goes to the position where the stroke begins.
        # POINT  = stroke shape is translated so the first target point stays fixed.
        self._pen_anchor_mode = _SECTION_PEN_ANCHOR_DEFAULT

        # F tap / hold handling.
        self._f_hold_active = False
        self._f_hold_started = 0.0
        self._f_hold_target = 'A'
        self._f_long_opened = False
        self._pie_target = 'A'
        self._shape_pie_target = 'A'

        # Interactive E Smooth drag for selected geometry.
        self._smooth_dragging = False
        self._brush_mode = None
        self._smooth_target = None
        self._smooth_start_x = 0.0
        self._smooth_amount = 0.0
        self._smooth_order = []
        self._smooth_closed = False
        self._smooth_base_panel = []
        self._smooth_base_local = []
        self._smooth_base_world = []
        self._smooth_base_depths = []
        self._smooth_brush_radius = 70.0
        self._smooth_brush_strength = 0.55
        self._smooth_brush_prev_mouse = Vector((0.0, 0.0))

        # Interactive E Post Smooth for the DRAWN PEN STROKE.
        # This edits only the visible, not-yet-applied stroke.
        self._pen_smooth_dragging = False
        self._brush_mode = None
        self._pen_smooth_target = None
        self._pen_smooth_start_x = 0.0
        self._pen_smooth_amount = 0.0
        self._pen_smooth_base = []
        self._pen_smooth_closed = False
        self._pen_smooth_last_delta_px = 0.0
        self._pen_smooth_start_amount = 0.0
        self._pen_smooth_cancel_stroke = []
        self._pen_smooth_cancel_amount = 0.0
        self._pen_smooth_brush_radius = 70.0
        self._pen_smooth_brush_strength = 0.55
        self._pen_smooth_brush_prev_mouse = Vector((0.0, 0.0))

        # Persistent original stroke + current Smooth level.
        self._pen_raw_stroke = []
        self._pen_smooth_level = 0.0
        self._secondary_pen_raw_stroke = []
        self._secondary_pen_smooth_level = 0.0

        # Floating Smooth HUD shown beside the mouse while E-dragging.
        self._smooth_hud_x = 0.0
        self._smooth_hud_y = 0.0

        chain = _selected_chain(context)

        _RUNNING = True
        _ACTIVE_SECTION_EDITOR = self
        if chain is not None:
            self._rebuild(context, chain)
            self._idle = False
        self._layout_panels(context)
        if self.detached:
            space = context.space_data
            for prop in space.bl_rna.properties:
                if prop.identifier.startswith('show_object_viewport_'):
                    self._preview_filter_backup[prop.identifier] = getattr(space, prop.identifier)
                    setattr(space, prop.identifier, False)
            self._preview_space = space
            self._preview_overlay_backup = space.overlay.show_overlays
            space.overlay.show_overlays = False
            space.show_region_ui = False
            space.show_region_toolbar = False
            if chain is not None:
                self._frame_preview(context)

        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw, (), 'WINDOW', 'POST_PIXEL'
        )
        self._handle_follow = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_follow_3d, (), 'WINDOW', 'POST_VIEW'
        )

        self._handle_labels = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_point_labels, (), 'WINDOW', 'POST_PIXEL'
        )
        self._timer = context.window_manager.event_timer_add(
            0.12, window=context.window
        )

        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()

        return {'RUNNING_MODAL'}

    def _owner_workspace_active(self, context):
        if context is None:
            return False

        window = getattr(context, "window", None)
        if window is None:
            return False

        try:
            if (
                self._owner_window_ptr
                and window.as_pointer() != self._owner_window_ptr
            ):
                return False

            workspace = getattr(window, "workspace", None)

            return bool(
                workspace is not None
                and (
                    not self._owner_workspace_ptr
                    or workspace.as_pointer() == self._owner_workspace_ptr
                )
            )
        except ReferenceError:
            return False

    def _owner_context_active(self, context):
        if context is None:
            return False

        window = getattr(context, "window", None)
        area = getattr(context, "area", None)

        if window is None or area is None:
            return False

        try:
            if (
                self._owner_window_ptr
                and window.as_pointer()
                != self._owner_window_ptr
            ):
                return False

            workspace = getattr(
                window,
                "workspace",
                None
            )

            if (
                workspace is None
                or (
                    self._owner_workspace_ptr
                    and workspace.as_pointer()
                    != self._owner_workspace_ptr
                )
            ):
                return False

            if (
                self._owner_area_ptr
                and area.as_pointer()
                != self._owner_area_ptr
            ):
                return False

        except ReferenceError:
            return False

        return True

    def _bmesh_ready(self):
        if self._obj is None or self._bm is None:
            return False

        try:
            return bool(self._bm.is_valid)
        except (ReferenceError, RuntimeError):
            return False

    def _ensure_live_bmesh(self, context):
        if self._bmesh_ready():
            return True

        chain = _selected_chain(context)
        if chain is None:
            return False

        return bool(self._rebuild(context, chain))

    def _primary_view_shortcut_label(self):
        return {
            'SECTION': 'Num7',
            'FRONT': 'Num1',
            'BACK': 'Ctrl+Num1',
            'RIGHT': 'Num3',
            'LEFT': 'Ctrl+Num3',
        }.get(self._primary_view, '')

    def _set_primary_view(
        self,
        context,
        view_name,
        auto_pair_secondary=False
    ):
        if view_name not in {
            'SECTION',
            'FRONT',
            'BACK',
            'RIGHT',
            'LEFT',
        }:
            return

        previous_view = self._primary_view

        if previous_view == view_name:
            return

        self._primary_view = view_name

        # Leaving SECTION creates a useful perpendicular pair once.
        # After that, View A / View B remain independent until Num7 returns
        # View A to SECTION.
        if (
            auto_pair_secondary
            and previous_view == 'SECTION'
        ):
            if view_name in {'FRONT', 'BACK'}:
                self._set_secondary_view('RIGHT')
            elif view_name in {'RIGHT', 'LEFT'}:
                self._set_secondary_view('FRONT')

        chain = _selected_chain(context)
        if chain is not None:
            self._rebuild(context, chain)

    def _named_primary_axes(self):
        # Screen horizontal, screen vertical, screen depth/camera-side normal.
        if self._primary_view == 'FRONT':
            return (
                Vector((1.0, 0.0, 0.0)),
                Vector((0.0, 0.0, 1.0)),
                Vector((0.0, -1.0, 0.0)),
            )

        if self._primary_view == 'BACK':
            return (
                Vector((-1.0, 0.0, 0.0)),
                Vector((0.0, 0.0, 1.0)),
                Vector((0.0, 1.0, 0.0)),
            )

        if self._primary_view == 'RIGHT':
            return (
                Vector((0.0, 1.0, 0.0)),
                Vector((0.0, 0.0, 1.0)),
                Vector((1.0, 0.0, 0.0)),
            )

        # LEFT
        return (
            Vector((0.0, -1.0, 0.0)),
            Vector((0.0, 0.0, 1.0)),
            Vector((-1.0, 0.0, 0.0)),
        )

    def _primary_values_from_local(self, local):
        if self._primary_view == 'SECTION':
            d = local - self._center
            return (
                Vector((
                    d.dot(self._u),
                    d.dot(self._v),
                )),
                d.dot(self._normal),
            )

        world = self._obj.matrix_world @ local
        d = world - self._primary_world_center

        return (
            Vector((
                d.dot(self._primary_world_u),
                d.dot(self._primary_world_v),
            )),
            d.dot(self._primary_world_n),
        )

    def _primary_local_from_values(self, sec, depth):
        if self._primary_view == 'SECTION':
            return (
                self._center
                + self._u * sec.x
                + self._v * sec.y
                + self._normal * depth
            )

        world = (
            self._primary_world_center
            + self._primary_world_u * sec.x
            + self._primary_world_v * sec.y
            + self._primary_world_n * depth
        )

        return self._obj.matrix_world.inverted() @ world

    def _primary_section_delta_to_world(self, sec_delta):
        if self._primary_view == 'SECTION':
            local_delta = (
                self._u * sec_delta.x
                + self._v * sec_delta.y
            )
            return self._obj.matrix_world.to_3x3() @ local_delta

        return (
            self._primary_world_u * sec_delta.x
            + self._primary_world_v * sec_delta.y
        )

    def _rebuild(self, context, chain=None):
        if chain is None:
            chain = _selected_chain(context)

        if chain is None:
            return False

        self._obj, self._bm, self._ordered, self._closed, self._signature = chain

        if self._primary_view == 'SECTION':
            (
                self._center,
                self._u,
                self._v,
                self._normal,
                self._coords,
                self._depths
            ) = _section_basis(
                self._bm,
                self._ordered,
                self._closed
            )
        else:
            # Named views are evaluated directly in world space so Front/Back/
            # Right/Left match Blender even if the object is rotated.
            mw = self._obj.matrix_world
            world_points = [
                mw @ self._bm.verts[vi].co.copy()
                for vi in self._ordered
            ]

            self._primary_world_center = (
                sum(world_points, Vector((0.0, 0.0, 0.0)))
                / len(world_points)
            )

            (
                self._primary_world_u,
                self._primary_world_v,
                self._primary_world_n,
            ) = self._named_primary_axes()

            self._coords = []
            self._depths = []

            for world in world_points:
                d = world - self._primary_world_center
                self._coords.append(Vector((
                    d.dot(self._primary_world_u),
                    d.dot(self._primary_world_v),
                )))
                self._depths.append(
                    d.dot(self._primary_world_n)
                )

        xs = [p.x for p in self._coords]
        ys = [p.y for p in self._coords]

        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)

        self._fit_center = Vector((
            (minx + maxx) * 0.5,
            (miny + maxy) * 0.5
        ))

        self._selected = {
            i for i in self._selected
            if i < len(self._ordered)
        }

        # A raw 2D rotation angle is meaningless for another section because
        # its U/V basis can differ. Re-project the remembered WORLD-down
        # direction into the freshly-built section plane and derive a new angle.
        if self._primary_view == 'SECTION':
            self._apply_remembered_section_orientation()
            self._apply_remembered_section_right()

        self._recompute_fit_scale()
        self._refresh_panel_points()
        self._hp_refresh_follow_visual()

        # A new selected chain deserves one fresh View B fit.
        # Coordinate-only edits do NOT call this repeatedly during motion.
        self._secondary_invalidate_view()

        return True

    def _section_display_angle(self):
        if self._primary_view != 'SECTION':
            return 0.0

        return float(self._section_rotation)

    @staticmethod
    def _rotate_2d(vec, angle):
        if abs(angle) <= 1.0e-12:
            return vec.copy()

        c = cos(angle)
        s = sin(angle)

        return Vector((
            vec.x * c - vec.y * s,
            vec.x * s + vec.y * c,
        ))

    @staticmethod
    def _normalize_angle(angle):
        while angle > 3.141592653589793:
            angle -= tau

        while angle <= -3.141592653589793:
            angle += tau

        return angle

    def _section_rotation_for_raw_down(
        self,
        raw_vec,
        include_flip=True
    ):
        if raw_vec is None or raw_vec.length < 1.0e-8:
            return None

        source_angle = atan2(
            raw_vec.y,
            raw_vec.x
        )

        if include_flip:
            # Interactive Num2/4/6 command:
            # requested direction must look DOWN on screen right now.
            target_angle = (
                0.5 * 3.141592653589793
                if self._flip_y
                else -0.5 * 3.141592653589793
            )
        else:
            # Session restore:
            # rebuild the remembered BASE orientation first.
            # Num8 flip is applied later by the normal display mapping.
            target_angle = -0.5 * 3.141592653589793

        return self._normalize_angle(
            target_angle - source_angle
        )

    def _remember_raw_down_world(
        self,
        raw_vec,
        mode='SELECTED'
    ):
        """
        Convert a SECTION 2D direction back to world space and remember it.
        This is what allows the next, differently-oriented section basis to
        reconstruct the same visual 'down' direction.
        """
        global _LAST_SECTION_DOWN_WORLD, _LAST_SECTION_ORIENT_MODE

        if (
            raw_vec is None
            or raw_vec.length < 1.0e-8
            or self._obj is None
        ):
            return

        local_dir = (
            self._u * raw_vec.x
            + self._v * raw_vec.y
        )

        world_dir = (
            self._obj.matrix_world.to_3x3()
            @ local_dir
        )

        if world_dir.length < 1.0e-8:
            return

        # Memory stores BASE orientation independently from Num8.
        # If the current display is vertically flipped, the visible-down
        # direction corresponds to the opposite pre-flip world direction.
        if self._flip_y:
            world_dir = -world_dir

        _LAST_SECTION_DOWN_WORLD = world_dir.normalized().copy()
        _LAST_SECTION_ORIENT_MODE = mode

    def _remember_world_down(
        self,
        world_dir,
        mode
    ):
        global _LAST_SECTION_DOWN_WORLD, _LAST_SECTION_ORIENT_MODE

        if (
            world_dir is None
            or world_dir.length < 1.0e-8
        ):
            return

        remembered = world_dir.copy()

        # Num4/Num6 should point visibly DOWN at the moment they are pressed.
        # Store the equivalent pre-flip base direction so Num8 remains a
        # completely independent remembered operation.
        if self._flip_y:
            remembered = -remembered

        _LAST_SECTION_DOWN_WORLD = remembered.normalized().copy()
        _LAST_SECTION_ORIENT_MODE = mode

    def _raw_from_world_direction(
        self,
        world_dir
    ):
        if (
            self._obj is None
            or world_dir is None
            or world_dir.length < 1.0e-8
        ):
            return None

        local_dir = (
            self._obj.matrix_world.to_3x3().inverted()
            @ world_dir
        )

        raw = Vector((
            local_dir.dot(self._u),
            local_dir.dot(self._v),
        ))

        if raw.length < 1.0e-8:
            return None

        return raw

    def _apply_remembered_section_orientation(self):
        global _LAST_SECTION_DOWN_WORLD

        if (
            self._primary_view != 'SECTION'
            or _LAST_SECTION_DOWN_WORLD is None
        ):
            return False

        raw = self._raw_from_world_direction(
            _LAST_SECTION_DOWN_WORLD
        )

        angle = self._section_rotation_for_raw_down(
            raw,
            include_flip=False
        )

        if angle is None:
            return False

        self._section_rotation = angle
        return True

    def _set_section_vector_down(
        self,
        raw_vec,
        remember=True,
        mode='SELECTED'
    ):
        """
        Rotate View A SECTION so raw_vec appears screen-down.
        """
        if self._primary_view != 'SECTION':
            return False

        angle = self._section_rotation_for_raw_down(
            raw_vec,
            include_flip=True
        )

        if angle is None:
            return False

        self._section_rotation = angle

        if remember:
            self._remember_raw_down_world(
                raw_vec,
                mode=mode
            )

        self._recompute_fit_scale()
        self._refresh_panel_points()
        return True

    def _orient_selected_down(self):
        if (
            self._primary_view != 'SECTION'
            or not self._selected
        ):
            return False

        valid = [
            i for i in sorted(self._selected)
            if 0 <= i < len(self._coords)
        ]

        if not valid:
            return False

        selected_center = (
            sum(
                (
                    self._coords[i]
                    for i in valid
                ),
                Vector((0.0, 0.0))
            )
            / len(valid)
        )

        raw = (
            selected_center
            - self._fit_center
        )

        # Symmetric selections can average to the exact section center.
        # Use the farthest selected point as a deterministic fallback.
        if raw.length < 1.0e-8:
            farthest_i = max(
                valid,
                key=lambda i: (
                    self._coords[i]
                    - self._fit_center
                ).length_squared
            )

            raw = (
                self._coords[farthest_i]
                - self._fit_center
            )

        changed = self._set_section_vector_down(
            raw,
            remember=True,
            mode='SELECTED'
        )

        if changed:
            self._remember_current_screen_right()

        return changed

    def _world_axis_section_raw_dir(
        self,
        world_axis
    ):
        if (
            self._primary_view != 'SECTION'
            or self._obj is None
        ):
            return None

        local_axis = (
            self._obj.matrix_world.to_3x3().inverted()
            @ world_axis
        )

        raw = Vector((
            local_axis.dot(self._u),
            local_axis.dot(self._v),
        ))

        if raw.length < 1.0e-8:
            return None

        return raw

    def _current_screen_right_world(self):
        """
        Return the current visible screen-right direction expressed in world
        space.  This captures the actual handedness after rotation + flip_x,
        so it can be reconstructed on another section with a different U/V
        basis.
        """
        if self._obj is None:
            return None

        sx = (
            -1.0
            if self._flip_x
            else 1.0
        )

        raw = self._rotate_2d(
            Vector((sx, 0.0)),
            -self._section_rotation
        )

        local_dir = (
            self._u * raw.x
            + self._v * raw.y
        )

        world_dir = (
            self._obj.matrix_world.to_3x3()
            @ local_dir
        )

        if world_dir.length < 1.0e-8:
            return None

        return world_dir.normalized()

    def _remember_current_screen_right(self):
        global _LAST_SECTION_RIGHT_WORLD

        world_dir = self._current_screen_right_world()

        if world_dir is None:
            return False

        _LAST_SECTION_RIGHT_WORLD = world_dir.copy()
        return True

    def _set_world_direction_right(
        self,
        world_dir,
        remember=True
    ):
        """
        Choose flip_x so the requested WORLD direction appears screen-right
        with the current SECTION rotation.
        """
        global _LAST_SECTION_RIGHT_WORLD

        raw = self._world_axis_section_raw_dir(
            world_dir
        )

        if raw is None:
            return False

        q = self._rotate_2d(
            raw,
            self._section_rotation
        )

        if abs(q.x) < 1.0e-8:
            return False

        self._flip_x = bool(
            q.x < 0.0
        )

        if remember:
            _LAST_SECTION_RIGHT_WORLD = world_dir.normalized().copy()

        self._refresh_panel_points()
        return True

    def _apply_remembered_section_right(self):
        global _LAST_SECTION_RIGHT_WORLD

        if (
            self._primary_view != 'SECTION'
            or _LAST_SECTION_RIGHT_WORLD is None
        ):
            return False

        return self._set_world_direction_right(
            _LAST_SECTION_RIGHT_WORLD,
            remember=False
        )

    def _screen_dir_for_world_axis(
        self,
        world_axis
    ):
        """
        Current on-screen direction of a world axis in View A SECTION,
        including rotation + horizontal/vertical flips.
        """
        raw = self._world_axis_section_raw_dir(
            world_axis
        )

        if raw is None:
            return None

        q = self._rotate_2d(
            raw,
            self._section_rotation
        )

        if self._flip_x:
            q.x *= -1.0

        if self._flip_y:
            q.y *= -1.0

        return q

    def _canonicalize_companion_axis_right(
        self,
        companion_world_axis
    ):
        """
        Num4 / Num6 canonical handedness:
        remember the companion WORLD axis itself as screen-right.
        """
        return self._set_world_direction_right(
            companion_world_axis,
            remember=True
        )

    def _orient_world_axis_down(
        self,
        world_axis,
        mode='WORLD'
    ):
        raw = self._world_axis_section_raw_dir(
            world_axis
        )

        if raw is None:
            return False

        self._remember_world_down(
            world_axis,
            mode
        )

        return self._set_section_vector_down(
            raw,
            remember=False,
            mode=mode
        )

    def _recompute_fit_scale(self):
        if not self._coords:
            self._fit_scale = 1.0
            return

        angle = self._section_display_angle()

        display_rel = [
            self._rotate_2d(
                p - self._fit_center,
                angle
            )
            for p in self._coords
        ]

        xs = [p.x for p in display_rel]
        ys = [p.y for p in display_rel]

        sx = max(max(xs) - min(xs), 1e-6)
        sy = max(max(ys) - min(ys), 1e-6)

        usable_w = self._panel_w - PANEL_PAD * 2
        usable_h = max(20, self._panel_h - PANEL_PAD * 2 - 64)

        base_scale = min(
            usable_w / sx,
            usable_h / sy
        ) * 0.90

        self._fit_scale = base_scale * self._view_zoom

    def _refresh_panel_points(self):
        pcx = self._panel_x + self._panel_w * 0.5
        pcy = self._panel_y + self._panel_h * 0.5 - 14

        sx = -1.0 if self._flip_x else 1.0
        sy = -1.0 if self._flip_y else 1.0
        angle = self._section_display_angle()

        self._panel_points = []

        for p in self._coords:
            q = self._rotate_2d(
                p - self._fit_center,
                angle
            )

            self._panel_points.append(Vector((
                pcx + q.x * self._fit_scale * sx,
                pcy + q.y * self._fit_scale * sy
            )))

    def _mouse_to_section(self, mx, my):
        pcx = self._panel_x + self._panel_w * 0.5
        pcy = self._panel_y + self._panel_h * 0.5 - 14

        sx = -1.0 if self._flip_x else 1.0
        sy = -1.0 if self._flip_y else 1.0
        angle = self._section_display_angle()
        scale = max(
            self._fit_scale,
            1.0e-12
        )

        display_rel = Vector((
            (mx - pcx) / (scale * sx),
            (my - pcy) / (scale * sy),
        ))

        raw_rel = self._rotate_2d(
            display_rel,
            -angle
        )

        return (
            self._fit_center
            + raw_rel
        )

    def _primary_height(self):
        return 34 if self._view_a_collapsed else self._panel_h

    def _secondary_x(self):
        return self._panel_x + self._panel_w + 12

    def _secondary_height(self):
        return 34 if self._view_b_collapsed else self._panel_h

    def _inside_primary_header(self, x, y):
        h = self._primary_height()
        return (
            self._panel_x <= x <= self._panel_x + self._panel_w
            and self._panel_y + h - 34 <= y <= self._panel_y + h
        )

    def _inside_secondary_header(self, x, y):
        sx = self._secondary_x()
        h = self._secondary_height()
        return (
            sx <= x <= sx + self._panel_w
            and self._panel_y + h - 34 <= y <= self._panel_y + h
        )

    def _inside_panel(self, x, y):
        if self._view_a_collapsed:
            return False
        return (
            self._panel_x <= x <= self._panel_x + self._panel_w
            and self._panel_y <= y <= self._panel_y + self._panel_h
        )

    def _inside_secondary(self, x, y):
        if self._view_b_collapsed:
            return False
        sx = self._secondary_x()
        return (
            sx <= x <= sx + self._panel_w
            and self._panel_y <= y <= self._panel_y + self._panel_h
        )

    def _set_secondary_view(self, view_name):
        """
        Blender-style orthographic views:
        NUM1       = FRONT  (camera on -Y, looking +Y)
        Ctrl+NUM1  = BACK   (camera on +Y, looking -Y)
        NUM3       = RIGHT  (camera on +X, looking -X)
        Ctrl+NUM3  = LEFT   (camera on -X, looking +X)
        """
        if view_name not in {'FRONT', 'BACK', 'RIGHT', 'LEFT'}:
            return

        self._secondary_view = view_name

        if view_name in {'FRONT', 'BACK'}:
            self._secondary_plane = 'XZ'
        else:
            self._secondary_plane = 'YZ'

        self._secondary_invalidate_view()

    def _secondary_view_shortcut_label(self):
        return {
            'FRONT': 'Num1',
            'BACK': 'Ctrl+Num1',
            'RIGHT': 'Num3',
            'LEFT': 'Ctrl+Num3',
        }.get(self._secondary_view, '')

    def _secondary_world_to_plane(self, world):
        # Match Blender-style orthographic screen orientation.
        if self._secondary_view == 'FRONT':
            return Vector((world.x, world.z))

        if self._secondary_view == 'BACK':
            # Back view mirrors horizontal X.
            return Vector((-world.x, world.z))

        if self._secondary_view == 'RIGHT':
            return Vector((world.y, world.z))

        # LEFT: horizontal Y is mirrored.
        return Vector((-world.y, world.z))

    def _secondary_plane_to_world(self, plane_co, original_world):
        world = original_world.copy()

        if self._secondary_view == 'FRONT':
            world.x = plane_co.x
            world.z = plane_co.y

        elif self._secondary_view == 'BACK':
            world.x = -plane_co.x
            world.z = plane_co.y

        elif self._secondary_view == 'RIGHT':
            world.y = plane_co.x
            world.z = plane_co.y

        else:  # LEFT
            world.y = -plane_co.x
            world.z = plane_co.y

        return world

    def _secondary_world_coords(self):
        if self._obj is None or self._bm is None:
            return []

        mw = self._obj.matrix_world

        return [
            self._secondary_world_to_plane(
                mw @ self._bm.verts[vi].co.copy()
            )
            for vi in self._ordered
        ]

    def _secondary_invalidate_view(self):
        self._secondary_view_valid = False

    def _secondary_refit_view(self):
        coords = self._secondary_world_coords()

        if not coords:
            self._secondary_view_valid = False
            return False

        sx = self._secondary_x()

        panel_center = Vector((
            sx + self._panel_w * 0.5,
            self._panel_y + self._panel_h * 0.5 - 2
        ))

        xs = [p.x for p in coords]
        ys = [p.y for p in coords]

        center = Vector((
            (min(xs) + max(xs)) * 0.5,
            (min(ys) + max(ys)) * 0.5
        ))

        span_x = max(
            max(xs) - min(xs),
            1e-6
        )
        span_y = max(
            max(ys) - min(ys),
            1e-6
        )

        usable_w = self._panel_w - PANEL_PAD * 2
        usable_h = self._panel_h - PANEL_PAD * 2 - 30

        base_scale = min(
            usable_w / span_x,
            usable_h / span_y
        ) * 0.90

        self._secondary_view_center = center
        self._secondary_view_panel_center = panel_center
        self._secondary_view_base_scale = base_scale
        self._secondary_view_valid = True

        return True

    def _secondary_fit_info(self):
        coords = self._secondary_world_coords()

        if not coords:
            return None

        if not self._secondary_view_valid:
            if not self._secondary_refit_view():
                return None

        scale = (
            self._secondary_view_base_scale
            * self._secondary_zoom
        )

        return (
            self._secondary_view_center.copy(),
            self._secondary_view_panel_center.copy(),
            scale,
            coords
        )

    def _secondary_capture_map(self):
        info = self._secondary_fit_info()
        if info is None:
            return False

        (
            self._secondary_frozen_center,
            self._secondary_frozen_panel_center,
            self._secondary_frozen_scale,
            coords
        ) = info

        # Snapshot of the START screen positions. These are used only for
        # proportional falloff / transform reference, not for drawing.
        sy = -1.0 if self._secondary_flip_y else 1.0

        self._secondary_frozen_points = [
            self._secondary_frozen_panel_center
            + Vector((
                (p.x - self._secondary_frozen_center.x)
                * self._secondary_frozen_scale,
                (p.y - self._secondary_frozen_center.y)
                * self._secondary_frozen_scale
                * sy,
            ))
            for p in coords
        ]

        self._secondary_map_frozen = True
        return True

    def _secondary_release_map(self):
        self._secondary_map_frozen = False
        self._secondary_frozen_points = []

    def _secondary_panel_points(self):
        sy = -1.0 if self._secondary_flip_y else 1.0

        if self._secondary_map_frozen:
            # Freeze ONLY the display mapping (center / scale / panel center).
            # The actual coordinates must remain live so View B visibly
            # follows the real mesh during direct drag / G/S/R.
            coords = self._secondary_world_coords()

            return [
                self._secondary_frozen_panel_center
                + Vector((
                    (p.x - self._secondary_frozen_center.x)
                    * self._secondary_frozen_scale,
                    (p.y - self._secondary_frozen_center.y)
                    * self._secondary_frozen_scale
                    * sy,
                ))
                for p in coords
            ]

        info = self._secondary_fit_info()
        if info is None:
            return []

        center, panel_center, scale, coords = info

        return [
            panel_center
            + Vector((
                (p.x - center.x) * scale,
                (p.y - center.y) * scale * sy,
            ))
            for p in coords
        ]

    def _secondary_mouse_to_plane(self, mx, my):
        if self._secondary_map_frozen:
            center = self._secondary_frozen_center
            panel_center = self._secondary_frozen_panel_center
            scale = max(self._secondary_frozen_scale, 1e-12)
        else:
            info = self._secondary_fit_info()
            if info is None:
                return Vector((0.0, 0.0))

            center, panel_center, scale, _ = info
            scale = max(scale, 1e-12)

        sy = -1.0 if self._secondary_flip_y else 1.0

        return Vector((
            (mx - panel_center.x) / scale + center.x,
            (my - panel_center.y) / (scale * sy) + center.y,
        ))

    def _secondary_depth_value(self, index):
        """
        Larger returned value means visually closer to the current View B camera.

        FRONT camera is on -Y -> smaller Y is closer.
        BACK  camera is on +Y -> larger Y is closer.
        RIGHT camera is on +X -> larger X is closer.
        LEFT  camera is on -X -> smaller X is closer.
        """
        mw = self._obj.matrix_world
        world = mw @ self._bm.verts[self._ordered[index]].co.copy()

        if self._secondary_view == 'FRONT':
            return -world.y

        if self._secondary_view == 'BACK':
            return world.y

        if self._secondary_view == 'RIGHT':
            return world.x

        return -world.x

    def _secondary_front_filter(self, indices, points=None):
        indices = list(indices)

        if self._xray or len(indices) <= 1:
            return indices

        if points is None:
            points = self._secondary_panel_points()

        ordered = sorted(
            indices,
            key=self._secondary_depth_value,
            reverse=True
        )

        visible = []
        radius = self._secondary_occlusion_px

        for i in ordered:
            p = points[i]

            if any(
                (p - points[j]).length <= radius
                for j in visible
            ):
                continue

            visible.append(i)

        return visible

    def _secondary_pick_point(self, x, y):
        points = self._secondary_panel_points()

        if not points:
            return None

        mouse = Vector((x, y))

        candidates = [
            i for i, p in enumerate(points)
            if (p - mouse).length <= 14.0
        ]

        if not candidates:
            return None

        candidates = self._secondary_front_filter(
            candidates,
            points
        )

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda i: (
                points[i] - mouse
            ).length
        )

    def _sync_primary_from_mesh(self):
        if self._obj is None or self._bm is None:
            return

        new_coords = []
        new_depths = []

        for vi in self._ordered:
            local = self._bm.verts[vi].co.copy()

            sec_now, depth_now = (
                self._primary_values_from_local(
                    local
                )
            )

            new_coords.append(sec_now)
            new_depths.append(depth_now)

        self._coords = new_coords
        self._depths = new_depths
        self._refresh_panel_points()

    def _secondary_begin_drag(self, mx, my):
        if not self._selected:
            return

        follow = self._hp_capture_direct_follow()
        if not self._push_history(follow):
            return
        self._direct_follow_state = follow

        if not self._secondary_capture_map():
            self._direct_follow_state = None
            return

        self._secondary_dragging = True
        self._secondary_drag_start_mouse = Vector((mx, my))

        mw = self._obj.matrix_world
        self._secondary_drag_base_world = [
            mw @ self._bm.verts[vi].co.copy()
            for vi in self._ordered
        ]

    def _secondary_weight(self, i, base_points):
        if i in self._selected:
            return 1.0

        if not self._prop_enabled or not self._selected:
            return 0.0

        # Match Blender-like visible-only proportional editing.
        # When X-Ray is OFF, rear/occluded projected points must not receive
        # proportional influence even if they are inside the radius.
        if not self._xray:
            visible = set(
                self._secondary_front_filter(
                    range(len(base_points)),
                    base_points
                )
            )

            if i not in visible:
                return 0.0

        p = base_points[i]

        d = min(
            (p - base_points[j]).length
            for j in self._selected
        )

        if d >= self._prop_radius_px:
            return 0.0

        return _smooth_falloff(
            d / self._prop_radius_px
        )

    def _secondary_apply_drag(self, context, mx, my):
        if not self._secondary_dragging:
            return

        start_plane = self._secondary_mouse_to_plane(
            self._secondary_drag_start_mouse.x,
            self._secondary_drag_start_mouse.y
        )
        now_plane = self._secondary_mouse_to_plane(mx, my)
        delta = now_plane - start_plane

        base_points = [
            p.copy()
            for p in self._secondary_frozen_points
        ]

        mw = self._obj.matrix_world
        inv = mw.inverted()

        for i, vi in enumerate(self._ordered):
            w = self._secondary_weight(i, base_points)
            if w <= 0.0:
                continue

            base_world = self._secondary_drag_base_world[i]
            base_plane = self._secondary_world_to_plane(
                base_world
            )

            world = self._secondary_plane_to_world(
                base_plane + delta * w,
                base_world
            )

            if self._lock_x:
                world.x = base_world.x
            if self._lock_y:
                world.y = base_world.y
            if self._lock_z:
                world.z = base_world.z

            world = self._hp_clip_world(base_world, world)
            self._bm.verts[vi].co = inv @ world

        self._hp_apply_direct_follow()
        bmesh.update_edit_mesh(
            self._obj.data,
            loop_triangles=False,
            destructive=False
        )

        self._sync_primary_from_mesh()
        context.area.tag_redraw()

    def _secondary_finish_drag(self, context):
        self._secondary_dragging = False
        self._direct_follow_state = None
        self._secondary_release_map()
        self._sync_primary_from_mesh()
        context.area.tag_redraw()

    def _secondary_apply_box_select(self):
        if (
            self._secondary_box_start is None
            or self._secondary_box_end is None
        ):
            return

        x1 = min(
            self._secondary_box_start.x,
            self._secondary_box_end.x
        )
        x2 = max(
            self._secondary_box_start.x,
            self._secondary_box_end.x
        )
        y1 = min(
            self._secondary_box_start.y,
            self._secondary_box_end.y
        )
        y2 = max(
            self._secondary_box_start.y,
            self._secondary_box_end.y
        )

        points = self._secondary_panel_points()

        picked_list = [
            i for i, p in enumerate(points)
            if x1 <= p.x <= x2
            and y1 <= p.y <= y2
        ]

        picked = set(
            self._secondary_front_filter(
                picked_list,
                points
            )
        )

        if self._secondary_box_additive:
            self._selected |= picked
        else:
            self._selected = picked

    def _secondary_axis_vector(self):
        if self._secondary_transform_axis == 'X':
            return Vector((1.0, 0.0, 0.0))
        if self._secondary_transform_axis == 'Y':
            return Vector((0.0, 1.0, 0.0))
        if self._secondary_transform_axis == 'Z':
            return Vector((0.0, 0.0, 1.0))
        return None

    def _secondary_axis_screen_delta(self, axis, mouse_delta):
        if not self._secondary_map_frozen:
            return 0.0

        scale = max(self._secondary_frozen_scale, 1e-12)
        sy = -1.0 if self._secondary_flip_y else 1.0
        origin = self._secondary_world_to_plane(Vector((0.0, 0.0, 0.0)))
        projected = self._secondary_world_to_plane(axis) - origin
        screen_axis = Vector((
            projected.x * scale,
            projected.y * scale * sy
        ))

        length_sq = screen_axis.length_squared
        if length_sq <= 1e-12:
            return 0.0

        return mouse_delta.dot(screen_axis) / length_sq

    def _secondary_begin_transform(self, mode, mx, my):
        if not self._selected:
            return

        follow = self._hp_capture_direct_follow()
        if not self._push_history(follow):
            return
        self._direct_follow_state = follow

        if not self._secondary_capture_map():
            self._direct_follow_state = None
            return

        self._secondary_transform_mode = mode
        self._secondary_transform_axis = None
        self._secondary_transform_start_mouse = Vector((mx, my))

        mw = self._obj.matrix_world

        self._secondary_transform_base_world = [
            mw @ self._bm.verts[vi].co.copy()
            for vi in self._ordered
        ]

        selected_world = [
            self._secondary_transform_base_world[i]
            for i in sorted(self._selected)
        ]

        self._secondary_transform_pivot_world = (
            sum(
                selected_world,
                Vector((0.0, 0.0, 0.0))
            )
            / len(selected_world)
        )

    def _secondary_rebase_transform(self, mx, my):
        if (
            self._secondary_transform_mode is None
            or not self._selected
        ):
            return False

        self._secondary_release_map()

        if not self._secondary_capture_map():
            return False

        self._direct_follow_state = self._hp_capture_direct_follow()
        self._secondary_transform_start_mouse = Vector((mx, my))

        mw = self._obj.matrix_world

        self._secondary_transform_base_world = [
            mw @ self._bm.verts[vi].co.copy()
            for vi in self._ordered
        ]

        selected_world = [
            self._secondary_transform_base_world[i]
            for i in sorted(self._selected)
        ]

        self._secondary_transform_pivot_world = (
            sum(
                selected_world,
                Vector((0.0, 0.0, 0.0))
            )
            / len(selected_world)
        )

        return True

    def _secondary_apply_transform(self, context, mx, my):
        if self._secondary_transform_mode is None:
            return

        mouse_delta = (
            Vector((mx, my))
            - self._secondary_transform_start_mouse
        )

        start_plane = self._secondary_mouse_to_plane(
            self._secondary_transform_start_mouse.x,
            self._secondary_transform_start_mouse.y
        )
        now_plane = self._secondary_mouse_to_plane(
            mx,
            my
        )
        plane_delta = now_plane - start_plane

        scale_factor = max(
            0.01,
            1.0 + mouse_delta.x / 140.0
        )
        angle = mouse_delta.x * 0.01

        axis = self._secondary_axis_vector()

        mw = self._obj.matrix_world
        inv = mw.inverted()

        pivot_plane = self._secondary_world_to_plane(
            self._secondary_transform_pivot_world
        )

        base_points = [
            p.copy()
            for p in self._secondary_frozen_points
        ]

        for i, vi in enumerate(self._ordered):
            w = self._secondary_weight(
                i,
                base_points
            )

            if w <= 0.0:
                continue

            base = self._secondary_transform_base_world[i]
            full_target = base.copy()

            if self._secondary_transform_mode == 'G':
                if axis is None:
                    base_plane = self._secondary_world_to_plane(
                        base
                    )
                    full_target = self._secondary_plane_to_world(
                        base_plane + plane_delta,
                        base
                    )
                else:
                    full_target = (
                        base
                        + axis
                        * self._secondary_axis_screen_delta(
                            axis,
                            mouse_delta
                        )
                    )

            elif self._secondary_transform_mode == 'S':
                if axis is None:
                    base_plane = self._secondary_world_to_plane(
                        base
                    )
                    target_plane = (
                        pivot_plane
                        + (base_plane - pivot_plane)
                        * scale_factor
                    )

                    full_target = self._secondary_plane_to_world(
                        target_plane,
                        base
                    )
                else:
                    rel = (
                        base
                        - self._secondary_transform_pivot_world
                    )
                    parallel = axis * rel.dot(axis)
                    perpendicular = rel - parallel

                    full_target = (
                        self._secondary_transform_pivot_world
                        + perpendicular
                        + parallel * scale_factor
                    )

            elif self._secondary_transform_mode == 'R':
                if axis is None:
                    base_plane = self._secondary_world_to_plane(
                        base
                    )
                    rel2 = base_plane - pivot_plane

                    c = cos(angle)
                    s = sin(angle)

                    rotated = Vector((
                        rel2.x * c - rel2.y * s,
                        rel2.x * s + rel2.y * c
                    ))

                    full_target = self._secondary_plane_to_world(
                        pivot_plane + rotated,
                        base
                    )
                else:
                    q = Quaternion(axis, angle)
                    rel = (
                        base
                        - self._secondary_transform_pivot_world
                    )

                    full_target = (
                        self._secondary_transform_pivot_world
                        + q @ rel
                    )

            target = base.lerp(
                full_target,
                w
            )

            if self._lock_x:
                target.x = base.x
            if self._lock_y:
                target.y = base.y
            if self._lock_z:
                target.z = base.z

            target = self._hp_clip_world(base, target)
            self._bm.verts[vi].co = inv @ target

        self._hp_apply_direct_follow()
        bmesh.update_edit_mesh(
            self._obj.data,
            loop_triangles=False,
            destructive=False
        )

        self._sync_primary_from_mesh()
        context.area.tag_redraw()

    def _secondary_confirm_transform(self, context):
        self._secondary_transform_mode = None
        self._direct_follow_state = None
        self._secondary_transform_axis = None
        self._secondary_release_map()
        self._sync_primary_from_mesh()
        context.area.tag_redraw()

    def _secondary_cancel_transform(self, context):
        self._secondary_transform_mode = None
        self._direct_follow_state = None
        self._secondary_transform_axis = None
        self._secondary_release_map()
        self._local_undo(context)

    def _secondary_pen_selected_order(self):
        n = len(self._ordered)
        if n < 2:
            return None, False

        sel = set(self._selected)

        if not sel:
            return list(range(n)), bool(self._closed)

        if len(sel) < 2:
            return None, False

        if not self._closed:
            lo = min(sel)
            hi = max(sel)

            if sel != set(range(lo, hi + 1)):
                return None, False

            return list(range(lo, hi + 1)), False

        if len(sel) == n:
            return list(range(n)), True

        starts = [
            i for i in sel
            if ((i - 1) % n) not in sel
        ]

        if len(starts) != 1:
            return None, False

        order = []
        i = starts[0]

        while i in sel:
            order.append(i)
            i = (i + 1) % n

            if len(order) > n:
                break

        if len(order) != len(sel):
            return None, False

        return order, False

    def _secondary_apply_pen(self, context):
        if len(self._secondary_pen_stroke) < 2:
            return

        order, full_loop = self._secondary_pen_selected_order()

        if order is None or len(order) < 2:
            self.report(
                {'WARNING'},
                "View B Pen: select one connected range"
            )
            return

        explicit_closed = (
            len(self._secondary_pen_stroke) >= 3
            and (
                self._secondary_pen_stroke[-1]
                - self._secondary_pen_stroke[0]
            ).length <= 20.0
        )

        if full_loop and not explicit_closed:
            inferred = self._infer_closed_arc_order(
                self._secondary_pen_stroke,
                self._secondary_panel_points(),
            )
            if inferred is None or len(inferred) < 2:
                self.report({'WARNING'}, "View B: closed-loop arc could not be inferred")
                return
            order = inferred
            full_loop = False
            _hp_log(f"pen B inferred closed arc points={len(order)}")

        stroke_closed = bool(
            full_loop and explicit_closed
        )

        smooth = _stabilize_2d(
            self._secondary_pen_stroke,
            level=self._secondary_pen_stabilizer,
            closed=stroke_closed
        )

        current_points = self._secondary_panel_points()
        target_panel = [
            current_points[i]
            for i in order
        ]

        smooth = self._prepare_pen_stroke(
            smooth,
            target_panel,
            stroke_closed
        )

        t_values = _path_t_values(
            target_panel,
            closed=stroke_closed
        )

        samples = _sample_polyline_2d(
            smooth,
            t_values,
            closed=stroke_closed
        )

        panel_points = self._secondary_panel_points()
        sample_map = {i: p.copy() for i, p in zip(order, samples)}
        selected = set(order)
        blend_settings = getattr(self, '_hp_settings', {})
        blend_radius = max(0, min(8, int(blend_settings.get('blend_radius', 2))))
        blend_strength = max(0.0, min(1.0, float(blend_settings.get('blend_strength', 0.65))))
        for endpoint in (order[0], order[-1]):
            delta = sample_map[endpoint] - panel_points[endpoint]
            direction = 1 if endpoint == order[0] else -1
            for step in range(1, blend_radius + 1):
                j = (endpoint + direction * step) % len(panel_points)
                if j in selected:
                    continue
                weight = blend_strength * (1.0 - step / (blend_radius + 1.0)) ** 2
                sample_map[j] = panel_points[j] + delta * weight
        if len(sample_map) > len(order):
            order = list(sample_map.keys())
            samples = [sample_map[i] for i in order]

        self._push_history()

        mw = self._obj.matrix_world
        inv = mw.inverted()

        base_world = [
            mw @ self._bm.verts[vi].co.copy()
            for vi in self._ordered
        ]

        for i, panel_p in zip(order, samples):
            plane_co = self._secondary_mouse_to_plane(
                panel_p.x,
                panel_p.y
            )

            world = self._secondary_plane_to_world(
                plane_co,
                base_world[i]
            )

            world = self._apply_world_locks(base_world[i], world)

            self._bm.verts[self._ordered[i]].co = inv @ world

        bmesh.update_edit_mesh(
            self._obj.data,
            loop_triangles=False,
            destructive=False
        )

        self._clear_all_pen_state()

        self._sync_primary_from_mesh()
        context.area.tag_redraw()

    def _primary_depth_value(self, index):
        # self._depths is already measured along View A's current camera/depth
        # direction in both SECTION and named-view modes.
        if index < 0 or index >= len(self._depths):
            return 0.0

        return self._depths[index]

    def _primary_front_filter(self, indices, points=None):
        indices = list(indices)

        if self._xray or len(indices) <= 1:
            return indices

        if points is None:
            points = self._panel_points

        ordered = sorted(
            indices,
            key=self._primary_depth_value,
            reverse=True
        )

        visible = []
        radius = self._primary_occlusion_px

        for i in ordered:
            p = points[i]

            if any(
                (p - points[j]).length <= radius
                for j in visible
            ):
                continue

            visible.append(i)

        return visible

    def _pick_point(self, x, y):
        if not self._panel_points:
            return None

        mouse = Vector((x, y))

        candidates = [
            i
            for i, p in enumerate(self._panel_points)
            if (p - mouse).length <= 14.0
        ]

        candidates = self._primary_front_filter(
            candidates,
            self._panel_points
        )

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda i: (
                self._panel_points[i] - mouse
            ).length
        )

    def _set_panel_stage(self, stage):
        self._panel_stage = max(0, min(len(self._panel_sizes) - 1, int(stage)))
        self._panel_w, self._panel_h = self._panel_sizes[self._panel_stage]
        self._recompute_fit_scale()
        self._refresh_panel_points()
        self._secondary_invalidate_view()

    def _reset_panel_size(self):
        self._set_panel_stage(len(self._panel_sizes) - 1)

    def _begin_resize_mode(self, mx):
        self._resize_mode = True
        self._resize_start_x = mx
        self._resize_start_stage = self._panel_stage

    def _update_resize_mode(self, mx):
        step_px = 90.0
        delta_stage = int(round((mx - self._resize_start_x) / step_px))
        self._set_panel_stage(self._resize_start_stage + delta_stage)

    def _confirm_resize_mode(self):
        self._resize_mode = False

    def _cancel_resize_mode(self):
        self._set_panel_stage(self._resize_start_stage)
        self._resize_mode = False

    def _start_drag(self, context, mx, my):
        if not self._ensure_live_bmesh(context):
            self._dragging = False
            return False

        follow = self._hp_capture_direct_follow()
        if not self._push_history(follow):
            self._dragging = False
            return False
        self._direct_follow_state = follow

        self._dragging = True
        self._drag_start_sec = self._mouse_to_section(mx, my)

        self._drag_base_coords = [
            p.copy() for p in self._coords
        ]

        self._drag_base_depths = list(self._depths)

        self._drag_base_panel = [
            p.copy() for p in self._panel_points
        ]

        mw = self._obj.matrix_world

        try:
            self._drag_base_world = [
                mw @ self._bm.verts[vi].co.copy()
                for vi in self._ordered
            ]
        except (ReferenceError, RuntimeError, IndexError):
            self._dragging = False
            self._direct_follow_state = None
            return False

        return True

    def _weight_for_index(self, i):
        if i in self._selected:
            return 1.0

        if not self._prop_enabled or not self._selected:
            return 0.0

        if not self._xray:
            visible = set(
                self._primary_front_filter(
                    range(len(self._drag_base_panel)),
                    self._drag_base_panel
                )
            )
            if i not in visible:
                return 0.0

        p = self._drag_base_panel[i]

        d = min(
            (p - self._drag_base_panel[j]).length
            for j in self._selected
        )

        if d >= self._prop_radius_px:
            return 0.0

        return _smooth_falloff(
            d / self._prop_radius_px
        )

    def _apply_drag(self, context, mx, my):
        if not self._dragging or not self._selected:
            return

        if not self._bmesh_ready():
            self._dragging = False
            self._direct_follow_state = None
            context.area.tag_redraw()
            return

        now = self._mouse_to_section(mx, my)
        delta = now - self._drag_start_sec

        mw = self._obj.matrix_world
        inv = mw.inverted()

        new_coords = []
        new_depths = []

        for i, vi in enumerate(self._ordered):
            w = self._weight_for_index(i)

            sec = (
                self._drag_base_coords[i]
                + delta * w
            )

            co = self._primary_local_from_values(
                sec,
                self._drag_base_depths[i]
            )

            world = mw @ co
            base_world = self._drag_base_world[i]

            if self._lock_x:
                world.x = base_world.x
            if self._lock_y:
                world.y = base_world.y
            if self._lock_z:
                world.z = base_world.z

            world = self._hp_clip_world(base_world, world)
            local = inv @ world
            self._bm.verts[vi].co = local

            sec_now, depth_now = (
                self._primary_values_from_local(local)
            )

            new_coords.append(sec_now)
            new_depths.append(depth_now)

        self._coords = new_coords
        self._depths = new_depths

        self._hp_apply_direct_follow()
        bmesh.update_edit_mesh(
            self._obj.data,
            loop_triangles=False,
            destructive=False
        )

        self._refresh_panel_points()
        context.area.tag_redraw()

    def _apply_box_select(self):
        if self._box_start is None or self._box_end is None:
            return

        x1 = min(self._box_start.x, self._box_end.x)
        x2 = max(self._box_start.x, self._box_end.x)

        y1 = min(self._box_start.y, self._box_end.y)
        y2 = max(self._box_start.y, self._box_end.y)

        picked = {
            i for i, p in enumerate(self._panel_points)
            if x1 <= p.x <= x2 and y1 <= p.y <= y2
        }

        if not self._xray:
            picked = set(
                self._primary_front_filter(
                    picked,
                    self._panel_points
                )
            )

        if self._box_additive:
            self._selected |= picked
        else:
            self._selected = picked

    def _push_history(self, follow_state=None):
        if not self._bmesh_ready():
            return False

        if follow_state is not None:
            self._history.append({
                'hp_pen': True, 'base': follow_state['base'],
                'counts': follow_state['counts'], 'data': follow_state['data'],
            })
            if len(self._history) > self._history_limit:
                self._history.pop(0)
            return True

        try:
            snapshot = [
                self._bm.verts[vi].co.copy()
                for vi in self._ordered
            ]
        except (ReferenceError, RuntimeError, IndexError):
            return False

        if self._history:
            prev = self._history[-1]
            if not isinstance(prev, dict) and len(prev) == len(snapshot):
                same = all(
                    (a - b).length < 1e-10
                    for a, b in zip(prev, snapshot)
                )
                if same:
                    return True

        self._history.append(snapshot)

        if len(self._history) > self._history_limit:
            self._history.pop(0)

        return True

    def _topology_undo(self, context):
        if self._topology_undo_steps <= 0:
            return False

        try:
            bpy.ops.ed.undo()
        except Exception as exc:
            self.report(
                {'WARNING'},
                f"Topology Undo failed: {exc}"
            )
            return False

        self._topology_undo_steps -= 1
        self._history.clear()

        self._topology_pending_selected = (
            self._topology_selection_history.pop()
            if self._topology_selection_history
            else set()
        )

        # bpy.ops.ed.undo replaces mesh data/BMesh. Reacquire on TIMER;
        # touching self._bm here would risk stale references.
        self._topology_rebuild_pending = True
        context.area.tag_redraw()
        return True

    def _local_undo(self, context):
        if not self._history:
            if self._topology_undo(context):
                return True

            return False

        snapshot = self._history.pop()

        if isinstance(snapshot, dict) and snapshot.get('hp_pen'):
            if not self._hp_restore(snapshot):
                self.report({'WARNING'}, '接続が変わったためペンのUndoを中止しました')
            context.area.tag_redraw()
            return True

        if len(snapshot) != len(self._ordered):
            self.report({'WARNING'}, "HP Section: topology changed; local undo cancelled")
            return False

        for vi, co in zip(self._ordered, snapshot):
            self._bm.verts[vi].co = co.copy()

        bmesh.update_edit_mesh(
            self._obj.data,
            loop_triangles=False,
            destructive=False
        )

        chain = _selected_chain(context)
        if chain is not None:
            self._rebuild(context, chain)

        context.area.tag_redraw()
        return True

    def _apply_world_locks(self, old_world, new_world):
        world = new_world.copy()

        if self._lock_x:
            world.x = old_world.x
        if self._lock_y:
            world.y = old_world.y
        if self._lock_z:
            world.z = old_world.z

        return self._hp_clip_world(old_world, world)

    def _hp_clip_world(self, old_world, new_world):
        """Honor enabled Mirror Clipping for direct BMesh coordinate edits."""
        obj = self._obj
        if obj is None:
            return new_world
        inv = obj.matrix_world.inverted()
        old_local = inv @ old_world
        new_local = inv @ new_world
        changed = False
        for modifier in obj.modifiers:
            if (modifier.type != 'MIRROR' or not modifier.use_clip
                    or not modifier.show_viewport or not modifier.show_in_editmode):
                continue
            mirror_obj = modifier.mirror_object
            frame = (mirror_obj.matrix_world.inverted() @ obj.matrix_world
                     if mirror_obj is not None else None)
            base = frame @ old_local if frame is not None else old_local
            target = frame @ new_local if frame is not None else new_local.copy()
            threshold = max(1e-8, float(modifier.merge_threshold))
            for axis, enabled in enumerate(modifier.use_axis):
                if not enabled:
                    continue
                start, end = base[axis], target[axis]
                if abs(start) <= threshold or abs(end) <= threshold or start * end < 0.0:
                    target[axis] = 0.0
                    changed = True
            if frame is not None:
                new_local = frame.inverted() @ target
            else:
                new_local = target
        return obj.matrix_world @ new_local if changed else new_world

    def _begin_smooth_drag(
        self,
        context,
        target,
        mx,
        my
    ):
        if target == 'B':
            order, full_loop = (
                self._secondary_pen_selected_order()
            )
        else:
            order, full_loop = (
                self._pen_selected_order()
            )

        if order is None or len(order) < 3:
            self.report(
                {'WARNING'},
                "Smooth: select one connected range with at least 3 points"
            )
            _hp_smooth_debug(
                f"BEGIN rejected target={target} order={order}"
            )
            return False

        history_len = len(self._history)
        follow = self._hp_capture_direct_follow()
        if not self._push_history(follow):
            return False
        self._smooth_history_len = history_len
        self._direct_follow_state = follow
        _hp_log(f"vertex brush begin target={target} selected={len(order)}")

        self._smooth_dragging = True
        self._brush_mode = 'VERTEX_B' if target == 'B' else 'VERTEX_A'
        self._smooth_target = target
        self._smooth_start_x = float(mx)
        self._smooth_amount = 0.0
        self._smooth_hud_x = float(mx)
        self._smooth_hud_y = float(my)
        self._smooth_brush_prev_mouse = Vector((mx, my))
        self._smooth_order = list(order)
        self._smooth_closed = bool(full_loop)

        mw = self._obj.matrix_world

        self._smooth_base_local = [
            self._bm.verts[vi].co.copy()
            for vi in self._ordered
        ]

        self._smooth_base_world = [
            mw @ co
            for co in self._smooth_base_local
        ]

        if target == 'B':
            if not self._secondary_capture_map():
                self._smooth_dragging = False
                self._brush_mode = None
                self._direct_follow_state = None
                if len(self._history) > history_len:
                    self._history.pop()
                return False

            self._smooth_base_panel = [
                p.copy()
                for p in self._secondary_panel_points()
            ]
        else:
            self._smooth_base_panel = [
                p.copy()
                for p in self._panel_points
            ]

            self._smooth_base_depths = list(
                self._depths
            )

        _hp_smooth_debug(
            "BEGIN "
            f"target={target} "
            f"selected={len(order)} "
            f"closed={self._smooth_closed} "
            f"indices={list(order)}"
        )

        context.area.tag_redraw()
        return True

    def _update_smooth_drag(
        self,
        context,
        mx,
        my
    ):
        if not self._smooth_dragging:
            return

        self._smooth_amount = self._smooth_brush_strength
        self._smooth_hud_x = float(mx)
        self._smooth_hud_y = float(my)
        motion = Vector((mx, my)) - self._smooth_brush_prev_mouse
        self._smooth_brush_prev_mouse = Vector((mx, my))

        if self._smooth_target == 'B':
            current = self._secondary_panel_points()
        else:
            current = [p.copy() for p in self._panel_points]

        changed = _hp_brush_repel_2d(
            current,
            Vector((mx, my)),
            motion,
            radius=self._smooth_brush_radius,
            strength=self._smooth_brush_strength,
            closed=self._smooth_closed and len(self._smooth_order) == len(self._ordered),
        )

        mw = self._obj.matrix_world
        inv = mw.inverted()
        max_delta_px = 0.0

        for original_i in self._smooth_order:
            before = current[original_i]
            panel_p = changed[original_i]
            max_delta_px = max(max_delta_px, (panel_p - before).length)

            if self._smooth_target == 'B':
                base_world = mw @ self._bm.verts[self._ordered[original_i]].co.copy()
                plane = self._secondary_mouse_to_plane(panel_p.x, panel_p.y)
                world = self._apply_world_locks(
                    base_world,
                    self._secondary_plane_to_world(plane, base_world),
                )
            else:
                sec = self._mouse_to_section(panel_p.x, panel_p.y)
                local = self._primary_local_from_values(sec, self._depths[original_i])
                world = self._apply_world_locks(
                    mw @ self._bm.verts[self._ordered[original_i]].co.copy(),
                    mw @ local,
                )

            self._bm.verts[self._ordered[original_i]].co = inv @ world

        self._hp_apply_direct_follow()
        bmesh.update_edit_mesh(
            self._obj.data,
            loop_triangles=False,
            destructive=False
        )

        self._sync_primary_from_mesh()

        self._smooth_last_delta_px = max_delta_px

        context.area.tag_redraw()

    def _finish_smooth_drag(
        self,
        context,
        cancel=False
    ):
        if not self._smooth_dragging:
            return

        if cancel:
            follow = getattr(self, '_direct_follow_state', None)
            if follow is not None:
                for vi, co in follow['base'].items():
                    self._bm.verts[vi].co = co.copy()
            else:
                for vi, co in zip(self._ordered, self._smooth_base_local):
                    self._bm.verts[vi].co = co.copy()

            bmesh.update_edit_mesh(
                self._obj.data,
                loop_triangles=False,
                destructive=False
            )

            # Remove the no-longer-needed undo snapshot made at begin.
            if len(self._history) > self._smooth_history_len:
                self._history.pop()

        if self._smooth_target == 'B':
            self._secondary_release_map()

        self._sync_primary_from_mesh()

        _hp_smooth_debug(
            "END "
            f"target={self._smooth_target} "
            f"cancel={cancel} "
            f"amount={self._smooth_amount:.3f} "
            f"max_delta_px={getattr(self, '_smooth_last_delta_px', 0.0):.3f}"
        )

        self._smooth_dragging = False
        self._direct_follow_state = None
        self._brush_mode = None
        self._smooth_target = None
        self._smooth_order = []
        self._smooth_base_panel = []
        self._smooth_base_local = []
        self._smooth_base_world = []
        self._smooth_base_depths = []

        context.area.tag_redraw()

    def _shape_tool_name(self, tool):
        return {
            'RELAX': 'Relax',
            'EQUAL': 'Equal Spacing',
            'LINE': 'Line',
            'ELLIPSE': 'Ellipse',
            'SYMMETRY': 'Symmetry',
        }.get(tool, tool)

    def _apply_primary_shape_tool(
        self,
        context,
        tool
    ):
        order, full_loop = (
            self._pen_selected_order()
        )

        if order is None or len(order) < 2:
            self.report(
                {'WARNING'},
                "Shape Tool: select one connected range"
            )
            return

        targets = _shape_targets_2d(
            self._panel_points,
            order,
            tool,
            closed=bool(full_loop)
        )

        if not targets:
            self.report(
                {'INFO'},
                f"{self._shape_tool_name(tool)}: current selection is not suitable"
            )
            return

        section_targets = {
            i: self._mouse_to_section(
                p.x,
                p.y
            )
            for i, p in targets.items()
        }

        self._apply_section_targets(
            context,
            section_targets
        )

    def _apply_secondary_shape_tool(
        self,
        context,
        tool
    ):
        order, full_loop = (
            self._secondary_pen_selected_order()
        )

        if order is None or len(order) < 2:
            self.report(
                {'WARNING'},
                "View B Shape Tool: select one connected range"
            )
            return

        points = self._secondary_panel_points()

        targets = _shape_targets_2d(
            points,
            order,
            tool,
            closed=bool(full_loop)
        )

        if not targets:
            self.report(
                {'INFO'},
                f"{self._shape_tool_name(tool)}: current selection is not suitable"
            )
            return

        self._push_history()

        mw = self._obj.matrix_world
        inv = mw.inverted()

        base_world = [
            mw @ self._bm.verts[vi].co.copy()
            for vi in self._ordered
        ]

        for i, panel_p in targets.items():
            plane = self._secondary_mouse_to_plane(
                panel_p.x,
                panel_p.y
            )

            new_world = (
                self._secondary_plane_to_world(
                    plane,
                    base_world[i]
                )
            )

            new_world = self._apply_world_locks(base_world[i], new_world)

            self._bm.verts[
                self._ordered[i]
            ].co = inv @ new_world

        bmesh.update_edit_mesh(
            self._obj.data,
            loop_triangles=False,
            destructive=False
        )

        self._sync_primary_from_mesh()
        context.area.tag_redraw()

    def _chain_vertex_refs(self):
        if self._bm is None:
            return []

        self._bm.verts.ensure_lookup_table()

        return [
            self._bm.verts[vi]
            for vi in self._ordered
        ]

    def _reselect_topology_chain(
        self,
        chain_verts,
        closed
    ):
        bm = self._bm

        if bm is None:
            return False

        chain_verts = [
            v
            for v in chain_verts
            if getattr(v, "is_valid", False)
        ]

        min_count = 3 if closed else 2

        if len(chain_verts) < min_count:
            return False

        for e in bm.edges:
            e.select_set(False)

        for v in bm.verts:
            v.select_set(False)

        pair_count = (
            len(chain_verts)
            if closed
            else len(chain_verts) - 1
        )

        selected_edges = []

        for i in range(pair_count):
            a = chain_verts[i]
            b = chain_verts[
                (i + 1) % len(chain_verts)
            ]

            edge = bm.edges.get((a, b))

            if edge is None:
                return False

            edge.select_set(True)
            a.select_set(True)
            b.select_set(True)
            selected_edges.append(edge)

        return bool(selected_edges)

    def _topology_undo_push(self):
        try:
            bpy.ops.ed.undo_push(
                message="HP Mini Topology"
            )
        except Exception:
            pass

    def _finish_after_topology(
        self,
        context,
        selected_indices=None
    ):
        bmesh.update_edit_mesh(
            self._obj.data,
            loop_triangles=False,
            destructive=True
        )

        self._history.clear()
        self._topology_undo_steps += 1
        self._topology_rebuild_pending = False

        self._clear_all_pen_state()
        self._dragging = False
        self._box_dragging = False
        self._transform_mode = None
        self._secondary_dragging = False
        self._secondary_box_dragging = False
        self._secondary_transform_mode = None

        chain = _selected_chain(context)

        if chain is None:
            self._topology_pending_selected = (
                set(selected_indices)
                if selected_indices is not None
                else set()
            )
            self._topology_rebuild_pending = True
        else:
            self._rebuild(
                context,
                chain
            )

            desired = (
                set(selected_indices)
                if selected_indices is not None
                else set()
            )

            self._selected = {
                i
                for i in desired
                if 0 <= i < len(self._ordered)
            }

        context.area.tag_redraw()

    def _apply_topology_tool(
        self,
        context,
        tool
    ):
        if (
            self._obj is None
            or self._bm is None
        ):
            return False

        bm = self._bm
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()

        chain = self._chain_vertex_refs()

        if len(chain) < 2:
            return False

        selected = set(self._selected)

        if not selected:
            self.report(
                {'WARNING'},
                "細分化: 小窓で点を選択してください"
            )
            return False

        n = len(chain)

        if tool == 'SUBDIVIDE':
            pair_count = (
                n
                if self._closed
                else n - 1
            )

            target_slots = []

            for i in range(pair_count):
                j = (i + 1) % n

                if (
                    i in selected
                    and j in selected
                ):
                    target_slots.append(i)

            if not target_slots:
                self.report(
                    {'WARNING'},
                    "細分化: 隣接する2点以上を選択してください"
                )
                return False

            self._topology_selection_history.append(
                set(selected)
            )
            self._topology_undo_push()

            desired_selected_refs = [
                chain[i]
                for i in sorted(selected)
                if 0 <= i < n
            ]

            # Split each selected chain edge individually.
            # This avoids keeping stale BMVert references after a bulk
            # subdivision operation.
            new_chain = []

            for i in range(pair_count):
                a = chain[i]
                b = chain[
                    (i + 1) % n
                ]

                if (
                    not a.is_valid
                    or not b.is_valid
                ):
                    self.report(
                        {'WARNING'},
                        "細分化中に元の点参照が無効になりました"
                    )
                    return False

                new_chain.append(a)

                if i not in target_slots:
                    continue

                edge = bm.edges.get(
                    (a, b)
                )

                if edge is None or not edge.is_valid:
                    self.report(
                        {'WARNING'},
                        "細分化対象のエッジを取得できませんでした"
                    )
                    return False

                new_edge, new_vert = (
                    bmesh.utils.edge_split(
                        edge,
                        a,
                        0.5
                    )
                )

                if (
                    new_vert is None
                    or not new_vert.is_valid
                ):
                    self.report(
                        {'WARNING'},
                        "細分化点の生成に失敗しました"
                    )
                    return False

                new_chain.append(
                    new_vert
                )
                desired_selected_refs.append(
                    new_vert
                )

            if not self._closed:
                last = chain[-1]

                if not last.is_valid:
                    self.report(
                        {'WARNING'},
                        "細分化後の終点が無効です"
                    )
                    return False

                new_chain.append(
                    last
                )

            bm.verts.index_update()
            bm.edges.index_update()
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()

            if not self._reselect_topology_chain(
                new_chain,
                self._closed
            ):
                self.report(
                    {'WARNING'},
                    "細分化後のチェーン再選択に失敗しました"
                )
                return False

            desired_selected_indices = {
                i
                for i, v in enumerate(new_chain)
                if any(
                    v is chosen
                    for chosen in desired_selected_refs
                )
            }

            self._finish_after_topology(
                context,
                desired_selected_indices
            )
            return True

        if tool == 'DISSOLVE':
            victim_indices = set(selected)

            if not self._closed:
                victim_indices.discard(0)
                victim_indices.discard(n - 1)

            survivors = [
                v
                for i, v in enumerate(chain)
                if i not in victim_indices
            ]

            min_survivors = (
                3 if self._closed else 2
            )

            if (
                not victim_indices
                or len(survivors) < min_survivors
            ):
                self.report(
                    {'WARNING'},
                    "溶解: 開いた線の両端は残し、必要な点数を確保してください"
                )
                return False

            victims = [
                chain[i]
                for i in sorted(victim_indices)
            ]

            desired_selected_refs = [
                chain[i]
                for i in sorted(selected)
                if (
                    0 <= i < n
                    and i not in victim_indices
                )
            ]

            self._topology_selection_history.append(
                set(selected)
            )
            self._topology_undo_push()

            bmesh.ops.dissolve_verts(
                bm,
                verts=victims,
                use_face_split=False,
                use_boundary_tear=False
            )

            if not self._reselect_topology_chain(
                survivors,
                self._closed
            ):
                self.report(
                    {'WARNING'},
                    "溶解後のチェーン再選択に失敗しました"
                )
                return False

            desired_selected_indices = {
                i
                for i, v in enumerate(survivors)
                if any(
                    v is chosen
                    for chosen in desired_selected_refs
                )
            }

            self._finish_after_topology(
                context,
                desired_selected_indices
            )
            return True

        if tool == 'MERGE':
            if len(selected) != 2:
                self.report(
                    {'WARNING'},
                    "マージ: 隣接する2点だけを選択してください"
                )
                return False

            a_i, b_i = sorted(selected)

            wrap_pair = bool(
                self._closed
                and a_i == 0
                and b_i == n - 1
            )

            adjacent = (
                b_i == a_i + 1
                or wrap_pair
            )

            if not adjacent:
                self.report(
                    {'WARNING'},
                    "マージ: 2点はチェーン上で隣接している必要があります"
                )
                return False

            if wrap_pair:
                keep_i = 0
                kill_i = n - 1
            else:
                keep_i = a_i
                kill_i = b_i

            keep = chain[keep_i]
            kill = chain[kill_i]

            self._topology_selection_history.append(
                set(selected)
            )
            self._topology_undo_push()

            mw = self._obj.matrix_world
            keep.co = mw.inverted() @ self._hp_clip_world(
                mw @ keep.co.copy(),
                mw @ ((keep.co + kill.co) * 0.5),
            )

            bmesh.ops.weld_verts(
                bm,
                targetmap={
                    kill: keep
                }
            )

            survivors = [
                v
                for i, v in enumerate(chain)
                if i != kill_i
            ]

            if not self._reselect_topology_chain(
                survivors,
                self._closed
            ):
                self.report(
                    {'WARNING'},
                    "マージ後のチェーン再選択に失敗しました"
                )
                return False

            desired_selected_indices = {
                i
                for i, v in enumerate(survivors)
                if v is keep
            }

            self._finish_after_topology(
                context,
                desired_selected_indices
            )
            return True

        return False

    def _apply_section_targets(self, context, target_by_index):
        if not target_by_index:
            return

        self._push_history()

        mw = self._obj.matrix_world
        inv = mw.inverted()

        old_worlds = [
            mw @ self._bm.verts[vi].co.copy()
            for vi in self._ordered
        ]

        for i, sec in target_by_index.items():
            if i < 0 or i >= len(self._ordered):
                continue

            local = self._primary_local_from_values(
                sec,
                self._depths[i]
            )

            new_world = mw @ local
            new_world = self._apply_world_locks(
                old_worlds[i],
                new_world
            )

            self._bm.verts[self._ordered[i]].co = inv @ new_world

        bmesh.update_edit_mesh(
            self._obj.data,
            loop_triangles=False,
            destructive=False
        )

        chain = _selected_chain(context)
        if chain is not None:
            self._rebuild(context, chain)

        context.area.tag_redraw()

    def _pen_selected_order(self):
        """
        Returns (order, full_loop).

        Pen semantics:
        - If mini-window points are selected, fit ONLY that connected selected range.
        - If nothing is selected:
            open source chain -> whole chain
            closed source loop -> whole loop
        - Disconnected mini selections are rejected.
        """
        n = len(self._panel_points)
        if n < 2:
            return None, False

        sel = set(self._selected)

        if not sel:
            return list(range(n)), bool(self._closed)

        if len(sel) < 2:
            self.report({'WARNING'}, "Pen Fit: select at least 2 points")
            return None, False

        if not self._closed:
            lo = min(sel)
            hi = max(sel)
            expected = set(range(lo, hi + 1))

            if sel != expected:
                self.report(
                    {'WARNING'},
                    "Pen Fit: selected points must be one contiguous range"
                )
                return None, False

            return list(range(lo, hi + 1)), False

        # Closed source loop.
        if len(sel) == n:
            return list(range(n)), True

        # A single connected run on a cyclic loop has exactly one point whose
        # previous neighbor is unselected. This also supports wrap-around
        # selections such as {n-2, n-1, 0, 1}.
        starts = [
            i for i in sel
            if ((i - 1) % n) not in sel
        ]

        if len(starts) != 1:
            self.report(
                {'WARNING'},
                "Pen Fit: on a loop, select one connected arc only"
            )
            return None, False

        order = []
        i = starts[0]

        while i in sel:
            order.append(i)
            i = (i + 1) % n

            if len(order) > n:
                break

        if len(order) != len(sel):
            self.report(
                {'WARNING'},
                "Pen Fit: on a loop, select one connected arc only"
            )
            return None, False

        return order, False

    def _infer_closed_arc_order(self, stroke, panel_points):
        """Infer the shortest loop arc touched by an open stroke.

        This lets a closed section behave like a local strip: the user can
        draw near the intended points without tracing a complete closed line.
        """
        n = len(panel_points)
        if n < 2 or len(stroke) < 2:
            return None

        start = min(range(n), key=lambda i: (panel_points[i] - stroke[0]).length)
        end = min(range(n), key=lambda i: (panel_points[i] - stroke[-1]).length)
        if start == end:
            return [start]

        forward = []
        i = start
        while True:
            forward.append(i)
            if i == end or len(forward) > n:
                break
            i = (i + 1) % n

        backward = []
        i = start
        while True:
            backward.append(i)
            if i == end or len(backward) > n:
                break
            i = (i - 1) % n

        return forward if len(forward) <= len(backward) else backward

    def _blend_pen_target_map(self, targets, order, closed, radius=2, strength=0.65):
        """Blend selected-arc endpoints into their immediate neighbors."""
        if not targets or len(order) < 2 or strength <= 0.0:
            return targets
        radius = max(0, min(8, int(radius)))
        strength = max(0.0, min(1.0, float(strength)))
        if radius == 0:
            return targets
        selected = set(order)
        n = len(self._panel_points)
        if n < 3:
            return targets
        result = dict(targets)
        for endpoint in (order[0], order[-1]):
            base = self._primary_local_from_values(
                self._mouse_to_section(
                    self._panel_points[endpoint].x,
                    self._panel_points[endpoint].y,
                ),
                self._depths[endpoint],
            )
            # Targets are section-space coordinates, so use the same display
            # coordinates as the untouched endpoint for a stable delta.
            base_sec = self._mouse_to_section(
                self._panel_points[endpoint].x,
                self._panel_points[endpoint].y,
            )
            delta = targets[endpoint] - base_sec
            direction = 1 if endpoint == order[0] else -1
            for step in range(1, radius + 1):
                j = endpoint + direction * step
                if closed:
                    j %= n
                elif j < 0 or j >= n:
                    continue
                if j in selected:
                    continue
                weight = strength * (1.0 - step / (radius + 1.0)) ** 2
                neighbor = self._mouse_to_section(
                    self._panel_points[j].x,
                    self._panel_points[j].y,
                )
                result[j] = neighbor + delta * weight
        return result

    def _pen_stroke_is_closed(self):
        if len(self._pen_stroke) < 3:
            return False

        # Explicitly closed only when user ends near where they started.
        threshold = 20.0
        return (
            self._pen_stroke[-1] - self._pen_stroke[0]
        ).length <= threshold

    def _pen_smooth_result(
        self,
        base,
        amount,
        closed
    ):
        base = [
            p.copy()
            for p in base
        ]

        if len(base) < 3:
            return base

        amount = max(
            0.0,
            min(20.0, float(amount))
        )

        whole = int(amount)
        frac = amount - whole

        current = [
            p.copy()
            for p in base
        ]

        for _ in range(whole):
            current = _moving_average_2d(
                current,
                radius=1,
                passes=1,
                closed=closed
            )

        if frac > 1e-8:
            nxt = _moving_average_2d(
                current,
                radius=1,
                passes=1,
                closed=closed
            )

            current = [
                a.lerp(b, frac)
                for a, b in zip(
                    current,
                    nxt
                )
            ]

        if not closed and len(current) >= 2:
            current[0] = base[0].copy()
            current[-1] = base[-1].copy()

        return current

    def _begin_pen_smooth_drag(
        self,
        context,
        target,
        mx,
        my
    ):
        stroke = (
            self._secondary_pen_stroke
            if target == 'B'
            else self._pen_stroke
        )

        drawing = (
            self._secondary_pen_drawing
            if target == 'B'
            else self._pen_drawing
        )

        if drawing or len(stroke) < 3:
            _hp_smooth_debug(
                "PEN BEGIN rejected "
                f"target={target} drawing={drawing} points={len(stroke)}"
            )
            return False

        if target == 'B':
            _, full_loop = self._secondary_pen_selected_order()
            closed = bool(
                full_loop
                and len(stroke) >= 3
                and (stroke[-1] - stroke[0]).length <= 20.0
            )

            if len(self._secondary_pen_raw_stroke) != len(stroke):
                self._secondary_pen_raw_stroke = [
                    p.copy() for p in stroke
                ]
                self._secondary_pen_smooth_level = 0.0

            raw = self._secondary_pen_raw_stroke
            current_amount = self._secondary_pen_smooth_level
        else:
            _, full_loop = self._pen_selected_order()
            closed = bool(
                full_loop and self._pen_stroke_is_closed()
            )

            if len(self._pen_raw_stroke) != len(stroke):
                self._pen_raw_stroke = [
                    p.copy() for p in stroke
                ]
                self._pen_smooth_level = 0.0

            raw = self._pen_raw_stroke
            current_amount = self._pen_smooth_level

        self._pen_smooth_dragging = True
        self._brush_mode = 'PEN_B' if target == 'B' else 'PEN_A'
        self._pen_smooth_target = target
        self._pen_smooth_start_x = float(mx)
        self._pen_smooth_start_amount = float(current_amount)
        self._pen_smooth_amount = float(current_amount)
        self._smooth_hud_x = float(mx)
        self._smooth_hud_y = float(my)
        self._pen_smooth_brush_prev_mouse = Vector((mx, my))
        _hp_log(f"pen brush begin target={target} points={len(stroke)}")

        # Always recompute from the original unsmoothed line.
        self._pen_smooth_base = [
            p.copy() for p in raw
        ]
        self._pen_smooth_closed = closed

        # Snapshot only for canceling this E gesture.
        self._pen_smooth_cancel_stroke = [
            p.copy() for p in stroke
        ]
        self._pen_smooth_cancel_amount = float(current_amount)
        self._pen_smooth_last_delta_px = 0.0

        _hp_smooth_debug(
            "PEN BEGIN "
            f"target={target} points={len(stroke)} "
            f"closed={closed} start_amount={current_amount:.3f}"
        )

        context.area.tag_redraw()
        return True

    def _update_pen_smooth_drag(
        self,
        context,
        mx,
        my
    ):
        if not self._pen_smooth_dragging:
            return

        amount = self._pen_smooth_brush_strength
        self._pen_smooth_amount = amount
        self._smooth_hud_x = float(mx)
        self._smooth_hud_y = float(my)
        motion = Vector((mx, my)) - self._pen_smooth_brush_prev_mouse
        self._pen_smooth_brush_prev_mouse = Vector((mx, my))

        current = (
            self._secondary_pen_stroke
            if self._pen_smooth_target == 'B'
            else self._pen_stroke
        )
        result = _hp_brush_repel_2d(
            current,
            Vector((mx, my)),
            motion,
            radius=self._pen_smooth_brush_radius,
            strength=self._pen_smooth_brush_strength,
            closed=self._pen_smooth_closed,
        )

        max_delta = 0.0
        for a, b in zip(self._pen_smooth_base, result):
            max_delta = max(max_delta, (b - a).length)

        self._pen_smooth_last_delta_px = max_delta

        if self._pen_smooth_target == 'B':
            self._secondary_pen_stroke = [
                p.copy() for p in result
            ]
            self._secondary_pen_smooth_level = amount
        else:
            self._pen_stroke = [
                p.copy() for p in result
            ]
            self._pen_smooth_level = amount

        context.area.tag_redraw()

    def _finish_pen_smooth_drag(
        self,
        context,
        cancel=False
    ):
        if not self._pen_smooth_dragging:
            return

        if cancel:
            restored = [
                p.copy()
                for p in self._pen_smooth_cancel_stroke
            ]

            if self._pen_smooth_target == 'B':
                self._secondary_pen_stroke = restored
                self._secondary_pen_smooth_level = (
                    self._pen_smooth_cancel_amount
                )
            else:
                self._pen_stroke = restored
                self._pen_smooth_level = (
                    self._pen_smooth_cancel_amount
                )

            self._pen_smooth_amount = self._pen_smooth_cancel_amount

        _hp_smooth_debug(
            "PEN END "
            f"target={self._pen_smooth_target} "
            f"cancel={cancel} amount={self._pen_smooth_amount:.3f} "
            f"max_delta_px={self._pen_smooth_last_delta_px:.3f}"
        )

        self._pen_smooth_dragging = False
        self._brush_mode = None
        self._pen_smooth_target = None
        self._pen_smooth_base = []
        self._pen_smooth_cancel_stroke = []
        self._pen_smooth_closed = False

        context.area.tag_redraw()

    def _smooth_drawn_pen_once(self, secondary=False):
        if secondary:
            stroke = self._secondary_pen_stroke
            if len(stroke) < 3:
                return False

            if len(self._secondary_pen_raw_stroke) != len(stroke):
                self._secondary_pen_raw_stroke = [
                    p.copy() for p in stroke
                ]
                self._secondary_pen_smooth_level = 0.0

            _, full_loop = self._secondary_pen_selected_order()
            closed = bool(
                full_loop
                and (stroke[-1] - stroke[0]).length <= 20.0
            )
            self._secondary_pen_smooth_level = min(
                20.0,
                self._secondary_pen_smooth_level + 1.0
            )
            self._secondary_pen_stroke = self._pen_smooth_result(
                self._secondary_pen_raw_stroke,
                self._secondary_pen_smooth_level,
                closed
            )
        else:
            stroke = self._pen_stroke
            if len(stroke) < 3:
                return False

            if len(self._pen_raw_stroke) != len(stroke):
                self._pen_raw_stroke = [
                    p.copy() for p in stroke
                ]
                self._pen_smooth_level = 0.0

            _, full_loop = self._pen_selected_order()
            closed = bool(
                full_loop and self._pen_stroke_is_closed()
            )
            self._pen_smooth_level = min(
                20.0,
                self._pen_smooth_level + 1.0
            )
            self._pen_stroke = self._pen_smooth_result(
                self._pen_raw_stroke,
                self._pen_smooth_level,
                closed
            )

        return True

    def _pen_anchor_label(self):
        return (
            "描き始め優先"
            if self._pen_anchor_mode == 'STROKE'
            else "元点固定"
        )

    def _toggle_pen_anchor_mode(self):
        global _SECTION_PEN_ANCHOR_DEFAULT

        self._pen_anchor_mode = (
            'POINT'
            if self._pen_anchor_mode == 'STROKE'
            else 'STROKE'
        )

        _SECTION_PEN_ANCHOR_DEFAULT = self._pen_anchor_mode

    def _clear_all_pen_state(self):
        self._pen_mode = False
        self._pen_drawing = False
        self._pen_stroke = []
        self._pen_raw_stroke = []
        self._pen_smooth_level = 0.0

        self._secondary_pen_mode = False
        self._secondary_pen_drawing = False
        self._secondary_pen_stroke = []
        self._secondary_pen_raw_stroke = []
        self._secondary_pen_smooth_level = 0.0

        self._pen_smooth_dragging = False
        self._pen_smooth_target = None
        self._pen_smooth_base = []
        self._pen_smooth_cancel_stroke = []

    def _activate_pen_target(self, target):
        # Pen is globally exclusive between View A and View B.
        self._clear_all_pen_state()

        if target == 'B':
            self._secondary_pen_mode = True
        else:
            self._pen_mode = True

    def _prepare_pen_stroke(
        self,
        smooth,
        target_panel,
        closed
    ):
        smooth = [
            p.copy()
            for p in smooth
        ]

        if not smooth or not target_panel:
            return smooth

        # First choose the direction that corresponds to the target range.
        if closed:
            key = (tuple(tuple(p) for p in smooth),
                   tuple(tuple(p) for p in target_panel), self._pen_anchor_mode)
            cached = getattr(self, '_closed_pen_fit_cache', None)
            if cached is None or cached[0] != key:
                fitted = _align_closed_pen_stroke(smooth, target_panel, self._pen_anchor_mode)
                self._closed_pen_fit_cache = (key, fitted)
            return [p.copy() for p in self._closed_pen_fit_cache[1]]

        if not closed:
            forward = (
                (smooth[0] - target_panel[0]).length
                + (smooth[-1] - target_panel[-1]).length
            )

            reverse = (
                (smooth[-1] - target_panel[0]).length
                + (smooth[0] - target_panel[-1]).length
            )

            if reverse < forward:
                smooth.reverse()

        # POINT START keeps the original first target point exactly in place
        # by translating the complete drawn shape before resampling.
        if self._pen_anchor_mode == 'POINT':
            offset = target_panel[0] - smooth[0]

            smooth = [
                p + offset
                for p in smooth
            ]

        return smooth

    def _apply_pen(self, context):
        if len(self._pen_stroke) < 2:
            self.report({'WARNING'}, "Draw a stroke first")
            return

        order, full_loop = self._pen_selected_order()

        if order is None or len(order) < 2:
            return

        explicit_closed = self._pen_stroke_is_closed()
        _hp_log(
            f"pen apply order={len(order)} full_loop={full_loop} "
            f"explicit_closed={explicit_closed} source_closed={self._closed}"
        )

        # IMPORTANT:
        # A closed source section no longer forces every pen stroke into a loop.
        # Whole-loop replacement is allowed only when the user deliberately
        # closes the drawn stroke.
        if full_loop and not explicit_closed:
            inferred = self._infer_closed_arc_order(
                self._pen_stroke,
                self._panel_points,
            )
            if inferred is None or len(inferred) < 2:
                self.report({'WARNING'}, "閉じたループ上の近い点を判定できません")
                return
            order = inferred
            full_loop = False
            _hp_log(f"pen inferred closed arc points={len(order)}")

        stroke_closed = bool(full_loop and explicit_closed)

        smooth = _stabilize_2d(
            self._pen_stroke,
            level=self._pen_stabilizer,
            closed=stroke_closed
        )
        _hp_log(
            f"pen apply stabilizer={self._pen_stabilizer} "
            f"raw={len(self._pen_stroke)} smooth={len(smooth)}"
        )

        if len(smooth) < 2:
            return

        target_panel = [
            self._panel_points[i]
            for i in order
        ]

        smooth = self._prepare_pen_stroke(
            smooth,
            target_panel,
            stroke_closed
        )

        t_values = _path_t_values(
            target_panel,
            closed=stroke_closed
        )

        samples = _sample_polyline_2d(
            smooth,
            t_values,
            closed=stroke_closed
        )

        targets = {}

        for original_i, panel_p in zip(order, samples):
            targets[original_i] = self._mouse_to_section(
                panel_p.x,
                panel_p.y
            )

        targets = self._blend_pen_target_map(
            targets,
            order,
            closed=bool(self._closed),
            radius=getattr(self, '_hp_settings', {}).get('blend_radius', 2),
            strength=getattr(self, '_hp_settings', {}).get('blend_strength', 0.65),
        )
        _hp_log(
            f"pen endpoint blend targets={len(targets)-len(order)} "
            f"radius={getattr(self, '_hp_settings', {}).get('blend_radius', 2)} "
            f"strength={getattr(self, '_hp_settings', {}).get('blend_strength', 0.65):.2f}"
        )

        self._apply_section_targets(
            context,
            targets
        )

        self._clear_all_pen_state()
        context.area.tag_redraw()

    def _cancel_pen(self, context):
        self._clear_all_pen_state()
        context.area.tag_redraw()


    def _axis_vector(self):
        if self._transform_axis == 'X':
            return Vector((1.0, 0.0, 0.0))
        if self._transform_axis == 'Y':
            return Vector((0.0, 1.0, 0.0))
        if self._transform_axis == 'Z':
            return Vector((0.0, 0.0, 1.0))
        return None

    def _begin_transform(self, mode, mx, my):
        if not self._selected:
            return

        follow = self._hp_capture_direct_follow()
        if not self._push_history(follow):
            return
        self._direct_follow_state = follow

        self._transform_mode = mode
        self._transform_axis = None
        self._transform_start_mouse = Vector((mx, my))

        # Freeze the 2D display mapping for this transform.
        self._transform_fit_center = self._fit_center.copy()
        self._transform_fit_scale = self._fit_scale
        self._transform_flip_y = self._flip_y
        self._transform_flip_x = self._flip_x
        self._transform_section_rotation = self._section_display_angle()

        self._transform_base_coords = [p.copy() for p in self._coords]
        self._transform_base_depths = list(self._depths)
        self._transform_base_panel = [p.copy() for p in self._panel_points]

        mw = self._obj.matrix_world
        self._transform_base_world = [
            mw @ self._bm.verts[vi].co.copy()
            for vi in self._ordered
        ]

        sel = sorted(self._selected)

        self._transform_pivot_sec = (
            sum(
                (self._transform_base_coords[i] for i in sel),
                Vector((0.0, 0.0))
            ) / len(sel)
        )

        self._transform_pivot_world = (
            sum(
                (self._transform_base_world[i] for i in sel),
                Vector((0.0, 0.0, 0.0))
            ) / len(sel)
        )

        # Reuse the existing proportional-weight machinery.
        self._drag_base_panel = [p.copy() for p in self._panel_points]

    def _set_transform_axis(self, axis):
        if self._transform_axis == axis:
            self._transform_axis = None
        else:
            self._transform_axis = axis

    def _transform_weight(self, i):
        if i in self._selected:
            return 1.0

        if not self._prop_enabled or not self._selected:
            return 0.0

        if not self._xray:
            visible = set(
                self._primary_front_filter(
                    range(len(self._transform_base_panel)),
                    self._transform_base_panel
                )
            )
            if i not in visible:
                return 0.0

        p = self._transform_base_panel[i]

        d = min(
            (p - self._transform_base_panel[j]).length
            for j in self._selected
        )

        if d >= self._prop_radius_px:
            return 0.0

        return _smooth_falloff(
            d / self._prop_radius_px
        )

    def _transform_mouse_to_section(self, mx, my):
        pcx = self._panel_x + self._panel_w * 0.5
        pcy = self._panel_y + self._panel_h * 0.5 - 14

        sx = -1.0 if self._transform_flip_x else 1.0
        sy = -1.0 if self._transform_flip_y else 1.0
        scale = max(
            self._transform_fit_scale,
            1e-12
        )

        display_rel = Vector((
            (mx - pcx) / (scale * sx),
            (my - pcy) / (scale * sy),
        ))

        raw_rel = self._rotate_2d(
            display_rel,
            -self._transform_section_rotation
        )

        return (
            self._transform_fit_center
            + raw_rel
        )

    def _apply_transform(self, context, mx, my):
        if self._transform_mode is None or not self._selected:
            return

        mw = self._obj.matrix_world
        inv = mw.inverted()
        axis = self._axis_vector()

        mouse_delta = Vector((mx, my)) - self._transform_start_mouse

        # Mouse horizontal movement is intentionally used for S/R,
        # matching Blender's modal feel without introducing extra UI.
        scale_factor = max(0.01, 1.0 + mouse_delta.x / 140.0)
        angle = mouse_delta.x * 0.01

        # G uses the actual mini-section mouse delta.
        start_sec = self._transform_mouse_to_section(
            self._transform_start_mouse.x,
            self._transform_start_mouse.y
        )
        now_sec = self._transform_mouse_to_section(mx, my)
        sec_delta = now_sec - start_sec

        # Convert current View A plane translation to world space.
        section_world_delta = (
            self._primary_section_delta_to_world(sec_delta)
        )

        for i, vi in enumerate(self._ordered):
            w = self._transform_weight(i)
            if w <= 0.0:
                continue

            base_world = self._transform_base_world[i]
            full_target = base_world.copy()

            if self._transform_mode == 'G':
                if axis is None:
                    full_target = base_world + section_world_delta
                else:
                    full_target = (
                        base_world
                        + axis * section_world_delta.dot(axis)
                    )

            elif self._transform_mode == 'S':
                if axis is None:
                    # Scale in the mini-window's section plane.
                    base_sec = self._transform_base_coords[i]
                    scaled_sec = (
                        self._transform_pivot_sec
                        + (base_sec - self._transform_pivot_sec) * scale_factor
                    )

                    local = self._primary_local_from_values(
                        scaled_sec,
                        self._transform_base_depths[i]
                    )
                    full_target = mw @ local

                else:
                    # World-axis-only scale.
                    rel = base_world - self._transform_pivot_world
                    parallel = axis * rel.dot(axis)
                    perpendicular = rel - parallel

                    full_target = (
                        self._transform_pivot_world
                        + perpendicular
                        + parallel * scale_factor
                    )

            elif self._transform_mode == 'R':
                if axis is None:
                    # Rotate in section plane.
                    base_sec = self._transform_base_coords[i]
                    rel = base_sec - self._transform_pivot_sec

                    c = cos(angle)
                    s = sin(angle)

                    rotated = Vector((
                        rel.x * c - rel.y * s,
                        rel.x * s + rel.y * c
                    ))

                    rotated_sec = self._transform_pivot_sec + rotated

                    local = self._primary_local_from_values(
                        rotated_sec,
                        self._transform_base_depths[i]
                    )
                    full_target = mw @ local

                else:
                    # Rotate in real world space around the requested axis.
                    q = Quaternion(axis, angle)
                    rel = base_world - self._transform_pivot_world
                    full_target = (
                        self._transform_pivot_world
                        + q @ rel
                    )

            # Proportional editing blends between original and transformed target.
            world = base_world.lerp(full_target, w)

            # Persistent axis locks still work on top of transform constraints.
            if self._lock_x:
                world.x = base_world.x
            if self._lock_y:
                world.y = base_world.y
            if self._lock_z:
                world.z = base_world.z

            world = self._hp_clip_world(base_world, world)
            self._bm.verts[vi].co = inv @ world

        self._hp_apply_direct_follow()
        bmesh.update_edit_mesh(
            self._obj.data,
            loop_triangles=False,
            destructive=False
        )

        # Re-evaluate the displayed section positions from current mesh.
        current_coords = []
        current_depths = []

        for vi in self._ordered:
            local = self._bm.verts[vi].co
            sec_now, depth_now = (
                self._primary_values_from_local(local)
            )

            current_coords.append(sec_now)
            current_depths.append(depth_now)

        self._coords = current_coords
        self._depths = current_depths
        self._refresh_panel_points()

        # View B reads directly from the real mesh on draw; forcing redraw here
        # guarantees the secondary view updates in the same interaction frame.
        context.area.tag_redraw()

    def _confirm_transform(self, context):
        self._transform_mode = None
        self._direct_follow_state = None
        self._transform_axis = None
        context.area.tag_redraw()

    def _cancel_transform(self, context):
        self._transform_mode = None
        self._direct_follow_state = None
        self._transform_axis = None
        self._local_undo(context)

    def _handle_ctrl_z(self, context):
        if self._brush_mode == 'VERTEX_A' or self._brush_mode == 'VERTEX_B':
            self._finish_smooth_drag(context, cancel=True)
            return True

        if self._brush_mode == 'PEN_A' or self._brush_mode == 'PEN_B':
            self._finish_pen_smooth_drag(context, cancel=True)
            return True

        if self._pen_smooth_dragging:
            self._finish_pen_smooth_drag(context, cancel=True)
            return True

        if self._smooth_dragging:
            self._finish_smooth_drag(context, cancel=True)
            return True

        if self._transform_mode is not None:
            self._cancel_transform(context)
            return True

        if self._secondary_transform_mode is not None:
            self._secondary_cancel_transform(context)
            return True

        if self._dragging:
            self._dragging = False
            self._direct_follow_state = None
            return self._local_undo(context)

        if self._secondary_dragging:
            self._secondary_dragging = False
            self._direct_follow_state = None
            self._secondary_release_map()
            return self._local_undo(context)

        if self._box_dragging:
            self._box_dragging = False
            self._box_start = None
            self._box_end = None
            context.area.tag_redraw()
            return True

        if self._secondary_box_dragging:
            self._secondary_box_dragging = False
            self._secondary_box_start = None
            self._secondary_box_end = None
            context.area.tag_redraw()
            return True

        if self._pen_drawing or self._secondary_pen_drawing:
            self._clear_all_pen_state()
            context.area.tag_redraw()
            return True

        return self._local_undo(context)

    def _zoom_view(self, factor):
        self._view_zoom = max(
            self._view_zoom_min,
            min(
                self._view_zoom_max,
                self._view_zoom * factor
            )
        )
        self._recompute_fit_scale()
        self._refresh_panel_points()

    def modal(self, context, event):
        try:
            result = self._modal_impl(context, event)
            self._tag_views(context)
            return result
        except Exception:
            # Blender stops calling a modal operator after an uncaught error.
            # Release its draw handlers and running flag so the editor can
            # reopen without restarting Blender.
            _hp_follow_log('modal_error\n' + traceback.format_exc())
            state = getattr(self, '_hp_menu', None)
            if state is not None:
                try:
                    self._hp_restore(state)
                except Exception:
                    _hp_follow_log('restore_error\n' + traceback.format_exc())
                self._hp_menu = None
            try:
                self._finish(context, release_workspace=False)
            except Exception:
                _hp_follow_log('cleanup_error\n' + traceback.format_exc())
                global _RUNNING, _ACTIVE_SECTION_EDITOR
                for attr in ('_handle', '_handle_follow'):
                    handle = getattr(self, attr, None)
                    if handle is not None:
                        try:
                            bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
                        except Exception:
                            pass
                        setattr(self, attr, None)
                self._finished = True
                _RUNNING = False
                _ACTIVE_SECTION_EDITOR = None
            try:
                self.report({'ERROR'}, 'HP Section: エラー後に断面窓を終了しました。再選択で開き直せます')
            except Exception:
                pass
            return {'CANCELLED'}

    def _modal_impl(self, context, event):
        if getattr(self, "_finished", False):
            return {'FINISHED'}

        # A window/workspace change ends this instance; target selection
        # changes only suspend it and never close its window.
        if not self._owner_workspace_active(context):
            return self._finish(
                context,
                release_workspace=False
            )

        if not self._owner_context_active(context):
            self._viewport_navigation_active = False
            return {'PASS_THROUGH'}

        if context.area.type != 'VIEW_3D':
            return {'PASS_THROUGH'}

        if self._preview_state and not self._target_available(context):
            self._preview_finish_edit(context, cancel=True)
            self._idle = True

        if event.type == 'TIMER':
            self._layout_panels(context)
            busy = (self._preview_state or self._dragging or self._box_dragging or self._brush_mode
                    or self._pen_mode or self._secondary_pen_mode
                    or self._transform_mode or self._secondary_transform_mode
                    or self._hp_menu is not None)
            if self._topology_rebuild_pending:
                chain = _selected_chain(context)
                if chain is not None:
                    desired = self._topology_pending_selected or set()
                    self._rebuild(context, chain)
                    self._selected = {i for i in desired if i < len(self._ordered)}
                    self._topology_pending_selected = None
                    self._topology_rebuild_pending = False
                    self._idle = False
            elif not busy:
                chain = _selected_chain(context)
                if chain is None:
                    self._idle = True
                    self._selected.clear()
                    self._signature = None
                elif self._idle or chain[0] != self._obj or not self._bmesh_ready() or chain[4] != self._signature:
                    self._history.clear()
                    self._topology_selection_history.clear()
                    self._topology_undo_steps = 0
                    self._selected.clear()
                    self._rebuild(context, chain)
                    self._idle = False
                    if self.detached:
                        self._frame_preview(context)
                else:
                    self._sync_primary_from_mesh()
            self._tag_views(context)
        if self._idle or not self._target_available(context):
            if event.type == 'ESC' and event.value == 'PRESS':
                return self._finish(context, release_workspace=True)
            return {'PASS_THROUGH'}
        # Outside the panels, Blender owns keyboard shortcuts and redo input.
        # Active gestures still receive release/cancel events outside their bounds.
        mx, my = event.mouse_region_x, event.mouse_region_y
        active_gesture = (self._dragging or self._secondary_dragging
                          or self._box_dragging or self._secondary_box_dragging
                          or self._brush_mode or self._pen_drawing
                          or self._secondary_pen_drawing or self._resize_mode
                          or self._transform_mode or self._secondary_transform_mode
                          or self._viewport_navigation_active or self._f_hold_active or getattr(self, '_hp_confirm_hold', None)
                          or self._hp_menu is not None)
        if (event.type == 'RIGHTMOUSE' and event.value == 'PRESS'
                and not active_gesture and self._preview_state is None
                and not self._pen_mode and not self._secondary_pen_mode):
            return {'PASS_THROUGH'}
        in_panels = (self._inside_panel(mx, my) or self._inside_secondary(mx, my)
                     or self._inside_primary_header(mx, my)
                     or self._inside_secondary_header(mx, my))
        if not active_gesture:
            preview_result = self._preview_event(context, event, in_panels)
            if preview_result is not None:
                return preview_result
        if event.type != 'TIMER' and not in_panels and not active_gesture:
            return {'PASS_THROUGH'}

        # Pass viewport navigation through even when the mini editor overlay
        # is under the pointer. Mouse movement during orbit must not become a
        # pen or brush stroke.
        if event.type == 'MIDDLEMOUSE':
            self._viewport_navigation_active = event.value != 'RELEASE'
            return {'PASS_THROUGH'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            # An MMB release may be swallowed by another viewport operator.
            self._viewport_navigation_active = False
        if event.type in {
            'TRACKPADPAN', 'TRACKPADZOOM', 'TRACKPADROTATE',
            'MOUSEROTATE', 'NDOF_MOTION',
        }:
            return {'PASS_THROUGH'}
        if (
            event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}
            and (event.ctrl or event.shift)
        ):
            return {'PASS_THROUGH'}
        if self._viewport_navigation_active and event.type in {
            'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE',
        }:
            return {'PASS_THROUGH'}

        # Local undo has to win before Pen / Brush / View B modal branches can
        # consume Ctrl+Z.
        if event.ctrl and event.type == 'Z' and event.value == 'PRESS':
            if self._handle_ctrl_z(context):
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}

        # Shift+F is the explicit Pen release shortcut. Handle it before the
        # F-hold/menu dispatcher so it also works while a hold is pending or
        # while the adjustment menu is visible.
        if (
            event.type == 'F'
            and event.value == 'PRESS'
            and event.shift
            and not event.ctrl
            and not event.alt
        ):
            if self._brush_mode == 'VERTEX_A' or self._brush_mode == 'VERTEX_B':
                self._finish_smooth_drag(context, cancel=True)
            elif self._brush_mode == 'PEN_A' or self._brush_mode == 'PEN_B':
                self._finish_pen_smooth_drag(context, cancel=True)
            if getattr(self, '_hp_menu', None) is not None:
                self._hp_end_adjust(context, False)
            self._hp_confirm_hold = None
            self._f_hold_active = False
            self._f_long_opened = False
            self._closed_pen_fit_cache = None
            self._clear_all_pen_state()
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # Dedicated brush dispatch. Once E has started a brush, no Pen,
        # proportional-edit, or viewport-wheel branch may consume its input.
        if self._brush_mode is not None:
            mx = event.mouse_region_x
            my = event.mouse_region_y
            if event.value == 'PRESS' and event.type in {
                'WHEELUPMOUSE', 'WHEELDOWNMOUSE'
            }:
                if self._brush_mode.startswith('PEN_'):
                    radius = self._pen_smooth_brush_radius
                    strength = self._pen_smooth_brush_strength
                else:
                    radius = self._smooth_brush_radius
                    strength = self._smooth_brush_strength

                if event.alt:
                    strength += 0.08 if event.type == 'WHEELUPMOUSE' else -0.08
                    strength = max(0.05, min(1.0, strength))
                    if self._brush_mode.startswith('PEN_'):
                        self._pen_smooth_brush_strength = strength
                    else:
                        self._smooth_brush_strength = strength
                    _hp_log(f"brush strength={strength:.2f}")
                else:
                    radius += 10.0 if event.type == 'WHEELUPMOUSE' else -10.0
                    radius = max(20.0, min(240.0, radius))
                    if self._brush_mode.startswith('PEN_'):
                        self._pen_smooth_brush_radius = radius
                    else:
                        self._smooth_brush_radius = radius
                    _hp_log(f"brush radius={radius:.1f}")
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
                if self._brush_mode.startswith('PEN_'):
                    self._update_pen_smooth_drag(context, mx, my)
                else:
                    self._update_smooth_drag(context, mx, my)
                return {'RUNNING_MODAL'}

            if event.type == 'E' and event.value == 'RELEASE':
                if self._brush_mode.startswith('PEN_'):
                    self._finish_pen_smooth_drag(context, cancel=False)
                else:
                    self._finish_smooth_drag(context, cancel=False)
                return {'RUNNING_MODAL'}

            if event.value == 'PRESS' and event.type in {'ESC', 'RIGHTMOUSE'}:
                if self._brush_mode.startswith('PEN_'):
                    self._finish_pen_smooth_drag(context, cancel=True)
                else:
                    self._finish_smooth_drag(context, cancel=True)
                return {'RUNNING_MODAL'}

            return {'RUNNING_MODAL'}

        result = self._hp_pen_key(context, event)
        if result is not None:
            return result

        # Esc / Shift+F always leaves Pen, including during E-smoothing.
        # It discards only the uncommitted stroke and keeps the editor open.
        if (event.value == 'PRESS'
                and (event.type == 'ESC' or
                     (event.type == 'F' and event.shift and not event.ctrl and not event.alt))
                and (self._pen_mode or self._secondary_pen_mode or self._pen_smooth_dragging)):
            self._clear_all_pen_state()
            self._f_hold_active = False
            self._f_long_opened = False
            self._closed_pen_fit_cache = None
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        mx = event.mouse_region_x
        my = event.mouse_region_y
        inside = self._inside_panel(mx, my)
        inside_b = self._inside_secondary(mx, my)

        # Defensive repair for any stale state from an older hot-reloaded build:
        # if both somehow became active, keep only the window under the cursor.
        if self._pen_mode and self._secondary_pen_mode:
            if inside_b:
                self._pen_mode = False
                self._pen_drawing = False
                self._pen_stroke = []
                self._pen_raw_stroke = []
                self._pen_smooth_level = 0.0
            else:
                self._secondary_pen_mode = False
                self._secondary_pen_drawing = False
                self._secondary_pen_stroke = []
                self._secondary_pen_raw_stroke = []
                self._secondary_pen_smooth_level = 0.0

            context.area.tag_redraw()

        # E held while a Pen stroke exists = visually smooth the DRAWN LINE.
        # This runs before geometry Smooth so the two operations cannot collide.
        if self._pen_smooth_dragging:
            if event.value == 'PRESS' and event.type in {
                'WHEELUPMOUSE',
                'WHEELDOWNMOUSE'
            }:
                if event.alt:
                    delta = 0.08 if event.type == 'WHEELUPMOUSE' else -0.08
                    self._pen_smooth_brush_strength = max(
                        0.05,
                        min(1.0, self._pen_smooth_brush_strength + delta)
                    )
                else:
                    delta = 10.0 if event.type == 'WHEELUPMOUSE' else -10.0
                    self._pen_smooth_brush_radius = max(
                        20.0,
                        min(240.0, self._pen_smooth_brush_radius + delta)
                    )
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'MOUSEMOVE':
                self._update_pen_smooth_drag(
                    context,
                    mx,
                    my
                )
                return {'RUNNING_MODAL'}

            if (
                event.type == 'E'
                and event.value == 'RELEASE'
            ):
                self._finish_pen_smooth_drag(
                    context,
                    cancel=False
                )
                return {'RUNNING_MODAL'}

            if (
                event.value == 'PRESS'
                and event.type in {
                    'RIGHTMOUSE',
                    'ESC'
                }
            ):
                self._finish_pen_smooth_drag(
                    context,
                    cancel=True
                )
                return {'RUNNING_MODAL'}

            return {'RUNNING_MODAL'}

        if self._smooth_dragging:
            if event.value == 'PRESS' and event.type in {
                'WHEELUPMOUSE',
                'WHEELDOWNMOUSE'
            }:
                if event.alt:
                    delta = 0.08 if event.type == 'WHEELUPMOUSE' else -0.08
                    self._smooth_brush_strength = max(
                        0.05,
                        min(1.0, self._smooth_brush_strength + delta)
                    )
                else:
                    delta = 10.0 if event.type == 'WHEELUPMOUSE' else -10.0
                    self._smooth_brush_radius = max(
                        20.0,
                        min(240.0, self._smooth_brush_radius + delta)
                    )
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'MOUSEMOVE':
                self._update_smooth_drag(
                    context,
                    mx,
                    my
                )
                return {'RUNNING_MODAL'}

            if (
                event.type == 'E'
                and event.value == 'RELEASE'
            ):
                self._finish_smooth_drag(
                    context,
                    cancel=False
                )
                return {'RUNNING_MODAL'}

            if (
                event.value == 'PRESS'
                and event.type in {
                    'RIGHTMOUSE',
                    'ESC'
                }
            ):
                self._finish_smooth_drag(
                    context,
                    cancel=True
                )
                return {'RUNNING_MODAL'}

            return {'RUNNING_MODAL'}

        # In Pen mode:
        # - after a stroke is drawn, F confirms/applies it
        # - before drawing, F switches Stroke Start / Point Start
        if (
            event.type == 'F'
            and not event.ctrl
            and event.value == 'PRESS'
            and (self._pen_mode or self._secondary_pen_mode)
        ):
            if bool(getattr(event, "is_repeat", False)):
                return {'RUNNING_MODAL'}

            if (
                self._pen_mode
                and self._pen_stroke
                and not self._pen_drawing
            ):
                self._apply_pen(context)
                return {'RUNNING_MODAL'}

            if (
                self._secondary_pen_mode
                and self._secondary_pen_stroke
                and not self._secondary_pen_drawing
            ):
                self._secondary_apply_pen(context)
                return {'RUNNING_MODAL'}

            self._toggle_pen_anchor_mode()
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # Outside pen mode:
        # quick F = enter pen using current mode
        # hold F   = open mode pie
        if (
            event.type == 'F'
            and not event.ctrl
            and not (self._pen_mode or self._secondary_pen_mode)
        ):
            if event.value == 'PRESS' and (inside or inside_b):
                if not self._f_hold_active:
                    self._f_hold_active = True
                    self._f_hold_started = time.perf_counter()
                    self._f_hold_target = 'B' if inside_b else 'A'
                    self._f_long_opened = False

                return {'RUNNING_MODAL'}

            if event.value == 'RELEASE' and self._f_hold_active:
                target = self._f_hold_target
                long_opened = self._f_long_opened

                self._f_hold_active = False
                self._f_long_opened = False

                if not long_opened:
                    self._activate_pen_target(target)
                    context.area.tag_redraw()

                return {'RUNNING_MODAL'}

        if (
            event.type == 'TIMER'
            and self._f_hold_active
            and not self._f_long_opened
            and (
                time.perf_counter()
                - self._f_hold_started
            ) >= PEN_F_HOLD_SECONDS
        ):
            self._f_long_opened = True
            self._f_hold_active = False

            # A long F press on an existing closed loop opens the adjustment
            # menu even when no pen stroke has been drawn yet.  This keeps the
            # loop-sharing controls available as a setup action.
            target = self._f_hold_target
            opened = False
            if not self._pen_mode and not self._secondary_pen_mode:
                opened = self._hp_begin_adjust(
                    context,
                    target,
                    show=True,
                    settings_only=True,
                )
            if opened:
                return {'RUNNING_MODAL'}

            self._pie_target = target

            bpy.ops.wm.call_menu_pie(
                name=HP_MT_section_pen_anchor_pie.bl_idname
            )

            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            quick = self._hp_quick_follow_button(mx, my)
            if quick is not None and not self._pen_mode and not self._secondary_pen_mode:
                self._hp_quick_follow_click(context, quick)
                return {'RUNNING_MODAL'}
            if self._inside_primary_header(mx, my):
                self._view_a_collapsed = not self._view_a_collapsed
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if self._inside_secondary_header(mx, my):
                self._view_b_collapsed = not self._view_b_collapsed
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

        # ---------------- View B editing ----------------
        if self._secondary_transform_mode is not None:
            if (
                event.value == 'PRESS'
                and self._prop_enabled
                and event.type in {
                    'WHEELUPMOUSE',
                    'WHEELDOWNMOUSE'
                }
            ):
                if event.type == 'WHEELUPMOUSE':
                    self._prop_radius_px = min(
                        320.0,
                        self._prop_radius_px + 12.0
                    )
                else:
                    self._prop_radius_px = max(
                        24.0,
                        self._prop_radius_px - 12.0
                    )

                self._secondary_apply_transform(
                    context,
                    mx,
                    my
                )
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if (
                event.value == 'PRESS'
                and event.type in {'X', 'Y', 'Z'}
            ):
                if self._secondary_transform_axis == event.type:
                    self._secondary_transform_axis = None
                else:
                    self._secondary_transform_axis = event.type

                self._secondary_rebase_transform(mx, my)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'MOUSEMOVE':
                self._secondary_apply_transform(
                    context,
                    mx,
                    my
                )
                return {'RUNNING_MODAL'}

            if (
                event.value == 'PRESS'
                and event.type in {
                    'LEFTMOUSE',
                    'RET',
                    'NUMPAD_ENTER'
                }
            ):
                self._secondary_confirm_transform(
                    context
                )
                return {'RUNNING_MODAL'}

            if (
                event.value == 'PRESS'
                and event.type in {
                    'RIGHTMOUSE',
                    'ESC'
                }
            ):
                self._secondary_cancel_transform(
                    context
                )
                return {'RUNNING_MODAL'}

            return {'RUNNING_MODAL'}

        if self._secondary_pen_mode:
            if (
                event.type == 'E'
                and not event.ctrl
                and event.value == 'PRESS'
                and self._secondary_pen_stroke
                and not self._secondary_pen_drawing
            ):
                self._begin_pen_smooth_drag(
                    context,
                    'B',
                    mx,
                    my
                )
                return {'RUNNING_MODAL'}

            if (
                event.type == 'S'
                and event.value == 'PRESS'
                and self._secondary_pen_stroke
                and not self._secondary_pen_drawing
            ):
                self._smooth_drawn_pen_once(
                    secondary=True
                )
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            # RMB behavior mirrors View A.
            if event.type == 'RIGHTMOUSE' and event.value == 'PRESS':
                if self._secondary_pen_stroke:
                    self._secondary_pen_stroke = []
                    self._secondary_pen_raw_stroke = []
                    self._secondary_pen_smooth_level = 0.0
                    self._secondary_pen_drawing = False
                else:
                    self._clear_all_pen_state()

                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.value == 'PRESS':
                if event.type == 'F':
                    self._clear_all_pen_state()
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}

                if event.type == 'WHEELUPMOUSE':
                    self._secondary_pen_stabilizer = min(
                        10,
                        self._secondary_pen_stabilizer + 1
                    )
                    _hp_log(f"pen stabilizer B={self._secondary_pen_stabilizer}")
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}

                if event.type == 'WHEELDOWNMOUSE':
                    self._secondary_pen_stabilizer = max(
                        0,
                        self._secondary_pen_stabilizer - 1
                    )
                    _hp_log(f"pen stabilizer B={self._secondary_pen_stabilizer}")
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}

                if event.type in {'RET', 'NUMPAD_ENTER'}:
                    self._secondary_apply_pen(
                        context
                    )
                    return {'RUNNING_MODAL'}

                if event.type == 'ESC':
                    self._clear_all_pen_state()
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}

            if (
                event.type == 'LEFTMOUSE'
                and event.value == 'PRESS'
                and inside_b
            ):
                self._secondary_pen_stroke = [
                    Vector((mx, my))
                ]
                self._secondary_pen_raw_stroke = []
                self._secondary_pen_smooth_level = 0.0
                self._secondary_pen_drawing = True
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if (
                event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}
                and self._secondary_pen_drawing
            ):
                p = Vector((mx, my))

                if (
                    not self._secondary_pen_stroke
                    or (
                        p - self._secondary_pen_stroke[-1]
                    ).length >= 1.5
                ):
                    self._secondary_pen_stroke.append(p)

                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if (
                event.type == 'LEFTMOUSE'
                and event.value == 'RELEASE'
                and self._secondary_pen_drawing
            ):
                p = Vector((mx, my))

                if (
                    not self._secondary_pen_stroke
                    or (
                        p - self._secondary_pen_stroke[-1]
                    ).length >= 0.75
                ):
                    self._secondary_pen_stroke.append(p)

                self._secondary_pen_drawing = False
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if inside_b:
                return {'RUNNING_MODAL'}

        if self._secondary_dragging:
            if event.type == 'MOUSEMOVE':
                self._secondary_apply_drag(
                    context,
                    mx,
                    my
                )
                return {'RUNNING_MODAL'}

            if (
                event.type == 'LEFTMOUSE'
                and event.value == 'RELEASE'
            ):
                self._secondary_apply_drag(
                    context,
                    mx,
                    my
                )
                self._secondary_finish_drag(
                    context
                )
                return {'RUNNING_MODAL'}

        if self._secondary_box_dragging:
            if event.type == 'MOUSEMOVE':
                self._secondary_box_end = Vector((mx, my))
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if (
                event.type == 'LEFTMOUSE'
                and event.value == 'RELEASE'
            ):
                self._secondary_box_end = Vector((mx, my))
                self._secondary_apply_box_select()

                self._secondary_box_dragging = False
                self._secondary_box_start = None
                self._secondary_box_end = None

                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

        if inside_b and event.value == 'PRESS':
            if event.alt and event.type == 'Z':
                self._xray = not self._xray
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'NUMPAD_1':
                if event.ctrl:
                    self._set_secondary_view('BACK')
                else:
                    self._set_secondary_view('FRONT')

                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'NUMPAD_3':
                if event.ctrl:
                    self._set_secondary_view('LEFT')
                else:
                    self._set_secondary_view('RIGHT')

                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'NUMPAD_8':
                self._secondary_flip_y = not self._secondary_flip_y
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if (
                event.type == 'M'
                and not event.ctrl
                and not event.shift
                and not event.alt
                and self._selected
            ):
                bpy.ops.wm.call_menu_pie(
                    name=HP_MT_section_topology_pie.bl_idname
                )
                return {'RUNNING_MODAL'}

            if (
                event.type == 'E'
                and not event.ctrl
                and event.value == 'PRESS'
            ):
                self._begin_smooth_drag(
                    context,
                    'B',
                    mx,
                    my
                )
                return {'RUNNING_MODAL'}

            if (
                event.type == 'F'
                and event.ctrl
                and self._selected
            ):
                self._shape_pie_target = 'B'
                bpy.ops.wm.call_menu_pie(
                    name=HP_MT_section_shape_pie.bl_idname
                )
                return {'RUNNING_MODAL'}

            if event.type == 'F' and not event.ctrl:
                self._activate_pen_target('B')
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if (
                event.type in {'G', 'S', 'R'}
                and self._selected
            ):
                self._secondary_begin_transform(
                    event.type,
                    mx,
                    my
                )
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'A':
                self._selected = set(
                    range(len(self._ordered))
                )
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'O':
                self._prop_enabled = not self._prop_enabled
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'WHEELUPMOUSE':
                if self._prop_enabled:
                    self._prop_radius_px = min(
                        320.0,
                        self._prop_radius_px + 12.0
                    )
                else:
                    self._secondary_zoom = min(
                        5.0,
                        self._secondary_zoom * 1.15
                    )

                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'WHEELDOWNMOUSE':
                if self._prop_enabled:
                    self._prop_radius_px = max(
                        24.0,
                        self._prop_radius_px - 12.0
                    )
                else:
                    self._secondary_zoom = max(
                        0.35,
                        self._secondary_zoom / 1.15
                    )

                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

        if (
            inside_b
            and event.type == 'LEFTMOUSE'
            and event.value == 'PRESS'
        ):
            idx = self._secondary_pick_point(
                mx,
                my
            )

            if idx is not None:
                if event.shift:
                    if idx in self._selected:
                        self._selected.remove(idx)
                        context.area.tag_redraw()
                        return {'RUNNING_MODAL'}
                    else:
                        self._selected.add(idx)
                else:
                    if idx not in self._selected:
                        self._selected = {idx}

                self._secondary_begin_drag(
                    mx,
                    my
                )

                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            self._secondary_box_dragging = True
            self._secondary_box_start = Vector((mx, my))
            self._secondary_box_end = Vector((mx, my))
            self._secondary_box_additive = bool(event.shift)
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # Window resize mode: Shift+S then move mouse left/right across 3 stages.
        if self._resize_mode:
            if event.type == 'MOUSEMOVE':
                self._update_resize_mode(mx)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.value == 'PRESS' and event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER'}:
                self._confirm_resize_mode()
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.value == 'PRESS' and event.type in {'RIGHTMOUSE', 'ESC'}:
                self._cancel_resize_mode()
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            return {'RUNNING_MODAL'}

        # Blender-like G/S/R transform modal.
        if self._transform_mode is not None:
            if (
                event.value == 'PRESS'
                and self._prop_enabled
                and event.type in {
                    'WHEELUPMOUSE',
                    'WHEELDOWNMOUSE'
                }
            ):
                if event.type == 'WHEELUPMOUSE':
                    self._prop_radius_px = min(
                        320.0,
                        self._prop_radius_px + 12.0
                    )
                else:
                    self._prop_radius_px = max(
                        24.0,
                        self._prop_radius_px - 12.0
                    )

                self._apply_transform(
                    context,
                    mx,
                    my
                )
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.value == 'PRESS' and event.type in {'X', 'Y', 'Z'}:
                self._set_transform_axis(event.type)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'MOUSEMOVE':
                self._apply_transform(context, mx, my)
                return {'RUNNING_MODAL'}

            if (
                event.value == 'PRESS'
                and event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER'}
            ):
                self._confirm_transform(context)
                return {'RUNNING_MODAL'}

            if (
                event.value == 'PRESS'
                and event.type in {'RIGHTMOUSE', 'ESC'}
            ):
                self._cancel_transform(context)
                return {'RUNNING_MODAL'}

            return {'RUNNING_MODAL'}

        # Normal mini-window wheel = camera/view zoom.
        # Existing special wheel behavior is preserved:
        # - Pen mode uses wheel for stroke stabilization.
        # - Proportional mode uses wheel for influence radius.
        if (
            inside
            and not self._pen_mode
            and not self._prop_enabled
            and event.value == 'PRESS'
        ):
            if event.type == 'WHEELUPMOUSE':
                self._zoom_view(1.15)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'WHEELDOWNMOUSE':
                self._zoom_view(1.0 / 1.15)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

        # Panel-scoped action keys.
        if inside and event.value == 'PRESS' and not self._pen_mode:
            if event.alt and event.type == 'Z':
                self._xray = not self._xray
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'NUMPAD_1':
                self._set_primary_view(
                    context,
                    'BACK' if event.ctrl else 'FRONT',
                    auto_pair_secondary=True
                )
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'NUMPAD_3':
                self._set_primary_view(
                    context,
                    'LEFT' if event.ctrl else 'RIGHT',
                    auto_pair_secondary=True
                )
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            # Num7 returns View A to its original SECTION-facing projection.
            # View B is intentionally left untouched.
            if event.type == 'NUMPAD_7':
                self._set_primary_view(
                    context,
                    'SECTION'
                )
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            # SECTION-only display orientation shortcuts.
            # They rotate the mini-window mapping, not the actual mesh.
            if (
                self._primary_view == 'SECTION'
                and event.type == 'NUMPAD_2'
            ):
                self._orient_selected_down()
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if (
                self._primary_view == 'SECTION'
                and event.type == 'NUMPAD_4'
            ):
                if self._orient_world_axis_down(
                    Vector((1.0, 0.0, 0.0)),
                    mode='WORLD_X'
                ):
                    # Num4 convention:
                    # +X = DOWN, projected +Y = RIGHT.
                    self._canonicalize_companion_axis_right(
                        Vector((0.0, 1.0, 0.0))
                    )

                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if (
                self._primary_view == 'SECTION'
                and event.type == 'NUMPAD_6'
            ):
                if self._orient_world_axis_down(
                    Vector((0.0, 1.0, 0.0)),
                    mode='WORLD_Y'
                ):
                    # Num6 convention:
                    # +Y = DOWN, projected +X = RIGHT.
                    self._canonicalize_companion_axis_right(
                        Vector((1.0, 0.0, 0.0))
                    )

                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if (
                event.type == 'M'
                and not event.ctrl
                and not event.shift
                and not event.alt
                and self._selected
            ):
                bpy.ops.wm.call_menu_pie(
                    name=HP_MT_section_topology_pie.bl_idname
                )
                return {'RUNNING_MODAL'}

            if (
                event.type == 'E'
                and not event.ctrl
                and event.value == 'PRESS'
            ):
                self._begin_smooth_drag(
                    context,
                    'A',
                    mx,
                    my
                )
                return {'RUNNING_MODAL'}

            if (
                event.type == 'F'
                and event.ctrl
                and self._selected
            ):
                self._shape_pie_target = 'A'
                bpy.ops.wm.call_menu_pie(
                    name=HP_MT_section_shape_pie.bl_idname
                )
                return {'RUNNING_MODAL'}

            if event.type == 'F' and not event.ctrl:
                self._activate_pen_target('A')
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'S' and event.shift:
                self._begin_resize_mode(mx)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type in {'G', 'S', 'R'} and self._selected:
                self._begin_transform(event.type, mx, my)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'A':
                self._selected = set(range(len(self._ordered)))
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'O':
                self._prop_enabled = not self._prop_enabled
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if (
                event.type == 'NUMPAD_8'
                and event.ctrl
                and self._primary_view == 'SECTION'
            ):
                self._flip_x = not self._flip_x
                self._remember_current_screen_right()

                self._refresh_panel_points()
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if (
                event.type == 'NUMPAD_8'
                and not event.ctrl
            ):
                self._flip_y = not self._flip_y

                global _LAST_SECTION_FLIP_Y
                _LAST_SECTION_FLIP_Y = bool(
                    self._flip_y
                )

                self._recompute_fit_scale()
                self._refresh_panel_points()
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'X':
                self._lock_x = not self._lock_x
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'Y':
                self._lock_y = not self._lock_y
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'Z':
                self._lock_z = not self._lock_z
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type in {'EQUAL', 'NUMPAD_PLUS'}:
                self._set_panel_stage(self._panel_stage + 1)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type in {'MINUS', 'NUMPAD_MINUS'}:
                self._set_panel_stage(self._panel_stage - 1)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'HOME':
                self._reset_panel_size()
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'WHEELUPMOUSE' and self._prop_enabled:
                self._prop_radius_px = min(320.0, self._prop_radius_px + 12.0)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'WHEELDOWNMOUSE' and self._prop_enabled:
                self._prop_radius_px = max(24.0, self._prop_radius_px - 12.0)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

        # Pen mode
        if self._pen_mode:
            if (
                event.type == 'E'
                and not event.ctrl
                and event.value == 'PRESS'
                and self._pen_stroke
                and not self._pen_drawing
            ):
                self._begin_pen_smooth_drag(
                    context,
                    'A',
                    mx,
                    my
                )
                return {'RUNNING_MODAL'}

            if (
                event.type == 'S'
                and event.value == 'PRESS'
                and self._pen_stroke
                and not self._pen_drawing
            ):
                self._smooth_drawn_pen_once(
                    secondary=False
                )
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            # RMB behavior:
            # - no stroke: leave Pen mode
            # - stroke already drawn: clear that stroke, remain in Pen mode
            if event.type == 'RIGHTMOUSE' and event.value == 'PRESS':
                if self._pen_stroke:
                    self._pen_stroke = []
                    self._pen_raw_stroke = []
                    self._pen_smooth_level = 0.0
                    self._pen_drawing = False
                else:
                    self._cancel_pen(context)

                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.value == 'PRESS':
                if event.type == 'F':
                    self._cancel_pen(context)
                    return {'RUNNING_MODAL'}

                if event.type == 'WHEELUPMOUSE':
                    self._pen_stabilizer = min(10, self._pen_stabilizer + 1)
                    _hp_log(f"pen stabilizer A={self._pen_stabilizer}")
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}

                if event.type == 'WHEELDOWNMOUSE':
                    self._pen_stabilizer = max(0, self._pen_stabilizer - 1)
                    _hp_log(f"pen stabilizer A={self._pen_stabilizer}")
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}

                if event.type in {'RET', 'NUMPAD_ENTER'}:
                    self._apply_pen(context)
                    return {'RUNNING_MODAL'}

            if event.type == 'LEFTMOUSE' and event.value == 'PRESS' and inside:
                self._pen_stroke = [Vector((mx, my))]
                self._pen_raw_stroke = []
                self._pen_smooth_level = 0.0
                self._pen_drawing = True
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'} and self._pen_drawing:
                p = Vector((mx, my))
                if (not self._pen_stroke) or (p - self._pen_stroke[-1]).length >= 1.5:
                    self._pen_stroke.append(p)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and self._pen_drawing:
                p = Vector((mx, my))
                if (not self._pen_stroke) or (p - self._pen_stroke[-1]).length >= 0.75:
                    self._pen_stroke.append(p)
                self._pen_drawing = False
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'ESC' and event.value == 'PRESS':
                self._cancel_pen(context)
                return {'RUNNING_MODAL'}

            if inside:
                return {'RUNNING_MODAL'}

        if event.type == 'ESC' and event.value == 'PRESS':
            if self._box_dragging:
                self._box_dragging = False
                self._box_start = None
                self._box_end = None
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            return self._finish(context, release_workspace=True)

        if event.type == 'TIMER':
            return {'PASS_THROUGH'}

        # Default interaction inside mini window:
        # point click selects / drags, empty drag creates a box selection.
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS' and inside:
            idx = self._pick_point(mx, my)

            if idx is not None:
                if event.shift:
                    if idx in self._selected:
                        self._selected.remove(idx)
                        context.area.tag_redraw()
                        return {'RUNNING_MODAL'}
                    else:
                        self._selected.add(idx)
                else:
                    if idx not in self._selected:
                        self._selected = {idx}

                if self._selected:
                    self._start_drag(context, mx, my)

                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            # Empty click -> box selection by default.
            self._box_dragging = True
            self._box_start = Vector((mx, my))
            self._box_end = Vector((mx, my))
            self._box_additive = bool(event.shift)
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'MOUSEMOVE' and self._dragging:
            self._apply_drag(context, mx, my)
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and self._dragging:
            self._apply_drag(context, mx, my)
            self._dragging = False
            self._direct_follow_state = None
            return {'RUNNING_MODAL'}

        if event.type == 'MOUSEMOVE' and self._box_dragging:
            self._box_end = Vector((mx, my))
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and self._box_dragging:
            self._box_end = Vector((mx, my))
            self._apply_box_select()
            self._box_dragging = False
            self._box_start = None
            self._box_end = None
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if inside or inside_b:
            return {'RUNNING_MODAL'}

        return {'PASS_THROUGH'}

    def _draw_panel_text(self, text, x, width=None):
        blf.enable(0, blf.CLIPPING)
        blf.clipping(0, x, self._panel_y, x + (width if width is not None else self._panel_w - 24), self._panel_y + self._panel_h)
        try:
            blf.draw(0, text)
        finally:
            blf.disable(0, blf.CLIPPING)

    def _draw_rect(self, x, y, w, h, color):
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')

        batch = batch_for_shader(
            shader,
            'TRI_FAN',
            {"pos": [
                (x, y),
                (x + w, y),
                (x + w, y + h),
                (x, y + h)
            ]}
        )

        shader.bind()
        shader.uniform_float("color", color)
        batch.draw(shader)

    def _draw_rect_outline(self, x1, y1, x2, y2, color):
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')

        pts = [
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2),
            (x1, y1)
        ]

        batch = batch_for_shader(
            shader,
            'LINE_STRIP',
            {"pos": pts}
        )

        gpu.state.line_width_set(1.5)

        shader.bind()
        shader.uniform_float("color", color)
        batch.draw(shader)

    def _draw_line(self, points, color, width=2.0, closed=False):
        if len(points) < 2:
            return

        pts = [
            (p.x, p.y)
            for p in points
        ]

        if closed:
            pts.append(pts[0])

        shader = gpu.shader.from_builtin('UNIFORM_COLOR')

        batch = batch_for_shader(
            shader,
            'LINE_STRIP',
            {"pos": pts}
        )

        gpu.state.line_width_set(width)

        shader.bind()
        shader.uniform_float("color", color)
        batch.draw(shader)

    def _draw_points(self, points, color, size=7.0):
        if not points:
            return

        shader = gpu.shader.from_builtin('UNIFORM_COLOR')

        batch = batch_for_shader(
            shader,
            'POINTS',
            {"pos": [
                (p.x, p.y)
                for p in points
            ]}
        )

        gpu.state.point_size_set(size)

        shader.bind()
        shader.uniform_float("color", color)
        batch.draw(shader)

    def _hp_draw_follow_ranges(self, context):
        """Show the actual affected quad spans over the mesh in the 3D view."""
        if (not getattr(self, '_hp_visual_spans', None)
                or not self._bmesh_ready() or context.region_data is None):
            return
        settings = getattr(self, '_hp_settings', {})
        if not settings.get('direct_follow', False):
            return
        palette = ((0.15, 0.9, 1.0), (1.0, 0.35, 0.8))
        mw = self._obj.matrix_world
        for side_index, spans in enumerate(self._hp_visual_spans):
            strength = max(0.0, min(1.0, float(settings.get(
                'follow_strength_a' if side_index == 0 else 'follow_strength_b', 1.0))))
            if strength <= 0.0:
                continue
            count = len(spans)
            for depth, span in enumerate(spans, 1):
                weight = strength * ((count - depth + 1.0) / count) ** float(
                    settings.get('falloff', 1.0))
                projected = {}
                vertices = {}
                for source, vi in span:
                    try:
                        projected[source] = location_3d_to_region_2d(
                            context.region, context.region_data, mw @ self._bm.verts[vi].co)
                        vertices[source] = vi
                    except (ReferenceError, RuntimeError, IndexError):
                        return
                color = (*palette[side_index], 0.3 + 0.65 * weight)
                for source, point in projected.items():
                    next_source = (source + 1) % len(self._ordered)
                    neighbor = projected.get(next_source)
                    pair = tuple(sorted((vertices[source], vertices.get(next_source, -1))))
                    if (point is not None and neighbor is not None
                            and pair in self._hp_visual_edges):
                        self._draw_line((point, neighbor), color, 2.0)
                visible = [point for point in projected.values() if point is not None]
                self._draw_points(visible, color, 5.0)
                if visible:
                    blf.size(0, 12)
                    blf.color(0, *palette[side_index], 1.0)
                    blf.position(0, visible[0].x + 6, visible[0].y + 5, 0)
                    blf.draw(0, f"{'A' if side_index == 0 else 'B'}{depth} {weight * 100:.0f}%")

    def _frame_preview(self, context):
        if not self._ordered:
            return
        points = [self._obj.matrix_world @ self._bm.verts[vi].co for vi in self._ordered]
        center = sum(points, Vector()) / len(points)
        rv3d = context.space_data.region_3d
        rv3d.view_location = center
        bottom = self._panel_y + self._panel_h + 24
        available = max(180, context.region.height - bottom - 40)
        rv3d.view_distance = max(0.1, max((p - center).length for p in points) * 2.2 * context.region.height / available)
        rv3d.update()
        # Frame the chain in the usable 3D area above the two panels.
        from bpy_extras.view3d_utils import region_2d_to_location_3d
        target = Vector((context.region.width * 0.5, bottom + available * 0.5))
        at_target = region_2d_to_location_3d(context.region, rv3d, target, center)
        rv3d.view_location += center - at_target
        rv3d.update()

    def _draw_point_labels(self):
        context = bpy.context
        if (getattr(self, '_finished', False) or self._idle
                or context.area is None or context.area.type != 'VIEW_3D'
                or not self._target_available(context)):
            return
        blf.size(0, 13)
        blf.color(0, 1.0, 0.65, 0.15, 1.0)
        for i in sorted(self._selected):
            if i >= len(self._ordered):
                continue
            world = self._obj.matrix_world @ self._bm.verts[self._ordered[i]].co
            screen = location_3d_to_region_2d(context.region, context.space_data.region_3d, world)
            if screen is not None:
                blf.position(0, screen.x + 9, screen.y + 9, 0)
                blf.draw(0, str(i + 1))
        if self._owner_context_active(context):
            for points in (self._panel_points, self._secondary_panel_points()):
                for i in sorted(self._selected):
                    if i < len(points):
                        blf.position(0, points[i].x + 9, points[i].y + 9, 0)
                        blf.draw(0, str(i + 1))

    def _draw_follow_3d(self):
        context = bpy.context
        if (getattr(self, '_finished', False) or self._idle
                or not self._target_available(context)
                or context.area is None or context.area.type != 'VIEW_3D'):
            return
        owner = self._owner_context_active(context)
        # Main scene views receive the same selected-point markers; the detached
        # preview additionally shows the entire active section, without a clone mesh.
        positions = [self._obj.matrix_world @ self._bm.verts[vi].co for vi in self._ordered]
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        old_depth = gpu.state.depth_test_get()
        old_blend = gpu.state.blend_get()
        old_size = 1.0
        old_width = gpu.state.line_width_get()
        try:
            gpu.state.depth_test_set('NONE')
            shader.bind()
            if self.detached and owner:
                shader.uniform_float('color', (0.3, 0.7, 1.0, 1.0))
                gpu.state.point_size_set(5.0)
                batch_for_shader(shader, 'POINTS', {'pos': positions}).draw(shader)
                lines = list(positions) + ([positions[0]] if self._closed else [])
                gpu.state.line_width_set(2.0)
                batch_for_shader(shader, 'LINE_STRIP', {'pos': lines}).draw(shader)
            selected = [positions[i] for i in sorted(self._selected) if i < len(positions)]
            if selected:
                shader.uniform_float('color', (1.0, 0.55, 0.08, 1.0))
                gpu.state.point_size_set(12.0)
                batch_for_shader(shader, 'POINTS', {'pos': selected}).draw(shader)
        finally:
            gpu.state.depth_test_set(old_depth)
            gpu.state.blend_set(old_blend)
            gpu.state.point_size_set(old_size)
            gpu.state.line_width_set(old_width)
        if not owner:
            return
        settings = getattr(self, '_hp_settings', {})
        if not settings.get('direct_follow', False):
            return
        spans_by_side = getattr(self, '_hp_visual_spans', [[], []])
        if not any(spans_by_side):
            return
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        palette = ((0.08, 0.95, 1.0), (1.0, 0.18, 0.72))
        mw = self._obj.matrix_world
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')
        try:
            for side_index, spans in enumerate(spans_by_side):
                strength = max(0.0, min(1.0, float(settings.get(
                    'follow_strength_a' if side_index == 0 else 'follow_strength_b', 1.0))))
                if strength <= 0.0:
                    continue
                count = len(spans)
                for depth, span in enumerate(spans, 1):
                    weight = strength * ((count - depth + 1.0) / count) ** float(
                        settings.get('falloff', 1.0))
                    positions = {}
                    vertices = {}
                    try:
                        for source, vi in span:
                            positions[source] = mw @ self._bm.verts[vi].co
                            vertices[source] = vi
                    except (ReferenceError, RuntimeError, IndexError):
                        return
                    line_positions = []
                    for source, position in positions.items():
                        next_source = (source + 1) % len(self._ordered)
                        neighbor = positions.get(next_source)
                        edge = tuple(sorted((vertices[source], vertices.get(next_source, -1))))
                        if neighbor is not None and edge in self._hp_visual_edges:
                            line_positions.extend((position, neighbor))
                    color = (*palette[side_index], 0.45 + 0.55 * weight)
                    shader.bind()
                    shader.uniform_float('color', color)
                    if line_positions:
                        gpu.state.line_width_set(4.0)
                        batch_for_shader(shader, 'LINES', {'pos': line_positions}).draw(shader)
                    if positions:
                        gpu.state.point_size_set(9.0)
                        batch_for_shader(shader, 'POINTS', {'pos': list(positions.values())}).draw(shader)
        finally:
            gpu.state.depth_test_set('NONE')
            gpu.state.blend_set('NONE')

    def _draw_circle(self, center, radius, color, segments=64):
        pts = []

        for i in range(segments + 1):
            a = tau * i / segments

            pts.append(Vector((
                center.x + cos(a) * radius,
                center.y + sin(a) * radius
            )))

        self._draw_line(
            pts,
            color,
            1.0,
            False
        )

    def _world_axis_to_panel_dir(self, world_axis):
        """
        Project a WORLD axis into the current section editor plane.
        Returns a normalized 2D panel direction.
        """
        if self._obj is None:
            return None

        if self._primary_view == 'SECTION':
            # Original arbitrary section plane lives in object-local space.
            local_axis = (
                self._obj.matrix_world.to_3x3().inverted()
                @ world_axis
            )

            panel_dir = Vector((
                local_axis.dot(self._u),
                local_axis.dot(self._v),
            ))
        else:
            panel_dir = Vector((
                world_axis.dot(self._primary_world_u),
                world_axis.dot(self._primary_world_v),
            ))

        if self._primary_view == 'SECTION':
            panel_dir = self._rotate_2d(
                panel_dir,
                self._section_rotation
            )

        if self._flip_x:
            panel_dir.x *= -1.0

        if self._flip_y:
            panel_dir.y *= -1.0

        if panel_dir.length < 1e-6:
            return None

        panel_dir.normalize()
        return panel_dir

    def _draw_axis_arrow(self, origin, direction, label, color):
        if direction is None:
            return

        length = 34.0
        end = origin + direction * length

        self._draw_line(
            [origin, end],
            color,
            2.5,
            False
        )

        # Simple arrow head.
        perp = Vector((-direction.y, direction.x))
        head_back = end - direction * 8.0

        self._draw_line(
            [
                end,
                head_back + perp * 4.0
            ],
            color,
            2.0,
            False
        )

        self._draw_line(
            [
                end,
                head_back - perp * 4.0
            ],
            color,
            2.0,
            False
        )

        blf.position(
            0,
            end.x + direction.x * 5.0 - 3.0,
            end.y + direction.y * 5.0 - 3.0,
            0
        )

        blf.size(0, 12)
        blf.color(
            0,
            color[0],
            color[1],
            color[2],
            1.0
        )
        blf.draw(0, label)

    def _draw_axis_indicator(self):
        # Keep it above the bottom help text and inside the right edge.
        origin = Vector((
            self._panel_x + self._panel_w - 76.0,
            self._panel_y + 88.0
        ))

        x_dir = self._world_axis_to_panel_dir(
            Vector((1.0, 0.0, 0.0))
        )

        y_dir = self._world_axis_to_panel_dir(
            Vector((0.0, 1.0, 0.0))
        )

        self._draw_axis_arrow(
            origin,
            x_dir,
            "X",
            (1.0, 0.32, 0.32, 0.95)
        )

        self._draw_axis_arrow(
            origin,
            y_dir,
            "Y",
            (0.35, 1.0, 0.42, 0.95)
        )

        # Small pivot dot.
        self._draw_points(
            [origin],
            (0.95, 0.95, 0.95, 0.95),
            5.0
        )

    def _draw_secondary_axis_indicator(self):
        if self._view_b_collapsed:
            return

        sx = self._secondary_x()
        origin = Vector((
            sx + self._panel_w - 76.0,
            self._panel_y + 88.0
        ))

        x_color = (1.0, 0.32, 0.32, 0.95)
        y_color = (0.35, 1.0, 0.42, 0.95)
        z_color = (0.38, 0.62, 1.0, 0.95)

        if self._secondary_view == 'FRONT':
            h_dir = Vector((1.0, 0.0))
            h_label = 'X'
            h_color = x_color

        elif self._secondary_view == 'BACK':
            h_dir = Vector((-1.0, 0.0))
            h_label = 'X'
            h_color = x_color

        elif self._secondary_view == 'RIGHT':
            h_dir = Vector((1.0, 0.0))
            h_label = 'Y'
            h_color = y_color

        else:  # LEFT
            h_dir = Vector((-1.0, 0.0))
            h_label = 'Y'
            h_color = y_color

        self._draw_axis_arrow(
            origin,
            h_dir,
            h_label,
            h_color
        )

        z_dir = Vector((
            0.0,
            -1.0 if self._secondary_flip_y else 1.0
        ))

        self._draw_axis_arrow(
            origin,
            z_dir,
            'Z',
            z_color
        )

        self._draw_points(
            [origin],
            (0.95, 0.95, 0.95, 0.95),
            5.0
        )

    def _draw_secondary_view(self):
        # BLF/text and other GPU drawing earlier in View A can alter GPU state.
        # Reassert alpha blending here so View B uses the same transparency
        # semantics as View A instead of occasionally becoming opaque black.
        gpu.state.blend_set('ALPHA')

        sx = self._secondary_x()
        h = self._secondary_height()

        self._draw_rect(
            sx,
            self._panel_y,
            self._panel_w,
            h,
            (0.025, 0.025, 0.025, 0.62)
        )

        self._draw_rect(
            sx + 1,
            self._panel_y + h - 34,
            self._panel_w - 2,
            33,
            (0.085, 0.085, 0.085, 0.86)
        )

        self._draw_rect_outline(
            sx,
            self._panel_y,
            sx + self._panel_w,
            self._panel_y + h,
            (0.65, 0.65, 0.65, 0.42)
        )

        blf.position(
            0,
            sx + 12,
            self._panel_y + h - 23,
            0
        )
        blf.size(0, 14)
        blf.color(0, 1.0, 1.0, 1.0, 1.0)

        prefix = '[+]' if self._view_b_collapsed else '[-]'
        xray_text = "ON" if self._xray else "OFF"

        shortcut_text = self._secondary_view_shortcut_label()

        self._draw_panel_text(
            f"{prefix} B : {self._secondary_view} [{shortcut_text}] X-Ray:{xray_text}", sx + 12
        )

        if self._view_b_collapsed:
            return

        points = self._secondary_panel_points()

        self._draw_line(
            points,
            (0.50, 0.76, 1.0, 1.0),
            2.0,
            self._closed
        )

        all_indices = list(range(len(points)))
        front_indices = set(
            self._secondary_front_filter(
                all_indices,
                points
            )
        )

        if self._xray:
            front_indices = set(all_indices)

        rear_unselected = [
            points[i]
            for i in all_indices
            if i not in front_indices
            and i not in self._selected
        ]

        front_unselected = [
            points[i]
            for i in all_indices
            if i in front_indices
            and i not in self._selected
        ]

        selected = [
            points[i]
            for i in all_indices
            if i in self._selected
        ]

        if rear_unselected:
            self._draw_points(
                rear_unselected,
                (0.5, 0.76, 1.0, 0.28),
                5.0
            )

        self._draw_points(
            front_unselected,
            (0.5, 0.76, 1.0, 1.0),
            7.0
        )

        self._draw_points(
            selected,
            (1.0, 0.55, 0.08, 1.0),
            10.0
        )

        if (
            self._prop_enabled
            and self._selected
        ):
            prop_points = [
                points[i]
                for i in sorted(self._selected)
                if 0 <= i < len(points)
            ]

            if prop_points:
                prop_center = (
                    sum(
                        prop_points,
                        Vector((0.0, 0.0))
                    )
                    / len(prop_points)
                )

                self._draw_circle(
                    prop_center,
                    self._prop_radius_px,
                    (0.55, 0.9, 1.0, 0.42)
                )

        if (
            self._secondary_box_dragging
            and self._secondary_box_start is not None
            and self._secondary_box_end is not None
        ):
            self._draw_rect_outline(
                self._secondary_box_start.x,
                self._secondary_box_start.y,
                self._secondary_box_end.x,
                self._secondary_box_end.y,
                (0.3, 0.8, 1.0, 0.95)
            )

        if self._secondary_pen_stroke and getattr(self, '_hp_menu', None) is None:
            order, full_loop = self._secondary_pen_selected_order()

            pen_closed = bool(
                full_loop
                and len(self._secondary_pen_stroke) >= 3
                and (
                    self._secondary_pen_stroke[-1]
                    - self._secondary_pen_stroke[0]
                ).length <= 20.0
            )

            pen_preview = _stabilize_2d(
                self._secondary_pen_stroke,
                level=self._secondary_pen_stabilizer,
                closed=pen_closed
            )

            if order is not None and order:
                current_points = self._secondary_panel_points()
                target_panel = [
                    current_points[i]
                    for i in order
                ]

                pen_preview = self._prepare_pen_stroke(
                    pen_preview,
                    target_panel,
                    pen_closed
                )

            self._draw_line(
                pen_preview,
                (1.0, 0.42, 0.08, 1.0),
                3.0,
                pen_closed
            )

        self._draw_secondary_axis_indicator()

        blf.position(
            0,
            sx + 12,
            self._panel_y + 12,
            0
        )
        blf.size(0, 11)
        blf.color(
            0,
            0.80,
            0.80,
            0.80,
            1.0
        )
        self._draw_panel_text("Num1 正面 / Num3 側面 / F ペン", sx + 12)

    def _draw_smooth_hud(self):
        if not (
            self._pen_smooth_dragging
            or self._smooth_dragging
        ):
            return

        amount = (
            self._pen_smooth_amount
            if self._pen_smooth_dragging
            else self._smooth_amount
        )

        if self._pen_smooth_dragging:
            radius = self._pen_smooth_brush_radius
            strength = self._pen_smooth_brush_strength
        else:
            radius = self._smooth_brush_radius
            strength = self._smooth_brush_strength

        mode = 'Pen line' if self._pen_smooth_dragging else 'Vertices'
        label = f"{mode}  Str {strength:.2f}  R {radius:.0f}"

        self._draw_circle(
            Vector((self._smooth_hud_x, self._smooth_hud_y)),
            radius,
            (0.25, 0.95, 0.70, 0.85),
        )

        x = self._smooth_hud_x + 18
        y = self._smooth_hud_y + 18

        # Backplate
        self._draw_rect(
            x - 7,
            y - 7,
            210,
            28,
            (0.02, 0.02, 0.02, 0.82)
        )

        blf.position(
            0,
            x,
            y,
            0
        )
        blf.size(
            0,
            13
        )
        blf.color(
            0,
            0.95,
            0.95,
            0.95,
            1.0
        )
        blf.draw(
            0,
            label
        )

    def _draw(self):
        if getattr(self, "_finished", False):
            return

        if not self._owner_context_active(
            bpy.context
        ):
            return

        if self._idle or not self._target_available(bpy.context):
            blf.size(0, 16)
            blf.color(0, 0.8, 0.9, 1.0, 1.0)
            blf.position(0, 24, 50, 0)
            blf.draw(0, "HP Section: select a connected edge loop in the main view")
            return

        gpu.state.blend_set('ALPHA')

        self._draw_preview_hud()
        self._hp_draw_follow_ranges(bpy.context)

        a_h = self._primary_height()

        self._draw_rect(
            self._panel_x,
            self._panel_y,
            self._panel_w,
            a_h,
            (0.025, 0.025, 0.025, 0.62)
        )

        self._draw_rect(
            self._panel_x + 1,
            self._panel_y + a_h - 34,
            self._panel_w - 2,
            33,
            (0.085, 0.085, 0.085, 0.86)
        )

        self._draw_rect_outline(
            self._panel_x,
            self._panel_y,
            self._panel_x + self._panel_w,
            self._panel_y + a_h,
            (0.65, 0.65, 0.65, 0.42)
        )

        blf.position(
            0,
            self._panel_x + 12,
            self._panel_y + a_h - 23,
            0
        )
        blf.size(0, 14)
        blf.color(0, 1.0, 1.0, 1.0, 1.0)

        a_prefix = '[+]' if self._view_a_collapsed else '[-]'
        primary_shortcut = self._primary_view_shortcut_label()
        xray_text = "ON" if self._xray else "OFF"
        self._draw_panel_text(
            f"{a_prefix} A : {self._primary_view} [{primary_shortcut}] X-Ray:{xray_text}", self._panel_x + 12
        )

        if not self._view_a_collapsed:
            self._draw_line(
                self._panel_points,
                (0.50, 0.76, 1.0, 1.0),
                2.0,
                self._closed
            )

            all_indices = list(
                range(len(self._panel_points))
            )
            front_indices = set(
                self._primary_front_filter(
                    all_indices,
                    self._panel_points
                )
            )

            rear_unselected = [
                self._panel_points[i]
                for i in all_indices
                if (
                    i not in front_indices
                    and i not in self._selected
                )
            ]

            front_unselected = [
                self._panel_points[i]
                for i in all_indices
                if (
                    i in front_indices
                    and i not in self._selected
                )
            ]

            selected = [
                self._panel_points[i]
                for i in all_indices
                if i in self._selected
            ]

            if rear_unselected:
                self._draw_points(
                    rear_unselected,
                    (0.5, 0.76, 1.0, 0.28),
                    5.0
                )

            self._draw_points(
                front_unselected,
                (0.5, 0.76, 1.0, 1.0),
                7.0
            )

            self._draw_points(
                selected,
                (1.0, 0.55, 0.08, 1.0),
                10.0
            )

            if (
                self._prop_enabled
                and self._selected
            ):
                prop_points = [
                    self._panel_points[i]
                    for i in sorted(self._selected)
                    if 0 <= i < len(self._panel_points)
                ]

                if prop_points:
                    prop_center = (
                        sum(
                            prop_points,
                            Vector((0.0, 0.0))
                        )
                        / len(prop_points)
                    )

                    self._draw_circle(
                        prop_center,
                        self._prop_radius_px,
                        (0.55, 0.9, 1.0, 0.42)
                    )

            if (
                self._box_dragging
                and self._box_start is not None
                and self._box_end is not None
            ):
                self._draw_rect_outline(
                    self._box_start.x,
                    self._box_start.y,
                    self._box_end.x,
                    self._box_end.y,
                    (0.3, 0.8, 1.0, 0.95)
                )

            if self._pen_stroke and getattr(self, '_hp_menu', None) is None:
                pen_order, pen_full_loop = self._pen_selected_order()
                pen_closed = bool(
                    pen_full_loop
                    and self._pen_stroke_is_closed()
                )

                pen_preview = _stabilize_2d(
                    self._pen_stroke,
                    level=self._pen_stabilizer,
                    closed=pen_closed
                )

                if pen_order is not None and pen_order:
                    target_panel = [
                        self._panel_points[i]
                        for i in pen_order
                    ]

                    pen_preview = self._prepare_pen_stroke(
                        pen_preview,
                        target_panel,
                        pen_closed
                    )

                self._draw_line(
                    pen_preview,
                    (1.0, 0.42, 0.08, 1.0),
                    3.0,
                    pen_closed
                )

            self._draw_axis_indicator()

            lock = "".join(
                c for c, b in (
                    ("X", self._lock_x),
                    ("Y", self._lock_y),
                    ("Z", self._lock_z)
                )
                if b
            ) or "-"

            mode = "ON" if self._prop_enabled else "OFF"

            blf.position(
                0,
                self._panel_x + 12,
                self._panel_y + 12,
                0
            )

            blf.size(0, 11)
            blf.color(
                0,
                0.80,
                0.80,
                0.80,
                1.0
            )

            self._draw_panel_text("F ペン / E ブラシ / Ctrl+F 整形", self._panel_x + 12)

            blf.position(
                0,
                self._panel_x + 12,
                self._panel_y + 29,
                0
            )

            self._draw_panel_text(f"選択:{len(self._selected)}  補正:{self._pen_stabilizer}  +/- サイズ", self._panel_x + 12)

        if not self._view_a_collapsed:
            settings = getattr(self, '_hp_settings', {})
            enabled = bool(settings.get('direct_follow', False))
            labels = (
                ('toggle', '追従 ON' if enabled else '追従 OFF'),
                ('A', f"A {float(settings.get('follow_strength_a', 1.0)) * 100:.0f}%"),
                ('B', f"B {float(settings.get('follow_strength_b', 1.0)) * 100:.0f}%"),
                ('log', 'ログコピー'),
            )
            left = self._panel_x + 8
            cell = (self._panel_w - 16) / 4
            top = self._panel_y + self._panel_h - 39
            for index, (key, label) in enumerate(labels):
                bx = left + index * cell
                active = enabled and (key == 'toggle' or (
                    key in {'A', 'B'} and int(settings.get(
                        'side_a' if key == 'A' else 'side_b', 0)) > 0))
                self._draw_rect(
                    bx, top - 28, cell - 2, 28,
                    (0.12, 0.40, 0.32, 0.96) if active else (0.14, 0.16, 0.19, 0.96),
                )
                blf.size(0, 12)
                blf.color(0, 1.0, 1.0, 1.0, 1.0)
                blf.position(0, bx + 7, top - 19, 0)
                self._draw_panel_text(label, bx + 7, width=cell - 12)

        self._draw_secondary_view()

        self._draw_smooth_hud()

        self._hp_draw_menu(bpy.context)

        gpu.state.blend_set('NONE')

    def _finish(self, context, release_workspace=False):
        global _RUNNING, _ACTIVE_SECTION_EDITOR, _PINNED_WORKSPACE_PTR

        # Idempotent: unregister, Blender cancellation, and the final modal
        # event can all reach this method. A stale instance must not stop a
        # newly started editor.
        if getattr(self, '_finished', False):
            return {'FINISHED'}
        if getattr(self, '_hp_menu', None) is not None:
            try:
                self._hp_end_adjust(context, commit=False)
            except Exception:
                self._hp_menu = None
        self._hp_confirm_hold = None

        if getattr(self, '_preview_state', None):
            self._preview_finish_edit(context, cancel=True)
        self._finished = True

        labels = getattr(self, '_handle_labels', None)
        self._handle_labels = None
        if labels is not None:
            bpy.types.SpaceView3D.draw_handler_remove(labels, 'WINDOW')
        try:
            space = getattr(self, '_preview_space', None)
            if space is not None:
                for key, value in self._preview_filter_backup.items():
                    setattr(space, key, value)
                space.overlay.show_overlays = self._preview_overlay_backup
        except (ReferenceError, RuntimeError):
            pass

        handle = getattr(self, '_handle', None)
        self._handle = None
        if handle is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
            except (ReferenceError, RuntimeError, ValueError):
                pass  # Blender may already have removed the handler.

        follow_handle = getattr(self, '_handle_follow', None)
        self._handle_follow = None
        if follow_handle is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(follow_handle, 'WINDOW')
            except (ReferenceError, RuntimeError, ValueError):
                pass

        timer = getattr(self, '_timer', None)
        self._timer = None
        wm = getattr(self, '_timer_owner', None)
        if wm is None:
            wm = getattr(context, 'window_manager', None)
        if timer is not None and wm is not None:
            try:
                wm.event_timer_remove(timer)
            except (ReferenceError, RuntimeError, ValueError):
                pass
        self._timer_owner = None

        if _ACTIVE_SECTION_EDITOR is self:
            _ACTIVE_SECTION_EDITOR = None
            _RUNNING = False
            if release_workspace:
                _PINNED_WORKSPACE_PTR = 0

        # Disable may be invoked from Preferences, not the owning View3D.
        if wm is not None:
            try:
                for window in wm.windows:
                    for area in window.screen.areas:
                        if area.type == 'VIEW_3D':
                            area.tag_redraw()
            except (ReferenceError, RuntimeError):
                pass
        return {'FINISHED'}

    def cancel(self, context):
        self._finish(context, release_workspace=True)


def _view_context(window_ptr, area_ptr):
    for window in bpy.context.window_manager.windows:
        if window.as_pointer() != window_ptr:
            continue
        area = next((a for a in window.screen.areas
                     if a.as_pointer() == area_ptr and a.type == 'VIEW_3D'), None)
        if area is not None:
            region = next((r for r in area.regions if r.type == 'WINDOW'), None)
            if region is not None:
                return window, area, region
    return None


def _section_preferences(context):
    addon = context.preferences.addons.get(__package__)
    return addon.preferences if addon is not None else None


def _session_after_finish(editor, context):
    editor._finish(context, release_workspace=True)
    fields = ('_selected', '_hp_settings', '_history', '_topology_undo_steps',
              '_topology_selection_history', '_primary_view', '_secondary_view',
              '_view_zoom', '_secondary_zoom', '_panel_stage', '_xray',
              '_view_a_collapsed', '_view_b_collapsed', '_lock_x', '_lock_y', '_lock_z',
              '_prop_enabled', '_prop_radius_px', '_pen_anchor_mode',
              '_pen_stabilizer', '_secondary_pen_stabilizer',
              '_smooth_brush_radius', '_smooth_brush_strength', '_flip_x', '_flip_y',
              '_secondary_flip_y', '_secondary_plane')
    return dict(obj=editor._obj, signature=editor._signature,
                values={key:copy.copy(getattr(editor,key)) for key in fields})


def _restore_session(editor, session, context):
    if session is None or editor is None:
        return
    same_target = editor._obj == session['obj'] and editor._signature == session['signature']
    target_fields = {'_selected','_history','_topology_undo_steps','_topology_selection_history'}
    for key,value in session['values'].items():
        if same_target or key not in target_fields:
            setattr(editor,key,value)
    editor._layout_panels(context)
    editor._rebuild(context)
    if editor.detached:
        editor._frame_preview(context)


def _close_owned_window_later(window_ptr):
    owned = next((w for w in bpy.context.window_manager.windows if w.as_pointer() == window_ptr),None)
    def close():
        try:
            if owned is None or owned.as_pointer() != window_ptr or owned not in bpy.context.window_manager.windows[:]:
                return None
        except ReferenceError:
            return None
        if _ACTIVE_SECTION_EDITOR is not None and _ACTIVE_SECTION_EDITOR._owner_window_ptr == window_ptr:
            return None
        window = next((w for w in bpy.context.window_manager.windows if w.as_pointer() == window_ptr),None)
        if window is not None and len(bpy.context.window_manager.windows)>1:
            with bpy.context.temp_override(window=window):
                bpy.ops.wm.window_close()
        return None
    bpy.app.timers.register(close, first_interval=0.1)


class HP_OT_section_inline(bpy.types.Operator):
    bl_idname = "hp.section_inline"
    bl_label = "HP Section: ビュー内に切り替え"
    bl_description = "断面エディターを閉じる操作なしで元の3Dビューに切り替えます"

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == 'VIEW_3D'

    def execute(self, context):
        editor = _ACTIVE_SECTION_EDITOR
        if editor is not None and not editor.detached and editor._owner_context_active(context):
            return {'FINISHED'}
        home = None
        old_window = None
        session = None
        if editor is not None:
            if editor.detached:
                home = _view_context(getattr(editor,'_home_window_ptr',0),getattr(editor,'_home_area_ptr',0))
                old_window = editor._owner_window_ptr
            session = _session_after_finish(editor,context)
        home = home or (context.window,context.area,next(r for r in context.area.regions if r.type == 'WINDOW'))
        with context.temp_override(window=home[0],area=home[1],region=home[2]):
            result = bpy.ops.hp.section_mini_editor('INVOKE_DEFAULT',detached=False)
            if result == {'RUNNING_MODAL'}:
                _restore_session(_ACTIVE_SECTION_EDITOR,session,bpy.context)
        if old_window is not None and result == {'RUNNING_MODAL'} and old_window != home[0].as_pointer():
            _close_owned_window_later(old_window)
        return {'FINISHED'} if result == {'RUNNING_MODAL'} else {'CANCELLED'}


class HP_OT_section_window(bpy.types.Operator):
    bl_idname = "hp.section_window"
    bl_label = "HP Section: 別ウィンドウに切り替え"
    bl_description = "断面エディターを別ウィンドウに切り替え、Windowsでは別モニターに配置します"

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == 'VIEW_3D'

    def execute(self, context):
        editor = _ACTIVE_SECTION_EDITOR
        if editor is not None and editor.detached:
            if _view_context(editor._owner_window_ptr,editor._owner_area_ptr) is not None:
                return {'FINISHED'}
            editor._finish(context,release_workspace=True)
            editor = None
        session = _session_after_finish(editor,context) if editor is not None else None
        home = (context.window.as_pointer(),context.area.as_pointer())
        preferences = _section_preferences(context)
        auto_move = preferences is None or preferences.move_to_other_monitor
        native = HP_Window_Placement.snapshot_windows() if auto_move else None
        previous = {w.as_pointer() for w in context.window_manager.windows}
        result = bpy.ops.screen.area_dupli('INVOKE_DEFAULT')
        if result != {'FINISHED'}:
            # Recover the original mode if native window creation failed.
            bpy.ops.hp.section_mini_editor('INVOKE_DEFAULT',detached=False)
            _restore_session(_ACTIVE_SECTION_EDITOR,session,context)
            return {'CANCELLED'}
        window = next((w for w in context.window_manager.windows
                       if w.as_pointer() not in previous), None)
        if window is None:
            self.report({'ERROR'}, "別ウィンドウを作成できませんでした")
            return {'CANCELLED'}
        area = next(a for a in window.screen.areas if a.type == 'VIEW_3D')
        region = next(r for r in area.regions if r.type == 'WINDOW')
        with context.temp_override(window=window, area=area, region=region):
            result = bpy.ops.hp.section_mini_editor('INVOKE_DEFAULT', detached=True)
            if result == {'RUNNING_MODAL'}:
                new = _ACTIVE_SECTION_EDITOR
                new._home_window_ptr, new._home_area_ptr = home
                _restore_session(new,session,bpy.context)
        if result == {'RUNNING_MODAL'} and auto_move and native is not None:
            owner = window.as_pointer()
            HP_Window_Placement.schedule_other_monitor(native,
                lambda: _ACTIVE_SECTION_EDITOR is not None and _ACTIVE_SECTION_EDITOR.detached
                and _ACTIVE_SECTION_EDITOR._owner_window_ptr == owner)
        return {'FINISHED'} if result == {'RUNNING_MODAL'} else {'CANCELLED'}


class HP_MT_section_display(bpy.types.Menu):
    bl_label = "HP 断面エディター"
    bl_idname = "HP_MT_section_display"

    def draw(self, context):
        self.layout.operator('hp.section_inline',text="ビュー内に切り替え",icon='VIEW3D')
        self.layout.operator('hp.section_window',text="別ウィンドウに切り替え",icon='WINDOW')
        self.layout.separator()
        row=self.layout.row()
        row.enabled=_ACTIVE_SECTION_EDITOR is not None
        row.operator('hp.section_close',text="断面エディターを閉じる",icon='X')


def _draw_section_context(self, context):
    self.layout.menu('HP_MT_section_display',icon='WINDOW')
    self.layout.separator()


_CONTEXT_MENUS = ('VIEW3D_MT_edit_mesh_context_menu','VIEW3D_MT_object_context_menu')


class HP_OT_section_close(bpy.types.Operator):
    bl_idname = "hp.section_close"
    bl_label = "断面エディターを閉じる"

    def execute(self, context):
        editor = _ACTIVE_SECTION_EDITOR
        if editor is not None:
            owner = editor._owner_window_ptr if editor.detached else None
            editor._finish(context, release_workspace=True)
            if owner is not None:
                _close_owned_window_later(owner)
        return {'FINISHED'}


class HP_PT_section_tools(bpy.types.Panel):
    bl_label = "HP Section"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'HP Tools'

    @classmethod
    def poll(cls, context):
        preferences = _section_preferences(context)
        return preferences is not None and preferences.show_section_sidebar

    def draw(self, context):
        layout = self.layout
        layout.operator('hp.section_window', icon='WINDOW')
        layout.operator('hp.section_inline', text="ビュー内に切り替え")
        layout.operator('hp.section_close', icon='X')
        if context.edit_object is not None and context.edit_object.type == 'CURVE':
            layout.operator('hp.curve_mini_editor', text="カーブエディターを開く")
        layout.label(text="選択解除後も開いたままです")
        layout.label(text="3D表示: 中ボタンで回転")
        layout.label(text="点の番号・橙色は全ビュー共通")


def _watch_editor_window():
    editor = _ACTIVE_SECTION_EDITOR
    if editor is not None:
        alive = any(w.as_pointer() == editor._owner_window_ptr
                    and any(a.as_pointer() == editor._owner_area_ptr and a.type == 'VIEW_3D'
                            for a in w.screen.areas)
                    for w in bpy.context.window_manager.windows)
        if not alive:
            editor._finish(bpy.context, release_workspace=True)
    return 0.35



classes = (
    HP_OT_section_shape_choice,
    HP_MT_section_shape_pie,
    HP_OT_section_topology_choice,
    HP_MT_section_topology_pie,
    HP_OT_section_pen_anchor_choice,
    HP_MT_section_pen_anchor_pie,
    HP_OT_section_mini_editor,
    HP_OT_section_window,
    HP_OT_section_inline,
    HP_MT_section_display,
    HP_OT_section_close,
    HP_PT_section_tools,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    for name in _CONTEXT_MENUS:
        getattr(bpy.types,name).prepend(_draw_section_context)

    if not bpy.app.timers.is_registered(_watch_editor_window):
        bpy.app.timers.register(
            _watch_editor_window,
            first_interval=0.5,
            persistent=True
        )


def unregister():
    global _RUNNING, _ACTIVE_SECTION_EDITOR, _PINNED_WORKSPACE_PTR

    for name in _CONTEXT_MENUS:
        getattr(bpy.types,name).remove(_draw_section_context)

    if bpy.app.timers.is_registered(_watch_editor_window):
        bpy.app.timers.unregister(_watch_editor_window)

    editor = _ACTIVE_SECTION_EDITOR
    if editor is not None:
        editor._finish(bpy.context, release_workspace=True)
    _ACTIVE_SECTION_EDITOR = None
    _RUNNING = False
    _PINNED_WORKSPACE_PTR = 0

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()


bl_info = {
    "name": "HP Curve Mini Editor",
    "author": "OpenAI + yashi",
    "version": (1, 4, 0),
    "blender": (4, 3, 0),
    "location": "3D View > Sidebar > HP Tools",
    "description": "Single-window orthographic Curve editor with Blender-style numpad views.",
    "category": "Curve",
}

import bpy
import time
from bpy.props import BoolProperty, EnumProperty
import blf
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector, Quaternion
from mathutils.geometry import interpolate_bezier
from math import cos, sin, tau, atan2

_CURVE_RUNNING = False
_ACTIVE_CURVE_EDITOR = None
_CURVE_PINNED_WORKSPACE_PTR = 0
_CURVE_PEN_ANCHOR_DEFAULT = 'STROKE'
PEN_F_HOLD_SECONDS = 0.45

PANEL_Y = 18
PANEL_W = 640
PANEL_H = 480
PANEL_GAP = 12
SECTION_X = 43
SECTION_W = 320
PREFERRED_X = SECTION_X + SECTION_W + PANEL_GAP
PANEL_PAD = 34


def _curve_target(context):
    obj = context.edit_object

    if (
        obj is None
        or obj.type != 'CURVE'
        or obj.mode != 'EDIT'
    ):
        return None

    selected_splines = []

    for spline_index, spline in enumerate(obj.data.splines):
        if spline.type == 'BEZIER':
            selected = [
                i
                for i, bp in enumerate(spline.bezier_points)
                if bp.select_control_point
            ]
            count = len(spline.bezier_points)
        else:
            selected = [
                i
                for i, p in enumerate(spline.points)
                if p.select
            ]
            count = len(spline.points)

        if selected:
            selected_splines.append(
                (
                    spline_index,
                    spline,
                    selected,
                    count
                )
            )

    if len(selected_splines) != 1:
        return None

    spline_index, spline, selected, count = selected_splines[0]

    if count < 2:
        return None

    return {
        "obj": obj,
        "spline_index": spline_index,
        "spline": spline,
        "selected": set(selected),
        "count": count,
        "bezier": spline.type == 'BEZIER',
        "cyclic": bool(spline.use_cyclic_u),
        "signature": (
            obj.as_pointer(),
            spline_index,
            spline.type,
            count,
        ),
    }



def _curve_live_signature(
    context,
    obj,
    spline_index
):
    """Validate the currently edited spline without requiring selection."""
    if (
        obj is None
        or context.edit_object is None
        or context.edit_object != obj
        or obj.type != 'CURVE'
        or obj.mode != 'EDIT'
    ):
        return None

    try:
        if (
            spline_index < 0
            or spline_index >= len(obj.data.splines)
        ):
            return None

        spline = obj.data.splines[spline_index]

        count = (
            len(spline.bezier_points)
            if spline.type == 'BEZIER'
            else len(spline.points)
        )

        if count < 2:
            return None

        return (
            obj.as_pointer(),
            spline_index,
            spline.type,
            count,
        )
    except (ReferenceError, IndexError):
        return None


def _smooth_falloff(t):
    t = max(0.0, min(1.0, t))
    s = 3.0 * t * t - 2.0 * t * t * t
    return 1.0 - s


def _hp_brush_repel_2d(points, mouse, motion, radius=70.0, strength=0.55):
    from .HP_Section_MiniEditor import _hp_brush_repel_2d as swept_brush
    return swept_brush(points, mouse, motion, radius, strength)


def _resample_spacing(points, spacing=3.0, closed=False):
    pts = [Vector(p) for p in points]

    if len(pts) < 2:
        return pts

    if closed and (pts[-1] - pts[0]).length > 0.5:
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


def _moving_average(points, radius=2, passes=2, closed=False):
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

            dst.append(
                acc / max(total, 1e-8)
            )

        pts = dst

    return pts


def _chaikin(points, iterations=1, closed=False):
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


def _stabilize(points, level=2, closed=False):
    if len(points) < 2:
        return [Vector(p) for p in points]

    level = max(0, min(5, int(level)))

    if level == 0:
        return [Vector(p) for p in points]

    pts = _resample_spacing(
        points,
        spacing=max(2.0, 4.5 - level * 0.35),
        closed=closed
    )

    if level > 0:
        radius = 1 + level // 2
        passes = 1 + level // 2

        pts = _moving_average(
            pts,
            radius=radius,
            passes=passes,
            closed=closed
        )

    return _chaikin(
        pts,
        iterations=1,
        closed=closed
    )


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
            (
                pts[(i + 1) % len(pts)]
                - pts[i]
            ).length
            for i in range(len(pts))
        ]

        total = sum(segs)

        if total < 1e-8:
            return [
                i / len(pts)
                for i in range(len(pts))
            ]

        out = [0.0]
        run = 0.0

        for i in range(len(pts) - 1):
            run += segs[i]
            out.append(run / total)

        return out

    segs = [
        (b - a).length
        for a, b in zip(
            pts[:-1],
            pts[1:]
        )
    ]

    total = sum(segs)

    if total < 1e-8:
        return [
            i / (len(pts) - 1)
            for i in range(len(pts))
        ]

    out = [0.0]
    run = 0.0

    for length in segs:
        run += length
        out.append(run / total)

    return out


def _sample_polyline(points, t_values, closed=False):
    pts = [Vector(p) for p in points]

    if not pts:
        return []

    if len(pts) == 1:
        return [
            pts[0].copy()
            for _ in t_values
        ]

    work = (
        pts + [pts[0].copy()]
        if closed
        else pts
    )

    seg_lens = [
        (b - a).length
        for a, b in zip(
            work[:-1],
            work[1:]
        )
    ]

    total = sum(seg_lens)

    if total < 1e-8:
        return [
            work[0].copy()
            for _ in t_values
        ]

    cumulative = [0.0]
    run = 0.0

    for length in seg_lens:
        run += length
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

            if (
                a_t <= t <= b_t
                or (
                    i == len(cumulative) - 2
                    and t >= a_t
                )
            ):
                factor = (
                    (t - a_t)
                    / max(b_t - a_t, 1e-8)
                )

                result.append(
                    work[i].lerp(
                        work[i + 1],
                        factor
                    )
                )
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


class HP_OT_curve_shape_choice(bpy.types.Operator):
    bl_idname = "hp.curve_shape_choice"
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
        editor = _ACTIVE_CURVE_EDITOR
        if editor is None:
            return {'CANCELLED'}

        editor._apply_shape_tool(context, self.tool)

        if context.area:
            context.area.tag_redraw()

        return {'FINISHED'}


class HP_MT_curve_shape_pie(bpy.types.Menu):
    bl_idname = "HP_MT_curve_shape_pie"
    bl_label = "整形"

    def draw(self, context):
        pie = self.layout.menu_pie()

        op = pie.operator(
            HP_OT_curve_shape_choice.bl_idname,
            text="等間隔化",
            icon='ALIGN_JUSTIFY'
        )
        op.tool = 'EQUAL'

        op = pie.operator(
            HP_OT_curve_shape_choice.bl_idname,
            text="直線化",
            icon='IPO_LINEAR'
        )
        op.tool = 'LINE'

        op = pie.operator(
            HP_OT_curve_shape_choice.bl_idname,
            text="楕円フィット",
            icon='MESH_CIRCLE'
        )
        op.tool = 'ELLIPSE'

        op = pie.operator(
            HP_OT_curve_shape_choice.bl_idname,
            text="左右対称化",
            icon='MOD_MIRROR'
        )
        op.tool = 'SYMMETRY'



class HP_OT_curve_topology_choice(bpy.types.Operator):
    bl_idname = "hp.curve_topology_choice"
    bl_label = "細分化"

    tool: EnumProperty(
        items=(
            ('SUBDIVIDE', "細分化", "選択範囲のカーブ区間を1段細分化します"),
            ('DISSOLVE', "点を溶解", "選択した制御点を溶解します"),
            ('MERGE', "2点を中央マージ", "隣接する2点を中央で1点化します"),
        ),
        default='SUBDIVIDE',
    )

    def execute(self, context):
        editor = _ACTIVE_CURVE_EDITOR

        if editor is None:
            return {'CANCELLED'}

        editor._apply_topology_tool(
            context,
            self.tool
        )

        if context.area:
            context.area.tag_redraw()

        return {'FINISHED'}


class HP_MT_curve_topology_pie(bpy.types.Menu):
    bl_idname = "HP_MT_curve_topology_pie"
    bl_label = "細分化"

    def draw(self, context):
        pie = self.layout.menu_pie()

        op = pie.operator(
            HP_OT_curve_topology_choice.bl_idname,
            text="細分化（1段）"
        )
        op.tool = 'SUBDIVIDE'

        op = pie.operator(
            HP_OT_curve_topology_choice.bl_idname,
            text="点を溶解"
        )
        op.tool = 'DISSOLVE'

        op = pie.operator(
            HP_OT_curve_topology_choice.bl_idname,
            text="2点を中央マージ"
        )
        op.tool = 'MERGE'


class HP_OT_curve_pen_anchor_choice(bpy.types.Operator):
    bl_idname = "hp.curve_pen_anchor_choice"
    bl_label = "ペン始点モード"

    mode: EnumProperty(
        items=(
            ('STROKE', "Stroke Start", "First point follows where the pen stroke starts"),
            ('POINT', "Point Start", "Keep the first target control point fixed"),
        ),
        default='STROKE',
    )

    def execute(self, context):
        global _CURVE_PEN_ANCHOR_DEFAULT

        editor = _ACTIVE_CURVE_EDITOR
        if editor is None:
            return {'CANCELLED'}

        _CURVE_PEN_ANCHOR_DEFAULT = self.mode
        editor._pen_anchor_mode = self.mode
        editor._activate_pen()

        if context.area:
            context.area.tag_redraw()

        return {'FINISHED'}


class HP_MT_curve_pen_anchor_pie(bpy.types.Menu):
    bl_idname = "HP_MT_curve_pen_anchor_pie"
    bl_label = "ペン始点モード"

    def draw(self, context):
        pie = self.layout.menu_pie()

        op = pie.operator(
            HP_OT_curve_pen_anchor_choice.bl_idname,
            text="描き始め位置を優先",
            icon='GREASEPENCIL'
        )
        op.mode = 'STROKE'

        op = pie.operator(
            HP_OT_curve_pen_anchor_choice.bl_idname,
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


def _hp_adjacent_rings(ring, faces, max_depth=6):
    """Follow only unambiguous quad strips, never jump by spatial proximity."""
    edge_faces={}
    for fi,face in enumerate(faces):
        for a,b in zip(face,face[1:]+face[:1]):
            edge_faces.setdefault(tuple(sorted((a,b))),[]).append(fi)
    def strip(current,first_face):
        current_set=set(current)
        mapping={}
        used=set()
        for i,a in enumerate(current):
            b=current[(i+1)%len(current)]
            candidates=[]
            linked=edge_faces.get(tuple(sorted((a,b))),[])
            if len(linked)>2: return None
            for fi in linked:
                face=faces[fi]
                if len(face)!=4 or set(face)&current_set != {a,b}: continue
                ai,bi=face.index(a),face.index(b)
                na=next(v for v in (face[(ai-1)%4],face[(ai+1)%4]) if v!=b)
                nb=next(v for v in (face[(bi-1)%4],face[(bi+1)%4]) if v!=a)
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
        if any(len(edge_faces.get(tuple(sorted((a,b))),[]))>2
               for a,b in zip(out,out[1:]+out[:1])): return None
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
    return (sides+[[],[]])[:2]


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


_hp_sample = _sample_polyline
_hp_average = _moving_average
_hp_stabilize = _stabilize

class HP_OT_curve_mini_editor(bpy.types.Operator):
    _hp_is_curve = True

    def _hp_begin_adjust(self, context, target, show=True):
        if getattr(self,'_hp_menu',None) is not None: return False
        curve=self._hp_is_curve
        if curve:
            self._refresh_selection_from_blender()
            order,full=self._selected_order()
            stroke=self._pen_stroke; level=self._pen_stabilizer
            points=self._panel_points()
        elif target=='B':
            order,full=self._secondary_pen_selected_order()
            stroke=self._secondary_pen_stroke; level=self._secondary_pen_stabilizer
            points=self._secondary_panel_points()
        else:
            order,full=self._pen_selected_order()
            stroke=self._pen_stroke; level=self._pen_stabilizer
            points=self._panel_points
        if order is None or len(order)<2 or len(stroke)<2:
            self.report({'WARNING'},'描線と連続した2点以上が必要です')
            return False
        closed=bool(full)
        if closed and (len(stroke)<3 or (stroke[-1]-stroke[0]).length>20):
            self.report({'WARNING'},'輪全体を編集する場合は描線を閉じてください')
            return False
        settings=dict(getattr(self,'_hp_settings',dict(correction=0.0,amount=1.0,shape=False,side_a=0,side_b=0,falloff=1.0)))
        if not closed: settings['shape']=False
        state=dict(target=target,order=list(order),closed=closed,
                   stroke=_hp_stabilize(stroke,level=level,closed=closed),
                   panel=[points[i].copy() for i in order],settings=settings,
                   original_settings=dict(settings),drag=None,wait_release=True,
                   data=self._obj.data, sides=[[],[]],show=show)
        if curve:
            state['base']=self._snapshot_all()
            state['source_world']=[self._point_world(i).copy() for i in order]
        else:
            self._bm.verts.ensure_lookup_table(); self._bm.verts.index_update()
            self._bm.edges.ensure_lookup_table()
            state['counts']=(len(self._bm.verts),len(self._bm.edges),len(self._bm.faces))
            state['ring']=[self._ordered[i] for i in order]
            # Only closed, complete rings can propagate through quad strips.
            if closed and (show or settings['side_a'] or settings['side_b']):
                faces=[tuple(v.index for v in f.verts) for f in self._bm.faces]
                state['sides']=_hp_adjacent_rings(state['ring'],faces)
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
            for vi,base,goal in zip(state['ring'],state['source_world'],goals):
                self._bm.verts[vi].co=inv@self._apply_world_locks(base,base.lerp(goal,amount))
            for side,key in zip(state['sides'],('side_a','side_b')):
                count=min(int(settings[key]),len(side))
                for depth,ring in enumerate(side[:count],1):
                    weight=(1.0-depth/(count+1.0))**settings['falloff']
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
            if self._hp_is_curve:
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
        if getattr(context,'area',None): context.area.tag_redraw()

    def _hp_menu_rows(self):
        state=self._hp_menu
        rows=[('correction','輪郭補正',0.0,20.0),('amount','反映量',0.0,1.0)]
        if not self._hp_is_curve and state['closed']:
            rows.extend([('side_a','隣接A ループ数',0,len(state['sides'][0])),
                         ('side_b','隣接B ループ数',0,len(state['sides'][1])),
                         ('falloff','遠くほど弱くする強さ',0.5,3.0)])
        return rows

    def _hp_menu_layout(self, context):
        rows=self._hp_menu_rows()
        width=min(380,max(240,context.region.width-24))
        height=178+len(rows)*44
        x=max(8,min(context.region.width-width-12,context.region.width*0.5-width*0.5))
        y=max(8,min(context.region.height-height-12,context.region.height*0.5-height*0.5))
        return x,y,width,height,rows

    def _hp_menu_value(self, context, key, mx):
        x,y,w,h,rows=self._hp_menu_layout(context)
        row=next(r for r in rows if r[0]==key)
        value=row[2]+max(0.0,min(1.0,(mx-x-18)/(w-36)))*(row[3]-row[2])
        if key.startswith('side_'): value=int(round(value))
        self._hp_menu['settings'][key]=value
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
        if event.type=='MOUSEMOVE' and state['drag']:
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
                    ry=y+h-118-n*44
                    if x+12<=mx<=x+w-12 and ry-10<=my<=ry+18:
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
        label(x+16,y+h-28,'ペン調整 — 立体に仮反映',16)
        settings=self._hp_menu['settings']
        self._draw_rect(x+12,y+h-82,w-24,30,(.10,.17,.23,1))
        mode='形だけ（中心・大きさを維持）' if settings['shape'] else '描線どおり（位置・大きさも反映）'
        if not self._hp_menu['closed']: mode='描線どおり（開いた線）'
        label(x+20,y+h-72,mode)
        for n,(key,title,lo,hi) in enumerate(rows):
            ry=y+h-118-n*44
            value=settings[key]
            text=f'{value*100:.0f}%' if key=='amount' else (str(int(value)) if key.startswith('side_') else f'{value:.1f}')
            if key.startswith('side_'): text+=f' / {hi}'
            label(x+18,ry+11,title+'  '+text,12)
            self._draw_rect(x+18,ry-7,w-36,7,(.15,.18,.22,1))
            fraction=(value-lo)/(hi-lo) if hi>lo else 0
            self._draw_rect(x+18,ry-7,(w-36)*fraction,7,(.25,.75,.62,1))
        label(x+16,y+65,'F / Enter 確定    Esc 描線へ戻る',12)
        self._draw_rect(x+12,y+18,w*.5-16,32,(.12,.40,.32,1))
        self._draw_rect(x+w*.5+4,y+18,w*.5-16,32,(.23,.25,.29,1))
        label(x+30,y+28,'確定'); label(x+w*.5+20,y+28,'キャンセル')

    bl_idname = "hp.curve_mini_editor"
    bl_label = "HP Curve Mini Editor"
    bl_options = {'REGISTER', 'UNDO'}

    start_top: BoolProperty(
        name="Start Top View",
        description="Open the editor in the section-facing top view",
        default=False,
        options={'HIDDEN'},
    )

    def invoke(self, context, event):
        global _CURVE_RUNNING, _ACTIVE_CURVE_EDITOR, _CURVE_PINNED_WORKSPACE_PTR

        if _CURVE_RUNNING or context.area.type != 'VIEW_3D':
            return {'CANCELLED'}

        target = _curve_target(context)

        if target is None:
            return {'CANCELLED'}

        current_workspace_ptr = (
            context.window.workspace.as_pointer()
            if (
                context.window is not None
                and context.window.workspace is not None
            )
            else 0
        )

        if (
            _CURVE_PINNED_WORKSPACE_PTR
            and current_workspace_ptr != _CURVE_PINNED_WORKSPACE_PTR
        ):
            return {'CANCELLED'}

        if not _CURVE_PINNED_WORKSPACE_PTR:
            _CURVE_PINNED_WORKSPACE_PTR = current_workspace_ptr

        self._finished = False
        self._viewport_navigation_active = False
        self._timer_owner = context.window_manager
        self._handle = None
        self._timer = None

        # Bind this overlay to the exact View3D/workspace that spawned it.
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

        self._obj = target["obj"]
        self._spline_index = target["spline_index"]
        self._spline = target["spline"]
        self._bezier = target["bezier"]
        self._cyclic = target["cyclic"]
        self._signature = target["signature"]

        self._selected = set(target["selected"])

        self._panel_w = max(180, min(400, context.region.width - 36))
        self._panel_h = max(160, min(300, context.region.height - 80))
        self._panel_x = max(18, context.region.width - self._panel_w - 18)
        self._panel_y = max(18, context.region.height - self._panel_h - 18)

        self._view = 'TOP' if self.start_top else 'FRONT'
        self._zoom = 1.0
        self._flip_y = False
        self._flip_x = False
        self._xray = False
        self._occlusion_px = 10.0

        self._view_valid = False
        self._view_center = Vector((0.0, 0.0))
        self._panel_center = Vector((0.0, 0.0))
        self._view_base_scale = 1.0

        self._dragging = False
        self._drag_start_mouse = Vector((0.0, 0.0))
        self._drag_base = []

        self._box_dragging = False
        self._box_start = None
        self._box_end = None
        self._box_additive = False

        self._transform_mode = None
        self._transform_axis = None
        self._transform_start_mouse = Vector((0.0, 0.0))
        self._transform_base = []
        self._transform_pivot = Vector((0.0, 0.0, 0.0))

        self._pen_mode = False
        self._pen_drawing = False
        self._pen_stroke = []
        self._pen_stabilizer = 2
        self._pen_anchor_mode = _CURVE_PEN_ANCHOR_DEFAULT

        self._f_hold_active = False
        self._f_hold_started = 0.0
        self._f_long_opened = False

        self._prop_enabled = False
        self._prop_radius_px = 90.0

        self._history = []
        self._history_limit = 64

        self._topology_undo_steps = 0
        self._topology_rebuild_pending = False

        # Interactive E Smooth drag.
        self._smooth_dragging = False
        self._smooth_start_x = 0.0
        self._smooth_amount = 0.0
        self._smooth_order = []
        self._smooth_closed = False
        self._smooth_base_panel = []
        self._smooth_base_snapshot = []
        self._smooth_brush_radius = 70.0
        self._smooth_brush_strength = 0.55
        self._smooth_brush_prev_mouse = Vector((0.0, 0.0))

        # Interactive E Post Smooth for the DRAWN PEN STROKE.
        self._pen_smooth_dragging = False
        self._pen_smooth_start_x = 0.0
        self._pen_smooth_amount = 0.0
        self._pen_smooth_base = []
        self._pen_smooth_closed = False
        self._pen_smooth_last_delta_px = 0.0
        self._pen_smooth_start_amount = 0.0
        self._pen_smooth_cancel_stroke = []
        self._pen_smooth_cancel_amount = 0.0

        self._pen_raw_stroke = []
        self._pen_smooth_level = 0.0

        self._smooth_hud_x = 0.0
        self._smooth_hud_y = 0.0

        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw,
            (),
            'WINDOW',
            'POST_PIXEL'
        )

        self._timer = context.window_manager.event_timer_add(
            0.12,
            window=context.window
        )

        _CURVE_RUNNING = True
        _ACTIVE_CURVE_EDITOR = self

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

    # ---------------------------------------------------------
    # Curve access / snapshots
    # ---------------------------------------------------------
    def _point_count(self):
        return (
            len(self._spline.bezier_points)
            if self._bezier
            else len(self._spline.points)
        )

    def _point_local(self, i):
        if self._bezier:
            return self._spline.bezier_points[i].co.copy()

        p = self._spline.points[i]
        return Vector((p.co.x, p.co.y, p.co.z))

    def _point_world(self, i):
        return (
            self._obj.matrix_world
            @ self._point_local(i)
        )

    def _snapshot_point(self, i):
        if self._bezier:
            bp = self._spline.bezier_points[i]
            return {
                "co": bp.co.copy(),
                "hl": bp.handle_left.copy(),
                "hr": bp.handle_right.copy(),
            }

        p = self._spline.points[i]
        return {
            "co4": p.co.copy(),
        }

    def _snapshot_all(self):
        return [
            self._snapshot_point(i)
            for i in range(self._point_count())
        ]

    def _restore_snapshot(self, snapshot):
        if len(snapshot) != self._point_count():
            return False

        if self._bezier:
            for i, data in enumerate(snapshot):
                bp = self._spline.bezier_points[i]
                bp.co = data["co"].copy()
                bp.handle_left = data["hl"].copy()
                bp.handle_right = data["hr"].copy()
        else:
            for i, data in enumerate(snapshot):
                self._spline.points[i].co = data["co4"].copy()

        self._obj.data.update_tag()
        return True

    def _push_history(self):
        snapshot = self._snapshot_all()

        self._history.append(snapshot)

        if len(self._history) > self._history_limit:
            self._history.pop(0)

    def _topology_undo(self, context):
        if self._topology_undo_steps <= 0:
            return False

        try:
            bpy.ops.ed.undo()
        except Exception as exc:
            self.report(
                {'WARNING'},
                f"Curve Topology Undo failed: {exc}"
            )
            return False

        self._topology_undo_steps -= 1
        self._history.clear()
        self._topology_rebuild_pending = True
        context.area.tag_redraw()
        return True

    def _undo_local(self, context):
        if not self._history:
            self._topology_undo(context)
            return

        snapshot = self._history.pop()

        if self._restore_snapshot(snapshot):
            context.area.tag_redraw()

    def _set_point_world_from_base(
        self,
        i,
        base_snapshot,
        new_world
    ):
        inv = self._obj.matrix_world.inverted()
        new_local = inv @ new_world

        if self._bezier:
            bp = self._spline.bezier_points[i]
            old_local = base_snapshot["co"]
            delta = new_local - old_local

            bp.co = new_local
            bp.handle_left = (
                base_snapshot["hl"] + delta
            )
            bp.handle_right = (
                base_snapshot["hr"] + delta
            )
        else:
            old4 = base_snapshot["co4"]

            self._spline.points[i].co = (
                new_local.x,
                new_local.y,
                new_local.z,
                old4.w
            )

    def _apply_world_transform_to_snapshot(
        self,
        i,
        base_snapshot,
        transform_fn
    ):
        mw = self._obj.matrix_world
        inv = mw.inverted()

        if self._bezier:
            bp = self._spline.bezier_points[i]

            co_w = mw @ base_snapshot["co"]
            hl_w = mw @ base_snapshot["hl"]
            hr_w = mw @ base_snapshot["hr"]

            bp.co = inv @ transform_fn(co_w)
            bp.handle_left = inv @ transform_fn(hl_w)
            bp.handle_right = inv @ transform_fn(hr_w)
        else:
            co4 = base_snapshot["co4"]
            co_w = mw @ Vector(
                (co4.x, co4.y, co4.z)
            )
            result = inv @ transform_fn(co_w)

            self._spline.points[i].co = (
                result.x,
                result.y,
                result.z,
                co4.w
            )

    def _sync_blender_selection(self):
        if self._bezier:
            for i, bp in enumerate(
                self._spline.bezier_points
            ):
                state = i in self._selected
                bp.select_control_point = state
        else:
            for i, p in enumerate(
                self._spline.points
            ):
                p.select = i in self._selected

    def _refresh_selection_from_blender(self):
        """Read the live Edit Mode selection back into the mini editor."""
        if self._bezier:
            self._selected = {
                i
                for i, bp in enumerate(self._spline.bezier_points)
                if bp.select_control_point
            }
        else:
            self._selected = {
                i
                for i, p in enumerate(self._spline.points)
                if p.select
            }

    # ---------------------------------------------------------
    # Camera / projection
    # ---------------------------------------------------------
    def _set_view(self, view_name):
        if view_name in {
            'FRONT',
            'BACK',
            'RIGHT',
            'LEFT',
            'TOP',
            'BOTTOM',
            'TOP_X_DOWN',
            'TOP_Y_DOWN',
        }:
            self._view = view_name
            self._view_valid = False

    def _view_shortcut(self):
        return {
            'FRONT': 'Num1',
            'BACK': 'Ctrl+Num1',
            'RIGHT': 'Num3',
            'LEFT': 'Ctrl+Num3',
            'TOP': 'Num7',
            'BOTTOM': 'Ctrl+Num7',
            'TOP_X_DOWN': 'Num4',
            'TOP_Y_DOWN': 'Num6',
        }[self._view]

    def _world_to_plane(self, world):
        if self._view == 'FRONT':
            return Vector((world.x, world.z))

        if self._view == 'BACK':
            return Vector((-world.x, world.z))

        if self._view == 'RIGHT':
            return Vector((world.y, world.z))

        if self._view == 'LEFT':
            return Vector((-world.y, world.z))

        if self._view == 'TOP':
            return Vector((world.x, world.y))

        if self._view == 'BOTTOM':
            return Vector((-world.x, world.y))

        if self._view == 'TOP_X_DOWN':
            return Vector((world.y, -world.x))

        return Vector((world.x, -world.y))

    def _plane_to_world(self, plane, original_world):
        result = original_world.copy()

        if self._view == 'FRONT':
            result.x = plane.x
            result.z = plane.y

        elif self._view == 'BACK':
            result.x = -plane.x
            result.z = plane.y

        elif self._view == 'RIGHT':
            result.y = plane.x
            result.z = plane.y

        elif self._view == 'LEFT':
            result.y = -plane.x
            result.z = plane.y

        elif self._view == 'TOP':
            result.x = plane.x
            result.y = plane.y

        elif self._view == 'BOTTOM':
            result.x = -plane.x
            result.y = plane.y

        elif self._view == 'TOP_X_DOWN':
            result.x = -plane.y
            result.y = plane.x

        else:
            result.x = plane.x
            result.y = -plane.y

        return result

    def _depth_value(self, i):
        world = self._point_world(i)

        if self._view == 'FRONT':
            return -world.y

        if self._view == 'BACK':
            return world.y

        if self._view == 'RIGHT':
            return world.x

        if self._view == 'LEFT':
            return -world.x

        if self._view in {
            'TOP',
            'BOTTOM',
            'TOP_X_DOWN',
            'TOP_Y_DOWN',
        }:
            return -world.z

        return world.z

    def _plane_coords(self):
        return [
            self._world_to_plane(
                self._point_world(i)
            )
            for i in range(self._point_count())
        ]

    def _refit(self):
        coords = self._plane_coords()

        if not coords:
            return False

        xs = [p.x for p in coords]
        ys = [p.y for p in coords]

        self._view_center = Vector((
            (min(xs) + max(xs)) * 0.5,
            (min(ys) + max(ys)) * 0.5,
        ))

        self._panel_center = Vector((
            self._panel_x + self._panel_w * 0.5,
            self._panel_y + self._panel_h * 0.5 - 2,
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

        self._view_base_scale = min(
            usable_w / span_x,
            usable_h / span_y
        ) * 0.90

        self._view_valid = True
        return True

    def _scale(self):
        if not self._view_valid:
            self._refit()

        return (
            self._view_base_scale
            * self._zoom
        )

    def _panel_points(self):
        if not self._view_valid:
            self._refit()

        scale = max(
            self._scale(),
            1e-12
        )

        sx = -1.0 if self._flip_x else 1.0
        sy = -1.0 if self._flip_y else 1.0

        return [
            self._panel_center
            + Vector((
                (p.x - self._view_center.x) * scale * sx,
                (p.y - self._view_center.y) * scale * sy,
            ))
            for p in self._plane_coords()
        ]

    def _mouse_to_plane(self, mx, my):
        if not self._view_valid:
            self._refit()

        scale = max(
            self._scale(),
            1e-12
        )

        sx = -1.0 if self._flip_x else 1.0
        sy = -1.0 if self._flip_y else 1.0

        return Vector((
            (mx - self._panel_center.x) / (scale * sx)
            + self._view_center.x,
            (my - self._panel_center.y) / (scale * sy)
            + self._view_center.y,
        ))

    def _inside(self, x, y):
        return (
            self._panel_x <= x
            <= self._panel_x + self._panel_w
            and self._panel_y <= y
            <= self._panel_y + self._panel_h
        )

    def _front_filter(
        self,
        indices,
        points=None
    ):
        indices = list(indices)

        if self._xray or len(indices) <= 1:
            return indices

        if points is None:
            points = self._panel_points()

        ordered = sorted(
            indices,
            key=self._depth_value,
            reverse=True
        )

        visible = []

        for i in ordered:
            if any(
                (
                    points[i]
                    - points[j]
                ).length <= self._occlusion_px
                for j in visible
            ):
                continue

            visible.append(i)

        return visible

    def _pick_point(self, x, y):
        points = self._panel_points()
        mouse = Vector((x, y))

        candidates = [
            i
            for i, p in enumerate(points)
            if (p - mouse).length <= 14.0
        ]

        candidates = self._front_filter(
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

    def _weight(self, i, base_points):
        if i in self._selected:
            return 1.0

        if (
            not self._prop_enabled
            or not self._selected
        ):
            return 0.0

        # Alt+Z OFF = visible/front-side proportional editing only.
        if not self._xray:
            visible = set(
                self._front_filter(
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

    # ---------------------------------------------------------
    # Direct drag / box
    # ---------------------------------------------------------
    def _begin_drag(self, mx, my):
        if not self._selected:
            return

        self._push_history()
        self._dragging = True
        self._drag_start_mouse = Vector((mx, my))
        self._drag_base = self._snapshot_all()

        self._drag_base_panel = [
            p.copy()
            for p in self._panel_points()
        ]

    def _apply_drag(self, context, mx, my):
        if not self._dragging:
            return

        start_plane = self._mouse_to_plane(
            self._drag_start_mouse.x,
            self._drag_start_mouse.y
        )

        now_plane = self._mouse_to_plane(
            mx,
            my
        )

        delta = now_plane - start_plane

        for i in range(self._point_count()):
            w = self._weight(
                i,
                self._drag_base_panel
            )

            if w <= 0.0:
                continue

            base_world = (
                self._obj.matrix_world
                @ (
                    self._drag_base[i]["co"]
                    if self._bezier
                    else Vector((
                        self._drag_base[i]["co4"].x,
                        self._drag_base[i]["co4"].y,
                        self._drag_base[i]["co4"].z,
                    ))
                )
            )

            base_plane = self._world_to_plane(
                base_world
            )

            new_world = self._plane_to_world(
                base_plane + delta * w,
                base_world
            )

            self._set_point_world_from_base(
                i,
                self._drag_base[i],
                new_world
            )

        self._obj.data.update_tag()
        context.area.tag_redraw()

    def _apply_box(self):
        if (
            self._box_start is None
            or self._box_end is None
        ):
            return

        x1 = min(
            self._box_start.x,
            self._box_end.x
        )
        x2 = max(
            self._box_start.x,
            self._box_end.x
        )
        y1 = min(
            self._box_start.y,
            self._box_end.y
        )
        y2 = max(
            self._box_start.y,
            self._box_end.y
        )

        points = self._panel_points()

        candidates = [
            i
            for i, p in enumerate(points)
            if x1 <= p.x <= x2
            and y1 <= p.y <= y2
        ]

        picked = set(
            self._front_filter(
                candidates,
                points
            )
        )

        if self._box_additive:
            self._selected |= picked
        else:
            self._selected = picked

        self._sync_blender_selection()

    # ---------------------------------------------------------
    # G / S / R
    # ---------------------------------------------------------
    def _axis_vector(self):
        if self._transform_axis == 'X':
            return Vector((1.0, 0.0, 0.0))

        if self._transform_axis == 'Y':
            return Vector((0.0, 1.0, 0.0))

        if self._transform_axis == 'Z':
            return Vector((0.0, 0.0, 1.0))

        return None

    def _begin_transform(
        self,
        mode,
        mx,
        my
    ):
        if not self._selected:
            return

        self._push_history()

        self._transform_mode = mode
        self._transform_axis = None
        self._transform_start_mouse = Vector((mx, my))
        self._transform_base = self._snapshot_all()

        worlds = []

        for i in sorted(self._selected):
            worlds.append(
                self._point_world(i)
            )

        self._transform_pivot = (
            sum(
                worlds,
                Vector((0.0, 0.0, 0.0))
            )
            / len(worlds)
        )

        self._transform_base_panel = [
            p.copy()
            for p in self._panel_points()
        ]

    def _screen_delta_to_world(
        self,
        plane_delta
    ):
        result = Vector((0.0, 0.0, 0.0))

        if self._view == 'FRONT':
            result.x = plane_delta.x
            result.z = plane_delta.y

        elif self._view == 'BACK':
            result.x = -plane_delta.x
            result.z = plane_delta.y

        elif self._view == 'RIGHT':
            result.y = plane_delta.x
            result.z = plane_delta.y

        else:
            result.y = -plane_delta.x
            result.z = plane_delta.y

        return result

    def _apply_transform(
        self,
        context,
        mx,
        my
    ):
        if self._transform_mode is None:
            return

        mouse_delta = (
            Vector((mx, my))
            - self._transform_start_mouse
        )

        start_plane = self._mouse_to_plane(
            self._transform_start_mouse.x,
            self._transform_start_mouse.y
        )
        now_plane = self._mouse_to_plane(
            mx,
            my
        )

        plane_delta = now_plane - start_plane
        world_delta = self._screen_delta_to_world(
            plane_delta
        )

        scale_factor = max(
            0.01,
            1.0 + mouse_delta.x / 140.0
        )
        angle = mouse_delta.x * 0.01

        axis = self._axis_vector()
        pivot_plane = self._world_to_plane(
            self._transform_pivot
        )

        for i in range(self._point_count()):
            w = self._weight(
                i,
                self._transform_base_panel
            )

            if w <= 0.0:
                continue

            base = self._transform_base[i]

            def target_fn(world):
                full = world.copy()

                if self._transform_mode == 'G':
                    if axis is None:
                        plane = self._world_to_plane(world)

                        full = self._plane_to_world(
                            plane + plane_delta,
                            world
                        )
                    else:
                        full = (
                            world
                            + axis * world_delta.dot(axis)
                        )

                elif self._transform_mode == 'S':
                    if axis is None:
                        plane = self._world_to_plane(world)

                        full = self._plane_to_world(
                            pivot_plane
                            + (
                                plane - pivot_plane
                            ) * scale_factor,
                            world
                        )
                    else:
                        rel = world - self._transform_pivot
                        parallel = axis * rel.dot(axis)
                        perpendicular = rel - parallel

                        full = (
                            self._transform_pivot
                            + perpendicular
                            + parallel * scale_factor
                        )

                elif self._transform_mode == 'R':
                    if axis is None:
                        plane = self._world_to_plane(world)
                        rel = plane - pivot_plane

                        c = cos(angle)
                        s = sin(angle)

                        rotated = Vector((
                            rel.x * c - rel.y * s,
                            rel.x * s + rel.y * c,
                        ))

                        full = self._plane_to_world(
                            pivot_plane + rotated,
                            world
                        )
                    else:
                        q = Quaternion(
                            axis,
                            angle
                        )

                        full = (
                            self._transform_pivot
                            + q @ (
                                world
                                - self._transform_pivot
                            )
                        )

                return world.lerp(full, w)

            self._apply_world_transform_to_snapshot(
                i,
                base,
                target_fn
            )

        self._obj.data.update_tag()
        context.area.tag_redraw()

    # ---------------------------------------------------------
    # Pen
    # ---------------------------------------------------------
    def _begin_smooth_drag(
        self,
        context,
        mx,
        my
    ):
        self._refresh_selection_from_blender()

        order, full_loop = self._selected_order()

        if order is None or len(order) < 3:
            self.report(
                {'WARNING'},
                "Smooth: select one connected range with at least 3 points"
            )
            _hp_smooth_debug(
                f"CURVE BEGIN rejected order={order}"
            )
            return False

        self._push_history()

        self._smooth_dragging = True
        self._smooth_start_x = float(mx)
        self._smooth_amount = 0.0
        self._smooth_hud_x = float(mx)
        self._smooth_hud_y = float(my)
        self._smooth_brush_prev_mouse = Vector((mx, my))
        self._smooth_order = list(order)
        self._smooth_closed = bool(full_loop)
        self._smooth_base_panel = [
            p.copy()
            for p in self._panel_points()
        ]
        self._smooth_base_snapshot = (
            self._snapshot_all()
        )

        _hp_smooth_debug(
            "CURVE BEGIN "
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

        current = [p.copy() for p in self._panel_points()]
        changed = _hp_brush_repel_2d(
            current,
            Vector((mx, my)),
            motion,
            radius=self._smooth_brush_radius,
            strength=self._smooth_brush_strength,
        )

        max_delta_px = 0.0

        for original_i in self._smooth_order:
            panel_p = changed[original_i]
            base_p = current[original_i]
            max_delta_px = max(
                max_delta_px,
                (panel_p - base_p).length
            )

            plane = self._mouse_to_plane(
                panel_p.x,
                panel_p.y
            )

            original_world = (
                self._obj.matrix_world
                @ (
                    self._smooth_base_snapshot[
                        original_i
                    ]["co"]
                    if self._bezier
                    else Vector((
                        self._smooth_base_snapshot[
                            original_i
                        ]["co4"].x,
                        self._smooth_base_snapshot[
                            original_i
                        ]["co4"].y,
                        self._smooth_base_snapshot[
                            original_i
                        ]["co4"].z,
                    ))
                )
            )

            new_world = self._plane_to_world(
                plane,
                original_world
            )

            self._set_point_world_from_base(
                original_i,
                self._smooth_base_snapshot[
                    original_i
                ],
                new_world
            )

        self._smooth_last_delta_px = max_delta_px
        self._obj.data.update_tag()
        context.area.tag_redraw()

    def _finish_smooth_drag(
        self,
        context,
        cancel=False
    ):
        if not self._smooth_dragging:
            return

        if cancel:
            self._restore_snapshot(
                self._smooth_base_snapshot
            )

            if self._history:
                self._history.pop()

        _hp_smooth_debug(
            "CURVE END "
            f"cancel={cancel} "
            f"amount={self._smooth_amount:.3f} "
            f"max_delta_px={getattr(self, '_smooth_last_delta_px', 0.0):.3f}"
        )

        self._smooth_dragging = False
        self._smooth_order = []
        self._smooth_base_panel = []
        self._smooth_base_snapshot = []

        self._obj.data.update_tag()
        context.area.tag_redraw()

    def _set_curve_point_selection(
        self,
        indices
    ):
        indices = set(indices)

        for spline_i, spline in enumerate(
            self._obj.data.splines
        ):
            active = (
                spline_i
                == self._spline_index
            )

            if spline.type == 'BEZIER':
                for i, bp in enumerate(
                    spline.bezier_points
                ):
                    state = (
                        active
                        and i in indices
                    )
                    bp.select_control_point = state
                    bp.select_left_handle = state
                    bp.select_right_handle = state
            else:
                for i, p in enumerate(
                    spline.points
                ):
                    p.select = (
                        active
                        and i in indices
                    )

    def _select_all_current_spline(self):
        try:
            spline = self._obj.data.splines[
                self._spline_index
            ]
        except Exception:
            return

        count = (
            len(spline.bezier_points)
            if spline.type == 'BEZIER'
            else len(spline.points)
        )

        self._set_curve_point_selection(
            range(count)
        )

    def _curve_point_local_from_spline(
        self,
        spline,
        index
    ):
        if spline.type == 'BEZIER':
            return spline.bezier_points[
                index
            ].co.copy()

        p = spline.points[index]

        return Vector((
            p.co.x,
            p.co.y,
            p.co.z
        ))

    def _topology_undo_push(self):
        try:
            bpy.ops.ed.undo_push(
                message="HP Curve Mini Topology"
            )
        except Exception:
            pass

    def _refresh_after_topology(self, context):
        target = _curve_target(context)

        if target is None:
            self._select_all_current_spline()
            target = _curve_target(context)

        if target is None:
            return False

        self._obj = target["obj"]
        self._spline_index = target["spline_index"]
        self._spline = target["spline"]
        self._bezier = target["bezier"]
        self._cyclic = target["cyclic"]
        self._signature = target["signature"]
        self._selected = set(target["selected"])
        self._view_valid = False

        return True

    def _ensure_curve_selection_after_topology(
        self,
        context
    ):
        if _curve_target(context) is None:
            self._select_all_current_spline()

        self._obj.data.update_tag()

    def _finish_after_topology(self, context):
        self._obj.data.update_tag()

        self._history.clear()
        self._topology_undo_steps += 1
        self._topology_rebuild_pending = False

        self._dragging = False
        self._box_dragging = False
        self._transform_mode = None
        self._pen_mode = False
        self._pen_drawing = False
        self._pen_stroke = []
        self._pen_raw_stroke = []
        self._pen_smooth_level = 0.0

        if not self._refresh_after_topology(
            context
        ):
            self._topology_rebuild_pending = True

        context.area.tag_redraw()

    def _apply_topology_tool(
        self,
        context,
        tool
    ):
        self._refresh_selection_from_blender()

        selected = set(self._selected)

        if not selected:
            self.report(
                {'WARNING'},
                "細分化: 制御点を選択してください"
            )
            return False

        n = self._point_count()

        if tool == 'SUBDIVIDE':
            order, _ = self._selected_order()

            if order is None or len(order) < 2:
                self.report(
                    {'WARNING'},
                    "細分化: 連続した2点以上を選択してください"
                )
                return False

            self._sync_blender_selection()
            self._topology_undo_push()

            try:
                bpy.ops.curve.subdivide(
                    number_cuts=1
                )
            except Exception as exc:
                self.report(
                    {'WARNING'},
                    f"Curve Subdivide failed: {exc}"
                )
                return False

            self._ensure_curve_selection_after_topology(
                context
            )
            self._finish_after_topology(
                context
            )
            return True

        if tool == 'DISSOLVE':
            victims = set(selected)

            if not self._cyclic:
                victims.discard(0)
                victims.discard(n - 1)

            min_remaining = (
                3 if self._cyclic else 2
            )

            if (
                not victims
                or n - len(victims) < min_remaining
            ):
                self.report(
                    {'WARNING'},
                    "溶解: 開いたカーブの両端は残し、必要な点数を確保してください"
                )
                return False

            self._set_curve_point_selection(
                victims
            )
            self._topology_undo_push()

            try:
                bpy.ops.curve.dissolve_verts()
            except Exception as exc:
                self.report(
                    {'WARNING'},
                    f"Curve Dissolve failed: {exc}"
                )
                return False

            self._ensure_curve_selection_after_topology(
                context
            )
            self._finish_after_topology(
                context
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
                self._cyclic
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
                    "マージ: 2点は隣接している必要があります"
                )
                return False

            if wrap_pair:
                keep_i = 0
                kill_i = n - 1
            else:
                keep_i = a_i
                kill_i = b_i

            spline = self._spline

            keep_local = (
                self._curve_point_local_from_spline(
                    spline,
                    keep_i
                )
            )
            kill_local = (
                self._curve_point_local_from_spline(
                    spline,
                    kill_i
                )
            )
            mid = (
                keep_local + kill_local
            ) * 0.5

            if self._bezier:
                keep = spline.bezier_points[
                    keep_i
                ]

                delta = mid - keep.co
                keep.co = mid
                keep.handle_left += delta
                keep.handle_right += delta
            else:
                keep = spline.points[
                    keep_i
                ]
                old_w = keep.co.w

                keep.co = (
                    mid.x,
                    mid.y,
                    mid.z,
                    old_w
                )

            # Delete only the second point after moving the survivor to center.
            self._set_curve_point_selection(
                {kill_i}
            )
            self._topology_undo_push()

            try:
                bpy.ops.curve.delete(
                    type='VERT'
                )
            except Exception as exc:
                self.report(
                    {'WARNING'},
                    f"Curve Merge failed: {exc}"
                )
                return False

            self._ensure_curve_selection_after_topology(
                context
            )
            self._finish_after_topology(
                context
            )
            return True

        return False

    def _apply_shape_tool(
        self,
        context,
        tool
    ):
        self._refresh_selection_from_blender()

        order, full_loop = self._selected_order()

        if order is None or len(order) < 2:
            self.report(
                {'WARNING'},
                "Shape Tool: select one connected control-point range"
            )
            return

        points = self._panel_points()

        targets = _shape_targets_2d(
            points,
            order,
            tool,
            closed=bool(full_loop)
        )

        if not targets:
            self.report(
                {'INFO'},
                "Shape Tool: current selection is not suitable"
            )
            return

        self._push_history()
        base = self._snapshot_all()

        for i, panel_p in targets.items():
            plane = self._mouse_to_plane(
                panel_p.x,
                panel_p.y
            )

            original_world = self._point_world(i)

            new_world = self._plane_to_world(
                plane,
                original_world
            )

            self._set_point_world_from_base(
                i,
                base[i],
                new_world
            )

        self._obj.data.update_tag()
        context.area.tag_redraw()

    def _selected_order(self):
        n = self._point_count()

        if n < 2:
            return None, False

        sel = set(self._selected)

        if not sel:
            return list(range(n)), self._cyclic

        if len(sel) < 2:
            return None, False

        if not self._cyclic:
            lo = min(sel)
            hi = max(sel)

            if sel != set(
                range(lo, hi + 1)
            ):
                return None, False

            return list(
                range(lo, hi + 1)
            ), False

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
            current = _moving_average(
                current,
                radius=1,
                passes=1,
                closed=closed
            )

        if frac > 1e-8:
            nxt = _moving_average(
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
        mx,
        my
    ):
        if (
            self._pen_drawing
            or len(self._pen_stroke) < 3
        ):
            _hp_smooth_debug(
                "CURVE PEN BEGIN rejected "
                f"drawing={self._pen_drawing} points={len(self._pen_stroke)}"
            )
            return False

        _, full_loop = self._selected_order()

        closed = bool(
            full_loop
            and len(self._pen_stroke) >= 3
            and (
                self._pen_stroke[-1]
                - self._pen_stroke[0]
            ).length <= 20.0
        )

        if len(self._pen_raw_stroke) != len(self._pen_stroke):
            self._pen_raw_stroke = [
                p.copy() for p in self._pen_stroke
            ]
            self._pen_smooth_level = 0.0

        self._pen_smooth_dragging = True
        self._pen_smooth_start_x = float(mx)
        self._pen_smooth_start_amount = float(
            self._pen_smooth_level
        )
        self._pen_smooth_amount = float(
            self._pen_smooth_level
        )
        self._smooth_hud_x = float(mx)
        self._smooth_hud_y = float(my)
        self._pen_smooth_base = [
            p.copy() for p in self._pen_raw_stroke
        ]
        self._pen_smooth_closed = closed
        self._pen_smooth_cancel_stroke = [
            p.copy() for p in self._pen_stroke
        ]
        self._pen_smooth_cancel_amount = float(
            self._pen_smooth_level
        )
        self._pen_smooth_last_delta_px = 0.0

        _hp_smooth_debug(
            "CURVE PEN BEGIN "
            f"points={len(self._pen_stroke)} "
            f"closed={closed} start_amount={self._pen_smooth_level:.3f}"
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

        amount = (
            self._pen_smooth_start_amount
            + (float(mx) - self._pen_smooth_start_x) / 15.0
        )
        amount = max(0.0, min(20.0, amount))

        self._pen_smooth_amount = amount
        self._smooth_hud_x = float(mx)
        self._smooth_hud_y = float(my)

        result = self._pen_smooth_result(
            self._pen_smooth_base,
            amount,
            self._pen_smooth_closed
        )

        max_delta = 0.0
        for a, b in zip(self._pen_smooth_base, result):
            max_delta = max(max_delta, (b - a).length)

        self._pen_smooth_last_delta_px = max_delta
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
            self._pen_stroke = [
                p.copy()
                for p in self._pen_smooth_cancel_stroke
            ]
            self._pen_smooth_level = (
                self._pen_smooth_cancel_amount
            )
            self._pen_smooth_amount = (
                self._pen_smooth_cancel_amount
            )

        _hp_smooth_debug(
            "CURVE PEN END "
            f"cancel={cancel} amount={self._pen_smooth_amount:.3f} "
            f"max_delta_px={self._pen_smooth_last_delta_px:.3f}"
        )

        self._pen_smooth_dragging = False
        self._pen_smooth_base = []
        self._pen_smooth_cancel_stroke = []
        self._pen_smooth_closed = False

        context.area.tag_redraw()

    def _smooth_drawn_pen_once(self):
        if len(self._pen_stroke) < 3:
            return False

        if len(self._pen_raw_stroke) != len(self._pen_stroke):
            self._pen_raw_stroke = [
                p.copy() for p in self._pen_stroke
            ]
            self._pen_smooth_level = 0.0

        _, full_loop = self._selected_order()
        closed = bool(
            full_loop
            and (
                self._pen_stroke[-1]
                - self._pen_stroke[0]
            ).length <= 20.0
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
        global _CURVE_PEN_ANCHOR_DEFAULT

        self._pen_anchor_mode = (
            'POINT'
            if self._pen_anchor_mode == 'STROKE'
            else 'STROKE'
        )

        _CURVE_PEN_ANCHOR_DEFAULT = self._pen_anchor_mode

    def _activate_pen(self):
        # Re-read Blender's current selection at Pen entry so the editor
        # cannot use a stale internal selection after external clicks.
        self._refresh_selection_from_blender()

        self._pen_mode = True
        self._pen_drawing = False
        self._pen_stroke = []

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

        if self._pen_anchor_mode == 'POINT':
            offset = target_panel[0] - smooth[0]

            smooth = [
                p + offset
                for p in smooth
            ]

        return smooth

    def _apply_pen(self, context):
        if len(self._pen_stroke) < 2:
            return

        self._refresh_selection_from_blender()
        order, full_loop = self._selected_order()

        if order is None or len(order) < 2:
            self.report(
                {'WARNING'},
                "Curve Pen: select one connected control-point range (or no points for whole spline)"
            )
            return

        explicit_closed = (
            len(self._pen_stroke) >= 3
            and (
                self._pen_stroke[-1]
                - self._pen_stroke[0]
            ).length <= 20.0
        )

        if full_loop and not explicit_closed:
            self.report(
                {'WARNING'},
                "Whole cyclic spline: close the pen stroke"
            )
            return

        closed = bool(
            full_loop and explicit_closed
        )

        smooth = _stabilize(
            self._pen_stroke,
            level=self._pen_stabilizer,
            closed=closed
        )

        panel_points = self._panel_points()
        target_panel = [
            panel_points[i]
            for i in order
        ]

        smooth = self._prepare_pen_stroke(
            smooth,
            target_panel,
            closed
        )

        t_values = _path_t_values(
            target_panel,
            closed=closed
        )

        samples = _sample_polyline(
            smooth,
            t_values,
            closed=closed
        )

        self._push_history()

        base = self._snapshot_all()

        for i, panel_p in zip(
            order,
            samples
        ):
            plane = self._mouse_to_plane(
                panel_p.x,
                panel_p.y
            )

            original_world = self._point_world(i)

            new_world = self._plane_to_world(
                plane,
                original_world
            )

            self._set_point_world_from_base(
                i,
                base[i],
                new_world
            )

        self._obj.data.update_tag()

        self._pen_mode = False
        self._pen_drawing = False
        self._pen_stroke = []
        self._pen_raw_stroke = []
        self._pen_smooth_level = 0.0

        context.area.tag_redraw()

    # ---------------------------------------------------------
    # Modal
    # ---------------------------------------------------------
    def modal(self, context, event):
        if getattr(self, "_finished", False):
            return {'FINISHED'}

        if not self._owner_workspace_active(context):
            return self._finish(
                context,
                release_workspace=False
            )

        if not self._owner_context_active(context):
            self._viewport_navigation_active = False
            return {'PASS_THROUGH'}

        active = (self._dragging or self._box_dragging or self._transform_mode
                  or self._smooth_dragging or self._pen_smooth_dragging or self._pen_drawing
                  or self._viewport_navigation_active or self._f_hold_active or getattr(self, '_hp_confirm_hold', None)
                  or getattr(self, '_hp_menu', None) is not None)
        if event.type != 'TIMER' and not self._inside(event.mouse_region_x, event.mouse_region_y) and not active:
            return {'PASS_THROUGH'}

        # Keep Blender viewport orbit, pan, and modified-wheel navigation
        # available above the curve editor overlay.
        if event.type == 'MIDDLEMOUSE':
            self._viewport_navigation_active = event.value != 'RELEASE'
            return {'PASS_THROUGH'}
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

        if (
            context.edit_object is None
            or context.edit_object != self._obj
            or self._obj.mode != 'EDIT'
        ):
            return self._finish(context)

        result = self._hp_pen_key(context, event)
        if result is not None:
            return result

        # Esc / Shift+F always leaves Pen, including during E-smoothing.
        # It discards only the uncommitted stroke and keeps the editor open.
        if (event.value == 'PRESS'
                and (event.type == 'ESC' or
                     (event.type == 'F' and event.shift and not event.ctrl and not event.alt))
                and (self._pen_mode or self._pen_smooth_dragging)):
            self._pen_mode = False
            self._pen_drawing = False
            self._pen_stroke = []
            self._pen_raw_stroke = []
            self._pen_smooth_level = 0.0
            self._pen_smooth_dragging = False
            self._pen_smooth_base = []
            self._pen_smooth_cancel_stroke = []
            self._f_hold_active = False
            self._f_long_opened = False
            self._closed_pen_fit_cache = None
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        mx = event.mouse_region_x
        my = event.mouse_region_y
        inside = self._inside(mx, my)

        # In Pen mode, E-drag visually smooths the not-yet-applied stroke.
        if self._pen_smooth_dragging:
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
                'WHEELDOWNMOUSE',
            }:
                if event.alt:
                    delta = (
                        0.08
                        if event.type == 'WHEELUPMOUSE'
                        else -0.08
                    )
                    self._smooth_brush_strength = max(
                        0.05,
                        min(1.0, self._smooth_brush_strength + delta),
                    )
                else:
                    delta = (
                        10.0
                        if event.type == 'WHEELUPMOUSE'
                        else -10.0
                    )
                    self._smooth_brush_radius = max(
                        20.0,
                        min(240.0, self._smooth_brush_radius + delta),
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

        # Pen active:
        # - after a stroke is drawn, F confirms/applies it
        # - before drawing, F switches Stroke Start / Point Start
        if (
            event.type == 'F'
            and not event.ctrl
            and event.value == 'PRESS'
            and self._pen_mode
        ):
            if bool(getattr(event, "is_repeat", False)):
                return {'RUNNING_MODAL'}

            if (
                self._pen_stroke
                and not self._pen_drawing
            ):
                self._apply_pen(context)
                return {'RUNNING_MODAL'}

            self._toggle_pen_anchor_mode()
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # Pen inactive:
        # quick F enters pen, long F opens the mode pie.
        if event.type == 'F' and not event.ctrl and not self._pen_mode:
            if event.value == 'PRESS' and inside:
                if not self._f_hold_active:
                    self._f_hold_active = True
                    self._f_hold_started = time.perf_counter()
                    self._f_long_opened = False

                return {'RUNNING_MODAL'}

            if event.value == 'RELEASE' and self._f_hold_active:
                long_opened = self._f_long_opened

                self._f_hold_active = False
                self._f_long_opened = False

                if not long_opened:
                    self._activate_pen()
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

            bpy.ops.wm.call_menu_pie(
                name=HP_MT_curve_pen_anchor_pie.bl_idname
            )

            return {'RUNNING_MODAL'}

        if (
            event.ctrl
            and event.type == 'Z'
            and event.value == 'PRESS'
        ):
            self._undo_local(context)
            return {'RUNNING_MODAL'}

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

            if (
                event.value == 'PRESS'
                and event.type in {
                    'X',
                    'Y',
                    'Z'
                }
            ):
                if self._transform_axis == event.type:
                    self._transform_axis = None
                else:
                    self._transform_axis = event.type

                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'MOUSEMOVE':
                self._apply_transform(
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
                self._transform_mode = None
                self._transform_axis = None
                return {'RUNNING_MODAL'}

            if (
                event.value == 'PRESS'
                and event.type in {
                    'RIGHTMOUSE',
                    'ESC'
                }
            ):
                self._transform_mode = None
                self._transform_axis = None
                self._undo_local(context)
                return {'RUNNING_MODAL'}

            return {'RUNNING_MODAL'}

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
                self._smooth_drawn_pen_once()
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            # RMB:
            # - no stroke -> leave Pen mode
            # - drawn stroke -> clear current stroke and stay in Pen mode
            if event.type == 'RIGHTMOUSE' and event.value == 'PRESS':
                if self._pen_stroke:
                    self._pen_stroke = []
                    self._pen_raw_stroke = []
                    self._pen_smooth_level = 0.0
                    self._pen_drawing = False
                else:
                    self._pen_mode = False
                    self._pen_drawing = False
                    self._pen_stroke = []
                    self._pen_raw_stroke = []
                    self._pen_smooth_level = 0.0

                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.value == 'PRESS':
                if event.type == 'F':
                    self._pen_mode = False
                    self._pen_drawing = False
                    self._pen_stroke = []
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}

                if event.type == 'WHEELUPMOUSE':
                    self._pen_stabilizer = min(
                        5,
                        self._pen_stabilizer + 1
                    )
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}

                if event.type == 'WHEELDOWNMOUSE':
                    self._pen_stabilizer = max(
                        0,
                        self._pen_stabilizer - 1
                    )
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}

                if event.type in {
                    'RET',
                    'NUMPAD_ENTER'
                }:
                    self._apply_pen(context)
                    return {'RUNNING_MODAL'}

                if event.type == 'ESC':
                    self._pen_mode = False
                    self._pen_drawing = False
                    self._pen_stroke = []
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}

            if (
                event.type == 'LEFTMOUSE'
                and event.value == 'PRESS'
                and inside
            ):
                self._pen_stroke = [
                    Vector((mx, my))
                ]
                self._pen_raw_stroke = []
                self._pen_smooth_level = 0.0
                self._pen_drawing = True
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if (
                event.type == 'MOUSEMOVE'
                and self._pen_drawing
            ):
                p = Vector((mx, my))

                if (
                    not self._pen_stroke
                    or (
                        p - self._pen_stroke[-1]
                    ).length >= 1.5
                ):
                    self._pen_stroke.append(p)

                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if (
                event.type == 'LEFTMOUSE'
                and event.value == 'RELEASE'
                and self._pen_drawing
            ):
                p = Vector((mx, my))

                if (
                    not self._pen_stroke
                    or (
                        p - self._pen_stroke[-1]
                    ).length >= 0.75
                ):
                    self._pen_stroke.append(p)

                self._pen_drawing = False
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if inside:
                return {'RUNNING_MODAL'}

        if self._dragging:
            if event.type == 'MOUSEMOVE':
                self._apply_drag(
                    context,
                    mx,
                    my
                )
                return {'RUNNING_MODAL'}

            if (
                event.type == 'LEFTMOUSE'
                and event.value == 'RELEASE'
            ):
                self._apply_drag(
                    context,
                    mx,
                    my
                )
                self._dragging = False
                return {'RUNNING_MODAL'}

        if self._box_dragging:
            if event.type == 'MOUSEMOVE':
                self._box_end = Vector((mx, my))
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if (
                event.type == 'LEFTMOUSE'
                and event.value == 'RELEASE'
            ):
                self._box_end = Vector((mx, my))
                self._apply_box()

                self._box_dragging = False
                self._box_start = None
                self._box_end = None

                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

        if inside and event.value == 'PRESS':
            if event.type == 'NUMPAD_1':
                self._set_view(
                    'BACK'
                    if event.ctrl
                    else 'FRONT'
                )
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'NUMPAD_3':
                self._set_view(
                    'LEFT'
                    if event.ctrl
                    else 'RIGHT'
                )
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'NUMPAD_7':
                self._set_view(
                    'BOTTOM'
                    if event.ctrl
                    else 'TOP'
                )
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'NUMPAD_2':
                self._set_view('TOP')
                self._flip_x = False
                self._flip_y = False
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'NUMPAD_4':
                self._set_view('TOP_X_DOWN')
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'NUMPAD_6':
                self._set_view('TOP_Y_DOWN')
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'NUMPAD_8':
                if event.ctrl:
                    self._flip_x = not self._flip_x
                else:
                    self._flip_y = not self._flip_y
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.alt and event.type == 'Z':
                self._xray = not self._xray
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
                    name=HP_MT_curve_topology_pie.bl_idname
                )
                return {'RUNNING_MODAL'}

            if (
                event.type == 'E'
                and not event.ctrl
                and event.value == 'PRESS'
            ):
                self._begin_smooth_drag(
                    context,
                    mx,
                    my
                )
                return {'RUNNING_MODAL'}

            if (
                event.type == 'F'
                and event.ctrl
                and self._selected
            ):
                bpy.ops.wm.call_menu_pie(
                    name=HP_MT_curve_shape_pie.bl_idname
                )
                return {'RUNNING_MODAL'}

            if event.type == 'F' and not event.ctrl:
                self._pen_mode = True
                self._pen_drawing = False
                self._pen_stroke = []
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if (
                event.type in {'G', 'S', 'R'}
                and self._selected
            ):
                self._begin_transform(
                    event.type,
                    mx,
                    my
                )
                return {'RUNNING_MODAL'}

            if event.type == 'A':
                self._selected = set(
                    range(self._point_count())
                )
                self._sync_blender_selection()
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
                    self._zoom = min(
                        5.0,
                        self._zoom * 1.15
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
                    self._zoom = max(
                        0.35,
                        self._zoom / 1.15
                    )

                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

        if (
            inside
            and event.type == 'LEFTMOUSE'
            and event.value == 'PRESS'
        ):
            idx = self._pick_point(
                mx,
                my
            )

            if idx is not None:
                if event.shift:
                    if idx in self._selected:
                        self._selected.remove(idx)
                    else:
                        self._selected.add(idx)
                else:
                    if idx not in self._selected:
                        self._selected = {idx}

                self._sync_blender_selection()

                if self._selected:
                    self._begin_drag(
                        mx,
                        my
                    )

                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            self._box_dragging = True
            self._box_start = Vector((mx, my))
            self._box_end = Vector((mx, my))
            self._box_additive = bool(event.shift)

            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if (
            event.type == 'ESC'
            and event.value == 'PRESS'
        ):
            return self._finish(context)

        if event.type == 'TIMER':
            if self._topology_rebuild_pending:
                if self._refresh_after_topology(
                    context
                ):
                    self._topology_rebuild_pending = False

                context.area.tag_redraw()
                return {'PASS_THROUGH'}

            # Autostart still requires a selected range. Once this editor is
            # open, however, zero selected points is a valid editing state.
            live_signature = _curve_live_signature(
                context,
                self._obj,
                self._spline_index
            )

            if live_signature is None:
                return self._finish(
                    context,
                    release_workspace=True
                )

            if live_signature != self._signature:
                return self._finish(
                    context,
                    release_workspace=False
                )

            context.area.tag_redraw()
            return {'PASS_THROUGH'}

        if inside:
            return {'RUNNING_MODAL'}

        return {'PASS_THROUGH'}

    # ---------------------------------------------------------
    # Drawing
    # ---------------------------------------------------------
    def _draw_rect(
        self,
        x,
        y,
        w,
        h,
        color
    ):
        shader = gpu.shader.from_builtin(
            'UNIFORM_COLOR'
        )

        batch = batch_for_shader(
            shader,
            'TRI_FAN',
            {
                "pos": [
                    (x, y),
                    (x + w, y),
                    (x + w, y + h),
                    (x, y + h),
                ]
            }
        )

        shader.bind()
        shader.uniform_float(
            "color",
            color
        )
        batch.draw(shader)

    def _draw_rect_outline(
        self,
        x1,
        y1,
        x2,
        y2,
        color
    ):
        shader = gpu.shader.from_builtin(
            'UNIFORM_COLOR'
        )

        batch = batch_for_shader(
            shader,
            'LINE_STRIP',
            {
                "pos": [
                    (x1, y1),
                    (x2, y1),
                    (x2, y2),
                    (x1, y2),
                    (x1, y1),
                ]
            }
        )

        gpu.state.line_width_set(1.5)
        shader.bind()
        shader.uniform_float(
            "color",
            color
        )
        batch.draw(shader)

    def _draw_line(
        self,
        points,
        color,
        width=2.0,
        closed=False
    ):
        if len(points) < 2:
            return

        pts = [
            (p.x, p.y)
            for p in points
        ]

        if closed:
            pts.append(pts[0])

        shader = gpu.shader.from_builtin(
            'UNIFORM_COLOR'
        )

        batch = batch_for_shader(
            shader,
            'LINE_STRIP',
            {"pos": pts}
        )

        gpu.state.line_width_set(width)
        shader.bind()
        shader.uniform_float(
            "color",
            color
        )
        batch.draw(shader)

    def _draw_points(
        self,
        points,
        color,
        size
    ):
        if not points:
            return

        shader = gpu.shader.from_builtin(
            'UNIFORM_COLOR'
        )

        batch = batch_for_shader(
            shader,
            'POINTS',
            {
                "pos": [
                    (p.x, p.y)
                    for p in points
                ]
            }
        )

        gpu.state.point_size_set(size)
        shader.bind()
        shader.uniform_float(
            "color",
            color
        )
        batch.draw(shader)

    def _draw_circle(
        self,
        center,
        radius,
        color,
        segments=64
    ):
        if radius <= 0.0:
            return

        from math import cos, sin, tau

        pts = [
            (
                center.x + cos(
                    tau * i / segments
                ) * radius,
                center.y + sin(
                    tau * i / segments
                ) * radius,
            )
            for i in range(segments + 1)
        ]

        shader = gpu.shader.from_builtin(
            'UNIFORM_COLOR'
        )

        batch = batch_for_shader(
            shader,
            'LINE_STRIP',
            {"pos": pts}
        )

        gpu.state.line_width_set(1.25)
        shader.bind()
        shader.uniform_float(
            "color",
            color
        )
        batch.draw(shader)


    def _curve_display_world(self):
        mw = self._obj.matrix_world

        if not self._bezier:
            return [
                mw @ self._point_local(i)
                for i in range(self._point_count())
            ]

        bps = self._spline.bezier_points
        result = []

        seg_count = (
            len(bps)
            if self._cyclic
            else len(bps) - 1
        )

        for i in range(seg_count):
            j = (i + 1) % len(bps)

            segment = interpolate_bezier(
                bps[i].co,
                bps[i].handle_right,
                bps[j].handle_left,
                bps[j].co,
                12
            )

            if result and segment:
                segment = segment[1:]

            result.extend(
                mw @ p
                for p in segment
            )

        return result

    def _display_curve_panel(self):
        if not self._view_valid:
            self._refit()

        scale = self._scale()
        sy = -1.0 if self._flip_y else 1.0

        return [
            self._panel_center
            + Vector((
                (
                    self._world_to_plane(world).x
                    - self._view_center.x
                ) * scale,
                (
                    self._world_to_plane(world).y
                    - self._view_center.y
                ) * scale * sy,
            ))
            for world in self._curve_display_world()
        ]

    def _draw_axis(self):
        origin = Vector((
            self._panel_x + self._panel_w - 76.0,
            self._panel_y + 88.0
        ))

        x_color = (
            1.0,
            0.32,
            0.32,
            0.95
        )
        y_color = (
            0.35,
            1.0,
            0.42,
            0.95
        )
        z_color = (
            0.38,
            0.62,
            1.0,
            0.95
        )

        if self._view == 'FRONT':
            h_dir = Vector((1.0, 0.0))
            h_label = 'X'
            h_color = x_color

        elif self._view == 'BACK':
            h_dir = Vector((-1.0, 0.0))
            h_label = 'X'
            h_color = x_color

        elif self._view == 'RIGHT':
            h_dir = Vector((1.0, 0.0))
            h_label = 'Y'
            h_color = y_color

        elif self._view == 'LEFT':
            h_dir = Vector((-1.0, 0.0))
            h_label = 'Y'
            h_color = y_color

        elif self._view == 'TOP':
            h_dir = Vector((1.0, 0.0))
            h_label = 'X'
            h_color = x_color

        elif self._view == 'BOTTOM':
            h_dir = Vector((-1.0, 0.0))
            h_label = 'X'
            h_color = x_color

        elif self._view == 'TOP_X_DOWN':
            h_dir = Vector((1.0, 0.0))
            h_label = 'Y'
            h_color = y_color

        else:
            h_dir = Vector((1.0, 0.0))
            h_label = 'X'
            h_color = x_color

        self._draw_arrow(
            origin,
            h_dir,
            h_label,
            h_color
        )

        vertical_dir = Vector((
            0.0,
            -1.0 if self._flip_y else 1.0
        ))

        vertical_label = (
            'Y'
            if self._view in {
                'TOP',
                'BOTTOM',
                'TOP_X_DOWN',
                'TOP_Y_DOWN',
            }
            else 'Z'
        )
        vertical_color = (
            y_color
            if vertical_label == 'Y'
            else z_color
        )

        self._draw_arrow(
            origin,
            vertical_dir,
            vertical_label,
            vertical_color
        )

        self._draw_points(
            [origin],
            (0.95, 0.95, 0.95, 0.95),
            5.0
        )

    def _draw_arrow(
        self,
        origin,
        direction,
        label,
        color
    ):
        length = 34.0
        end = origin + direction * length
        perp = Vector(
            (-direction.y, direction.x)
        )
        back = end - direction * 8.0

        self._draw_line(
            [origin, end],
            color,
            2.5
        )

        self._draw_line(
            [
                end,
                back + perp * 4.0,
            ],
            color,
            2.0
        )

        self._draw_line(
            [
                end,
                back - perp * 4.0,
            ],
            color,
            2.0
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

        radius = (
            70.0
            if self._pen_smooth_dragging
            else self._smooth_brush_radius
        )
        strength = (
            0.55
            if self._pen_smooth_dragging
            else self._smooth_brush_strength
        )

        self._draw_circle(
            Vector((self._smooth_hud_x, self._smooth_hud_y)),
            radius,
            (0.25, 0.95, 0.70, 0.85),
        )

        x = self._smooth_hud_x + 18
        y = self._smooth_hud_y + 18

        # Tiny local backplate.
        shader = gpu.shader.from_builtin(
            'UNIFORM_COLOR'
        )
        batch = batch_for_shader(
            shader,
            'TRI_FAN',
            {
                "pos": [
                    (x - 7, y - 7),
            (x + 210, y - 7),
            (x + 210, y + 21),
                    (x - 7, y + 21),
                ]
            }
        )
        shader.bind()
        shader.uniform_float(
            "color",
            (0.02, 0.02, 0.02, 0.82)
        )
        batch.draw(shader)

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
            f"Smooth  Str {strength:.2f}  R {radius:.0f}"
        )

    def _draw(self):
        if getattr(self, "_finished", False):
            return

        if not self._owner_context_active(
            bpy.context
        ):
            return

        try:
            live_signature = _curve_live_signature(
                bpy.context,
                self._obj,
                self._spline_index
            )

            if live_signature != self._signature:
                return
        except Exception:
            return

        gpu.state.blend_set('ALPHA')

        body = (
            0.025,
            0.025,
            0.025,
            0.62
        )
        header = (
            0.085,
            0.085,
            0.085,
            0.86
        )
        border = (
            0.65,
            0.65,
            0.65,
            0.42
        )

        self._draw_rect(
            self._panel_x,
            self._panel_y,
            self._panel_w,
            self._panel_h,
            body
        )

        self._draw_rect(
            self._panel_x + 1,
            self._panel_y + self._panel_h - 34,
            self._panel_w - 2,
            33,
            header
        )

        self._draw_rect_outline(
            self._panel_x,
            self._panel_y,
            self._panel_x + self._panel_w,
            self._panel_y + self._panel_h,
            border
        )

        points = self._panel_points()
        curve_line = self._display_curve_panel()

        self._draw_line(
            curve_line,
            (0.50, 0.76, 1.0, 1.0),
            2.0,
            self._cyclic
            and not self._bezier
        )

        front = set(
            self._front_filter(
                range(len(points)),
                points
            )
        )

        if self._xray:
            front = set(
                range(len(points))
            )

        rear_unselected = [
            points[i]
            for i in range(len(points))
            if i not in front
            and i not in self._selected
        ]

        front_unselected = [
            points[i]
            for i in range(len(points))
            if i in front
            and i not in self._selected
        ]

        selected = [
            points[i]
            for i in range(len(points))
            if i in self._selected
        ]

        if rear_unselected:
            self._draw_points(
                rear_unselected,
                (1.0, 0.68, 0.18, 0.28),
                5.0
            )

        self._draw_points(
            front_unselected,
            (1.0, 0.68, 0.18, 1.0),
            7.0
        )

        self._draw_points(
            selected,
            (0.25, 1.0, 0.42, 1.0),
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
            self._box_dragging
            and self._box_start is not None
            and self._box_end is not None
        ):
            self._draw_rect_outline(
                self._box_start.x,
                self._box_start.y,
                self._box_end.x,
                self._box_end.y,
                (
                    0.3,
                    0.8,
                    1.0,
                    0.95
                )
            )

        if self._pen_stroke and getattr(self, '_hp_menu', None) is None:
            order, full_loop = self._selected_order()

            closed = bool(
                full_loop
                and len(self._pen_stroke) >= 3
                and (
                    self._pen_stroke[-1]
                    - self._pen_stroke[0]
                ).length <= 20.0
            )

            preview = _stabilize(
                self._pen_stroke,
                level=self._pen_stabilizer,
                closed=closed
            )

            if order is not None and order:
                target_panel = [
                    points[i]
                    for i in order
                ]

                preview = self._prepare_pen_stroke(
                    preview,
                    target_panel,
                    closed
                )

            self._draw_line(
                preview,
                (1.0, 0.42, 0.08, 1.0),
                3.0,
                closed
            )

        self._draw_axis()

        xray = (
            "ON"
            if self._xray
            else "OFF"
        )

        blf.position(
            0,
            self._panel_x + 12,
            self._panel_y + self._panel_h - 23,
            0
        )

        blf.size(0, 14)
        blf.color(
            0,
            1.0,
            1.0,
            1.0,
            1.0
        )

        blf.draw(
            0,
            f"CURVE : {self._view}  [{self._view_shortcut()}]  |  X-Ray:{xray}"
        )

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

        blf.draw(
            0,
            f"Num1 正面 / Num3 側面 / Num7 上面 / Alt+Z 透過"
        )

        blf.position(
            0,
            self._panel_x + 12,
            self._panel_y + 29,
            0
        )

        blf.draw(
            0,
            "F ペン / E ブラシ / Ctrl+F 整形 / M 細分化"
        )

        self._draw_smooth_hud()

        self._hp_draw_menu(bpy.context)

        gpu.state.blend_set('NONE')

    def _finish(self, context, release_workspace=False):
        global _CURVE_RUNNING, _ACTIVE_CURVE_EDITOR, _CURVE_PINNED_WORKSPACE_PTR

        # Idempotent: unregister, Blender cancellation, and the final modal
        # event can all reach this method. A stale instance must not stop a
        # newly started editor.
        if getattr(self, '_finished', False):
            return {'FINISHED'}
        if getattr(self, '_hp_menu', None) is not None:
            try:
                self._hp_end_adjust(context, commit=False)
            except (ReferenceError, RuntimeError):
                self._hp_menu = None
        self._hp_confirm_hold = None

        self._finished = True

        handle = getattr(self, '_handle', None)
        self._handle = None
        if handle is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
            except (ReferenceError, RuntimeError, ValueError):
                pass  # Blender may already have removed the handler.

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

        if _ACTIVE_CURVE_EDITOR is self:
            _ACTIVE_CURVE_EDITOR = None
            _CURVE_RUNNING = False
            if release_workspace:
                _CURVE_PINNED_WORKSPACE_PTR = 0

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


def _curve_autostart():
    global _CURVE_PINNED_WORKSPACE_PTR

    if (
        _CURVE_RUNNING
        or bpy.context.window_manager is None
    ):
        return 0.35

    wm = bpy.context.window_manager

    for window in wm.windows:
        if (
            _CURVE_PINNED_WORKSPACE_PTR
            and (
                window.workspace is None
                or window.workspace.as_pointer()
                != _CURVE_PINNED_WORKSPACE_PTR
            )
        ):
            continue

        for area in window.screen.areas:
            if area.type != 'VIEW_3D':
                continue

            region = next(
                (
                    r
                    for r in area.regions
                    if r.type == 'WINDOW'
                ),
                None
            )

            if region is None:
                continue

            try:
                with bpy.context.temp_override(
                    window=window,
                    area=area,
                    region=region
                ):
                    if _curve_target(bpy.context) is not None:
                        bpy.ops.hp.curve_mini_editor(
                            'INVOKE_DEFAULT'
                        )
                        return 0.35
            except Exception:
                pass

    if _CURVE_PINNED_WORKSPACE_PTR:
        for window in wm.windows:
            if (
                window.workspace is not None
                and window.workspace.as_pointer()
                == _CURVE_PINNED_WORKSPACE_PTR
            ):
                _CURVE_PINNED_WORKSPACE_PTR = 0
                break

    return 0.35


classes = (
    HP_OT_curve_shape_choice,
    HP_MT_curve_shape_pie,
    HP_OT_curve_topology_choice,
    HP_MT_curve_topology_pie,
    HP_OT_curve_pen_anchor_choice,
    HP_MT_curve_pen_anchor_pie,
    HP_OT_curve_mini_editor,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)



def unregister():
    global _CURVE_RUNNING, _ACTIVE_CURVE_EDITOR, _CURVE_PINNED_WORKSPACE_PTR

    if bpy.app.timers.is_registered(_curve_autostart):
        bpy.app.timers.unregister(_curve_autostart)

    editor = _ACTIVE_CURVE_EDITOR
    if editor is not None:
        editor._finish(bpy.context, release_workspace=True)
    _ACTIVE_CURVE_EDITOR = None
    _CURVE_RUNNING = False
    _CURVE_PINNED_WORKSPACE_PTR = 0

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()

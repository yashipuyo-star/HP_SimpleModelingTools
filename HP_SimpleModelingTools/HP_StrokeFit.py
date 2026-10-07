
bl_info = {
    "name": "HP Stroke Fit",
    "author": "OpenAI + yashi",
    "version": (0, 26, 18),
    "blender": (4, 3, 0),
    "location": "3D View > Edit Mode > Shift+Alt+F",
    "description": "Quick Stroke Fit with projection planes, post-draw Smooth HUD, and F confirm.",
    "category": "Mesh",
}

import bpy
import bmesh
import blf
import gpu
import heapq
import math
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils
from mathutils import Vector
from mathutils.geometry import intersect_line_plane

addon_keymaps = []
_HP_ACTIVE_CURVE_PEN = None
_HP_LAST_PEN_RECORD = None
_HP_REOPEN_RECORD = None


def _hp_resample_screen_polyline(points, count):
    """Arc-length resample a freehand screen stroke to a fixed point count."""
    if not points:
        return []
    if len(points) == 1:
        return [points[0].copy() for _ in range(max(2, int(count)))]

    clean = [points[0].copy()]
    for p in points[1:]:
        if (p - clean[-1]).length > 1.0:
            clean.append(p.copy())
    if len(clean) < 2:
        return [clean[0].copy() for _ in range(max(2, int(count)))]

    lengths = [0.0]
    for a, b in zip(clean, clean[1:]):
        lengths.append(lengths[-1] + (b - a).length)
    total = lengths[-1]
    if total <= 1.0e-6:
        return [clean[0].copy() for _ in range(max(2, int(count)))]

    out = []
    seg = 0
    n = max(2, int(count))
    for i in range(n):
        target = total * (i / (n - 1))
        while seg < len(clean) - 2 and lengths[seg + 1] < target:
            seg += 1
        a = clean[seg]
        b = clean[seg + 1]
        span = lengths[seg + 1] - lengths[seg]
        fac = 0.0 if span <= 1.0e-8 else (target - lengths[seg]) / span
        out.append(a.lerp(b, max(0.0, min(1.0, fac))))
    return out


def _hp_surface_mesh_graph(bm, obj):
    """Snapshot an edit mesh as a world-space weighted vertex graph."""
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.index_update()
    bm.normal_update()

    normal_matrix = obj.matrix_world.to_3x3().inverted().transposed()
    coords = {}
    normals = {}
    adjacency = {vert.index: [] for vert in bm.verts}

    for vert in bm.verts:
        index = vert.index
        world_co = obj.matrix_world @ vert.co
        world_normal = normal_matrix @ vert.normal
        if world_normal.length > 1e-10:
            world_normal.normalize()
        else:
            world_normal = Vector((0.0, 0.0, 1.0))
        coords[index] = world_co.copy()
        normals[index] = world_normal.copy()

    for edge in bm.edges:
        a, b = edge.verts
        ia, ib = a.index, b.index
        length = (coords[ia] - coords[ib]).length
        adjacency[ia].append((ib, length))
        adjacency[ib].append((ia, length))
    return adjacency, coords, normals


def _hp_surface_dijkstra(adjacency, start, goal):
    """Return the shortest edge route, or None when the vertices disconnect."""
    if start not in adjacency or goal not in adjacency:
        return None
    if start == goal:
        return [start]

    distances = {start: 0.0}
    previous = {}
    queue = [(0.0, start)]
    while queue:
        distance, current = heapq.heappop(queue)
        if distance != distances.get(current):
            continue
        if current == goal:
            break
        for neighbor, weight in adjacency[current]:
            candidate = distance + max(0.0, float(weight))
            if candidate < distances.get(neighbor, float('inf')):
                distances[neighbor] = candidate
                previous[neighbor] = current
                heapq.heappush(queue, (candidate, neighbor))

    if goal not in distances:
        return None
    route = [goal]
    while route[-1] != start:
        parent = previous.get(route[-1])
        if parent is None:
            return None
        route.append(parent)
    route.reverse()
    return route


def _hp_selected_surface_chain(bm, selected_indices, active_index=None):
    """Order selected vertices when they form one open, unbranched chain."""
    selected = set(selected_indices)
    if len(selected) < 3:
        return None

    adjacency = {}
    for index in selected:
        vert = bm.verts[index]
        adjacency[index] = sorted(
            edge.other_vert(vert).index
            for edge in vert.link_edges
            if edge.other_vert(vert).index in selected
        )

    if any(len(neighbors) > 2 for neighbors in adjacency.values()):
        return None
    endpoints = [index for index, neighbors in adjacency.items() if len(neighbors) == 1]
    if len(endpoints) != 2:
        return None

    seen = set()
    stack = [endpoints[0]]
    while stack:
        index = stack.pop()
        if index in seen:
            continue
        seen.add(index)
        stack.extend(neighbor for neighbor in adjacency[index] if neighbor not in seen)
    if seen != selected:
        return None

    start = min(endpoints)
    if active_index in endpoints:
        start = endpoints[1] if endpoints[0] == active_index else endpoints[0]

    ordered = [start]
    previous = None
    current = start
    while True:
        candidates = [index for index in adjacency[current] if index != previous]
        if not candidates:
            break
        following = candidates[0]
        if following in ordered:
            return None
        ordered.append(following)
        previous, current = current, following

    if len(ordered) != len(selected) or ordered[-1] not in endpoints:
        return None
    return ordered


def _hp_surface_curve_tilts(samples):
    """Rotate each curve frame toward the local scalp normal."""
    if len(samples) < 2:
        return [0.0] * len(samples)

    world_up = Vector((0.0, 0.0, 1.0))
    alternate_up = Vector((0.0, 1.0, 0.0))
    tilts = []
    previous_angle = None
    previous_normal = None

    for index, (point, source_normal) in enumerate(samples):
        if index == 0:
            tangent = samples[1][0] - point
        elif index == len(samples) - 1:
            tangent = point - samples[index - 1][0]
        else:
            tangent = samples[index + 1][0] - samples[index - 1][0]
        if tangent.length < 1e-10:
            tangent = Vector((1.0, 0.0, 0.0))
        else:
            tangent.normalize()

        normal = source_normal.copy()
        if previous_normal is not None and normal.dot(previous_normal) < 0.0:
            normal.negate()
        normal -= tangent * normal.dot(tangent)
        if normal.length < 1e-10 and previous_normal is not None:
            normal = previous_normal.copy()
            normal -= tangent * normal.dot(tangent)
        if normal.length < 1e-10:
            normal = world_up - tangent * world_up.dot(tangent)
        if normal.length < 1e-10:
            normal = alternate_up - tangent * alternate_up.dot(tangent)
        if normal.length < 1e-10:
            normal = Vector((1.0, 0.0, 0.0))
        normal.normalize()
        previous_normal = normal.copy()

        reference = world_up - tangent * world_up.dot(tangent)
        if reference.length < 1e-10:
            reference = alternate_up - tangent * alternate_up.dot(tangent)
        reference.normalize()
        angle = math.atan2(tangent.dot(reference.cross(normal)), reference.dot(normal))
        if previous_angle is not None:
            while angle - previous_angle > math.pi:
                angle -= math.tau
            while angle - previous_angle < -math.pi:
                angle += math.tau
        tilts.append(angle)
        previous_angle = angle

    return tilts


def _hp_create_surface_curve(context, source_obj, samples, anchor_vertices=()):
    """Create an unoffset poly curve with tilt aligned to scalp normals."""
    clean = []
    for co, normal in samples:
        co = co.copy()
        normal = normal.copy()
        if not clean or (co - clean[-1][0]).length > 1e-7:
            clean.append((co, normal))
    if len(clean) < 2:
        return None

    curve_data = bpy.data.curves.new("HP Hair Path", "CURVE")
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 12
    curve_data.render_resolution_u = 16
    try:
        curve_data.twist_mode = 'Z_UP'
    except (AttributeError, TypeError, ValueError):
        pass

    # NURBS smooth the bends between selected scalp vertices while keeping
    # the route's sampled points and endpoint anchors available as controls.
    spline = curve_data.splines.new('NURBS')
    spline.points.add(len(clean) - 1)
    spline.order_u = min(4, len(clean))
    spline.use_endpoint_u = True
    tilts = _hp_surface_curve_tilts(clean)
    for point, (co, _normal), tilt in zip(spline.points, clean, tilts):
        point.co = (co.x, co.y, co.z, 1.0)
        point.tilt = tilt

    curve_obj = bpy.data.objects.new("HP Hair Path", curve_data)
    collection = (
        source_obj.users_collection[0]
        if source_obj.users_collection
        else context.scene.collection
    )
    collection.objects.link(curve_obj)
    curve_obj.show_in_front = True
    curve_obj["hp_pathpen_surface_path"] = True
    curve_obj["hp_pathpen_source_object"] = source_obj.name
    curve_obj["hp_pathpen_anchor_vertices"] = [int(index) for index in anchor_vertices]
    context.view_layer.update()
    return curve_obj


def _mesh_chain(context):
    obj = context.edit_object

    if (
        not obj
        or obj.type != 'MESH'
        or obj.mode != 'EDIT'
    ):
        return None

    bm = bmesh.from_edit_mesh(obj.data)

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    bm.verts.index_update()
    bm.edges.index_update()

    edges = [
        e for e in bm.edges
        if e.select
    ]

    if not edges:
        return None

    adj = {}

    for e in edges:
        a, b = e.verts

        adj.setdefault(
            a.index,
            []
        ).append(
            b.index
        )

        adj.setdefault(
            b.index,
            []
        ).append(
            a.index
        )

    if any(
        len(v) > 2
        for v in adj.values()
    ):
        return None

    verts_in = set(adj.keys())

    seen = set()
    stack = [
        next(iter(verts_in))
    ]

    while stack:
        vi = stack.pop()

        if vi in seen:
            continue

        seen.add(vi)

        stack.extend(
            n for n in adj[vi]
            if n not in seen
        )

    if seen != verts_in:
        return None

    endpoints = [
        vi
        for vi, links in adj.items()
        if len(links) == 1
    ]

    if len(endpoints) != 2:
        return None

    start = min(endpoints)

    ordered = [start]
    prev = None
    cur = start

    while True:
        candidates = [
            n
            for n in adj[cur]
            if n != prev
        ]

        if not candidates:
            break

        nxt = candidates[0]

        if nxt in ordered:
            break

        ordered.append(nxt)

        prev, cur = cur, nxt

    if len(ordered) != len(verts_in):
        return None

    return {
        "kind": "MESH",
        "obj": obj,
        "bm": bm,
        "ordered": ordered,
    }


def _curve_selection(context):
    obj = context.edit_object

    if (
        not obj
        or obj.type != 'CURVE'
        or obj.mode != 'EDIT'
    ):
        return None

    selected_splines = []

    for spline_index, spline in enumerate(obj.data.splines):

        if spline.type == 'BEZIER':

            ids = [
                i
                for i, bp in enumerate(spline.bezier_points)
                if bp.select_control_point
            ]

        else:

            ids = [
                i
                for i, p in enumerate(spline.points)
                if p.select
            ]

        if ids:
            selected_splines.append(
                (
                    spline_index,
                    spline,
                    ids
                )
            )

    if len(selected_splines) != 1:
        return None

    spline_index, spline, ids = selected_splines[0]

    if len(ids) < 2:
        return None

    # v004: one contiguous range only.
    ids_sorted = sorted(ids)

    if ids_sorted != list(
        range(
            ids_sorted[0],
            ids_sorted[-1] + 1
        )
    ):
        return None

    return {
        "kind": "CURVE",
        "obj": obj,
        "spline_index": spline_index,
        "spline": spline,
        "ordered": ids_sorted,
        "bezier": (
            spline.type == 'BEZIER'
        ),
    }


def _detect_target(context):
    obj = context.edit_object

    if not obj:
        return None

    if obj.type == 'MESH':
        return _mesh_chain(context)

    if obj.type == 'CURVE':
        return _curve_selection(context)

    return None


def _resample_spacing(points, spacing=3.0):
    if len(points) < 2:
        return [
            Vector(p)
            for p in points
        ]

    pts = [
        Vector(p)
        for p in points
    ]

    out = [
        pts[0].copy()
    ]

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

            cur = (
                cur
                + direction * need
            )

            out.append(
                cur.copy()
            )

            dist -= need
            carry = 0.0

            if dist <= 1e-8:
                break

            direction = (
                target - cur
            ).normalized()

        if dist > 1e-8:
            carry += dist
            cur = target.copy()

    if (
        out[-1] - pts[-1]
    ).length > 0.5:
        out.append(
            pts[-1].copy()
        )

    return out


def _moving_average(points, radius=2, passes=2):
    pts = [
        Vector(p)
        for p in points
    ]

    if (
        len(pts) < 3
        or radius <= 0
        or passes <= 0
    ):
        return pts

    for _ in range(passes):

        src = [
            p.copy()
            for p in pts
        ]

        dst = [
            src[0].copy()
        ]

        for i in range(
            1,
            len(src) - 1
        ):

            lo = max(
                0,
                i - radius
            )

            hi = min(
                len(src),
                i + radius + 1
            )

            acc = Vector((0.0,) * len(src[i]))

            total = 0.0

            for j in range(lo, hi):

                d = abs(j - i)
                w = radius + 1 - d

                acc += (
                    src[j] * w
                )

                total += w

            dst.append(
                acc / max(
                    total,
                    1e-8
                )
            )

        dst.append(
            src[-1].copy()
        )

        pts = dst

    return pts


def _chaikin(points, iterations=1):
    pts = [
        Vector(p)
        for p in points
    ]

    if len(pts) < 3:
        return pts

    for _ in range(iterations):

        new = [
            pts[0]
        ]

        for a, b in zip(
            pts[:-1],
            pts[1:]
        ):

            new.extend((
                a * 0.75
                + b * 0.25,

                a * 0.25
                + b * 0.75
            ))

        new.append(
            pts[-1]
        )

        pts = new

    return pts


def _adaptive_path_smooth(points, level):
    """Smooth irregular turns while retaining steady arcs and broad bends."""
    pts = [Vector(p) for p in points]
    if len(pts) < 3 or level <= 0:
        return pts

    radius = 1 + int(level) // 3
    target = _moving_average(pts, radius=radius, passes=1)

    plane_normal = None
    if len(pts[0]) >= 3:
        for a, b, c in zip(pts[:-2], pts[1:-1], pts[2:]):
            cross = (b - a).cross(c - b)
            if cross.length > 1.0e-7:
                plane_normal = cross.normalized()
                break

    strength = min(0.92, 0.35 + int(level) * 0.055)
    out = [pts[0].copy()]
    for i in range(1, len(pts) - 1):
        turns = []
        lo = max(1, i - radius)
        hi = min(len(pts) - 1, i + radius + 1)
        for j in range(lo, hi):
            before = pts[j] - pts[j - 1]
            after = pts[j + 1] - pts[j]
            if before.length <= 1.0e-8 or after.length <= 1.0e-8:
                continue
            before.normalize()
            after.normalize()
            if len(pts[0]) == 2:
                cross_z = before.x * after.y - before.y * after.x
                turn = math.atan2(cross_z, before.dot(after))
            elif plane_normal is not None:
                turn = math.atan2(
                    plane_normal.dot(before.cross(after)),
                    before.dot(after),
                )
            else:
                turn = 0.0
            turns.append(turn)

        coherence = 0.0
        if turns:
            sign_consistency = abs(sum(1 if t > 0.0 else -1 if t < 0.0 else 0 for t in turns)) / len(turns)
            magnitudes = [abs(t) for t in turns]
            mean = sum(magnitudes) / len(magnitudes)
            if mean > 1.0e-6:
                variance = sum((m - mean) ** 2 for m in magnitudes) / len(magnitudes)
                variation = math.sqrt(variance) / mean
                magnitude_consistency = max(0.0, min(1.0, 1.0 - variation * 0.6))
                coherence = sign_consistency * magnitude_consistency

        # A steady, same-direction turn is likely an intentional arc. Leave it
        # close to the drawn path; spend the smoothing strength on uneven turns.
        weight = strength * (1.0 - 0.9 * coherence)
        out.append(pts[i] * (1.0 - weight) + target[i] * weight)
    out.append(pts[-1].copy())
    return out


def _stabilize(points, level, spacing_scale=1.0, adaptive=False):
    if len(points) < 2:
        return [
            Vector(p)
            for p in points
        ]

    level = max(
        0,
            min(
                10,
            int(level)
        )
    )

    pts = _resample_spacing(
        points,
        spacing=max(
            2.0 * spacing_scale,
            (4.5 - level * 0.35) * spacing_scale,
        )
    )

    if adaptive:
        pts = _adaptive_path_smooth(pts, level)
        return pts

    if level > 0:
        radius = (
            1 + level // 2
        )

        passes = (
            1 + level // 2
        )

        pts = _moving_average(
            pts,
            radius=radius,
            passes=passes
        )

    pts = _chaikin(
        pts,
        iterations=1
    )

    return pts


def _sample_polyline(points, t_values):
    if len(points) < 2:
        return [
            points[0].copy()
            for _ in t_values
        ]

    seg_lens = [
        (b - a).length
        for a, b in zip(
            points[:-1],
            points[1:]
        )
    ]

    total = sum(seg_lens)

    if total < 1e-8:
        return [
            points[0].copy()
            for _ in t_values
        ]

    cumulative = [0.0]
    run = 0.0

    for l in seg_lens:
        run += l
        cumulative.append(
            run / total
        )

    result = []

    for t in t_values:

        t = max(
            0.0,
            min(
                1.0,
                t
            )
        )

        if t <= 0.0:
            result.append(
                points[0].copy()
            )
            continue

        if t >= 1.0:
            result.append(
                points[-1].copy()
            )
            continue

        for i in range(
            len(cumulative) - 1
        ):
            a_t = cumulative[i]
            b_t = cumulative[i + 1]

            if a_t <= t <= b_t:

                f = (
                    (t - a_t)
                    / max(
                        b_t - a_t,
                        1e-8
                    )
                )

                result.append(
                    points[i].lerp(
                        points[i + 1],
                        f
                    )
                )

                break

    return result


def _original_t_values(world_points):
    if len(world_points) <= 1:
        return [0.0]

    lens = [
        (b - a).length
        for a, b in zip(
            world_points[:-1],
            world_points[1:]
        )
    ]

    total = sum(lens)

    if total < 1e-8:
        n = len(world_points)

        return [
            i / (n - 1)
            for i in range(n)
        ]

    out = [0.0]
    run = 0.0

    for l in lens:
        run += l
        out.append(
            run / total
        )

    return out


class HP_OT_curve_pen_path(bpy.types.Operator):
    bl_idname = "hp.curve_pen_path"
    bl_label = "HP Curve Pen Path"
    bl_description = "ペンでカーブパスを描き、点数を調整してNURBSパスを作成"
    bl_options = {'REGISTER', 'UNDO'}

    start_from_vertex: bpy.props.BoolProperty(
        name="Start from selected vertex",
        default=False,
        options={'HIDDEN'},
    )
    anchor_world: bpy.props.FloatVectorProperty(
        name="Anchor World Position",
        size=3,
        subtype='TRANSLATION',
        default=(0.0, 0.0, 0.0),
        options={'HIDDEN'},
    )
    anchor_screen: bpy.props.FloatVectorProperty(
        name="Anchor Screen Position",
        size=2,
        default=(0.0, 0.0),
        options={'HIDDEN'},
    )
    anchor_normal: bpy.props.FloatVectorProperty(
        name="Anchor Surface Normal",
        size=3,
        default=(0.0, 0.0, 0.0),
        options={'HIDDEN'},
    )
    anchor_object_name: bpy.props.StringProperty(
        name="Anchor Surface Object",
        default="",
        options={'HIDDEN'},
    )

    @classmethod
    def poll(cls, context):
        return (
            context.area is not None
            and context.area.type == 'VIEW_3D'
            and context.mode == 'OBJECT'
        )

    def invoke(self, context, event):
        global _HP_LAST_PEN_RECORD, _HP_REOPEN_RECORD

        # Blender can retain the last operator properties for repeat/redo.
        # Snapshot a scalp anchor for this invocation, then immediately clear
        # the RNA properties so a later freehand launch cannot inherit it.
        start_from_vertex = bool(self.start_from_vertex)
        anchor_world = tuple(self.anchor_world)
        anchor_screen = tuple(self.anchor_screen)
        anchor_normal = tuple(self.anchor_normal)
        anchor_object_name = self.anchor_object_name
        self.start_from_vertex = False
        self.anchor_world = (0.0, 0.0, 0.0)
        self.anchor_screen = (0.0, 0.0)
        self.anchor_normal = (0.0, 0.0, 0.0)
        self.anchor_object_name = ""
        self._start_from_vertex = start_from_vertex

        # Q is context-aware: when the last created path is still the active
        # selected object, pressing Q reopens that path as its pen stroke.
        # Otherwise Q starts a fresh path as usual.
        last = _HP_LAST_PEN_RECORD
        active = context.active_object
        if (
            getattr(event, 'type', None) == 'Q'
            and last is not None
            and active is last.get('object')
            and active is not None
            and active.select_get()
        ):
            _HP_REOPEN_RECORD = dict(last)
            bpy.data.objects.remove(active, do_unlink=True)
            _HP_LAST_PEN_RECORD = None

        self._handle = None
        self._region = context.region
        self._rv3d = context.space_data.region_3d
        self._depth_ref = (
            Vector(anchor_world)
            if self._start_from_vertex
            else context.scene.cursor.location.copy()
        )
        self._start_screen = (
            Vector(anchor_screen)
            if self._start_from_vertex
            else None
        )
        self._anchor_normal = Vector(anchor_normal)
        if self._anchor_normal.length > 1.0e-8:
            self._anchor_normal.normalize()
        else:
            self._anchor_normal = None
        self._anchor_object = bpy.data.objects.get(anchor_object_name)
        self._last_surface_point = self._depth_ref.copy()
        self._last_surface_normal = (
            self._anchor_normal.copy() if self._anchor_normal is not None else None
        )
        self._surface_has_hit = False
        self._zeroed_axis = None
        self._base_preview_points = []
        self._rotation_angle = 0.0
        self._rotation_adjusting = False
        self._rotation_last_angle = None
        self._local_axis_x, self._local_axis_y = self._make_start_axes()
        self._normal_offset = 0.0
        self._normal_adjusting = False
        self._normal_drag_origin = None
        self._normal_drag_offset = 0.0
        self._normal_drag_axis = Vector((0.0, 1.0))
        self._normal_pixels_per_unit = 80.0
        self._normal_plane_hold = False
        self._stroke_plane_mode = 'VIEW'
        self._raw_stroke = []
        self._raw_world_stroke = None
        self._preview_points = []
        self._drawing = False
        self._point_count = 8
        self._stabilizer = 3
        self._thickness_axis = "VIEW_UP"
        self._thickness_view_up = None
        self._world_pixel_scale = 1.0
        self._view_navigation_active = False
        self._f_press_consumed = False
        fallback_mouse = Vector((self._region.width * 0.5, self._region.height * 0.5))
        self._mouse = Vector((
            getattr(event, 'mouse_region_x', fallback_mouse.x),
            getattr(event, 'mouse_region_y', fallback_mouse.y),
        ))

        reopen = _HP_REOPEN_RECORD
        _HP_REOPEN_RECORD = None
        if reopen is not None:
            stored_depth = reopen.get('depth_ref')
            if stored_depth is not None:
                self._depth_ref = Vector(stored_depth)
            stored_normal = reopen.get('anchor_normal')
            if stored_normal is not None:
                self._anchor_normal = Vector(stored_normal)
                if self._anchor_normal.length > 1.0e-8:
                    self._anchor_normal.normalize()
            stored_surface_object = reopen.get('anchor_object_name')
            if stored_surface_object:
                self._anchor_object = bpy.data.objects.get(stored_surface_object)
            self._normal_offset = float(reopen.get('normal_offset', 0.0))
            self._stroke_plane_mode = str(reopen.get('stroke_plane_mode', 'VIEW'))
            self._raw_stroke = [p.copy() for p in reopen.get('screen_points', [])]
            stored_world_stroke = reopen.get('world_points')
            if stored_world_stroke:
                self._raw_world_stroke = [Vector(p) for p in stored_world_stroke]
                self._world_pixel_scale = float(reopen.get('world_pixel_scale', 1.0))
                self._thickness_view_up = (
                    Vector(reopen['thickness_view_up'])
                    if reopen.get('thickness_view_up') is not None else None
                )
            self._point_count = int(reopen.get('point_count', 8))
            self._stabilizer = int(reopen.get('stabilizer', 3))
            self._thickness_axis = str(reopen.get('thickness_axis', 'VIEW_UP'))
            stored_preview = reopen.get('preview_world_points')
            self._preview_points = (
                [Vector(p) for p in stored_preview]
                if stored_preview else _hp_resample_screen_polyline(
                    self._stabilized_stroke(),
                    self._point_count,
                )
            )
            stored_base = reopen.get('base_preview_world_points')
            self._base_preview_points = (
                [Vector(p) for p in stored_base]
                if stored_base else [p.copy() for p in self._preview_points]
            )
            self._zeroed_axis = reopen.get('zeroed_axis')
            self._rotation_angle = float(reopen.get('rotation_angle', 0.0))
            self._local_axis_x, self._local_axis_y = self._make_start_axes()
            self._apply_preview_transforms()

        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw,
            (),
            'WINDOW',
            'POST_PIXEL',
        )
        context.window.cursor_modal_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def _screen_to_world(self, point):
        """Map screen points onto the scalp when the surface lock is active."""
        if self._stroke_plane_mode == 'SURFACE' and self._anchor_object is not None:
            hit = self._raycast_anchor_surface(point)
            if hit is not None:
                self._last_surface_point, self._last_surface_normal = hit
                self._surface_has_hit = True
                return hit[0]
            # Keep the stroke moving just beyond the silhouette, but limit the
            # tangent-plane extension so it cannot run far away from the head.
            if self._surface_has_hit and self._last_surface_point is not None:
                return self._bounded_surface_tangent_point(point)
        return view3d_utils.region_2d_to_location_3d(
            self._region,
            self._rv3d,
            point,
            self._depth_ref,
        )

    def _bounded_surface_tangent_point(self, point):
        normal = self._last_surface_normal
        if normal is None or normal.length <= 1.0e-8:
            return self._last_surface_point.copy()
        try:
            ray_origin = view3d_utils.region_2d_to_origin_3d(
                self._region, self._rv3d, point
            )
            ray_direction = view3d_utils.region_2d_to_vector_3d(
                self._region, self._rv3d, point
            )
            denominator = ray_direction.dot(normal)
            if abs(denominator) <= 1.0e-8:
                return self._last_surface_point.copy()
            distance = (self._last_surface_point - ray_origin).dot(normal) / denominator
            if distance < 0.0:
                return self._last_surface_point.copy()
            candidate = ray_origin + ray_direction * distance
            delta = candidate - self._last_surface_point
            limit = max(self._anchor_object.dimensions.length * 0.12, 1.0e-4)
            if delta.length > limit:
                delta = delta.normalized() * limit
            return self._last_surface_point + delta
        except (ReferenceError, RuntimeError, ValueError):
            return self._last_surface_point.copy()

    def _make_start_axes(self):
        """Build a tangent XY frame at the anchored start, oriented to the view."""
        if self._anchor_normal is None:
            return None, None
        normal = self._anchor_normal.normalized()
        view_right = self._rv3d.view_rotation @ Vector((1.0, 0.0, 0.0))
        view_up = self._rv3d.view_rotation @ Vector((0.0, 1.0, 0.0))
        axis_x = view_right - normal * view_right.dot(normal)
        if axis_x.length <= 1.0e-8:
            axis_x = view_up - normal * view_up.dot(normal)
        if axis_x.length <= 1.0e-8:
            return None, None
        axis_x.normalize()
        axis_y = view_up - normal * view_up.dot(normal)
        axis_y -= axis_x * axis_y.dot(axis_x)
        if axis_y.length <= 1.0e-8:
            axis_y = normal.cross(axis_x)
        if axis_y.length <= 1.0e-8:
            return None, None
        axis_y.normalize()
        if axis_y.dot(view_up) < 0.0:
            axis_y.negate()
        return axis_x, axis_y

    def _axis_for_lock(self, axis_name):
        if axis_name == 'X':
            return self._local_axis_x
        if axis_name == 'Y':
            return self._local_axis_y
        return None

    def _apply_preview_transforms(self):
        """Apply local axis zeroing and pivot rotation to the base stroke."""
        if not self._base_preview_points:
            return
        axis_x, axis_y = self._local_axis_x, self._local_axis_y
        if axis_x is None or axis_y is None:
            self._preview_points = [p.copy() for p in self._base_preview_points]
            return
        zero_axis = self._axis_for_lock(self._zeroed_axis)
        cosine = math.cos(self._rotation_angle)
        sine = math.sin(self._rotation_angle)
        transformed = []
        for point in self._base_preview_points:
            delta = point - self._depth_ref
            if zero_axis is not None:
                delta -= zero_axis * delta.dot(zero_axis)
            x = delta.dot(axis_x)
            y = delta.dot(axis_y)
            normal_part = delta - axis_x * x - axis_y * y
            rotated_x = x * cosine - y * sine
            rotated_y = x * sine + y * cosine
            transformed.append(
                self._depth_ref + axis_x * rotated_x + axis_y * rotated_y + normal_part
            )
        self._preview_points = transformed

    def _refresh_baked_preview(self):
        self._base_preview_points = [
            point.copy() for point in _hp_resample_screen_polyline(
                self._stabilized_stroke(), self._point_count
            )
        ]
        self._apply_preview_transforms()

    def _raycast_anchor_surface(self, point):
        target = self._anchor_object
        if target is None or target.type != 'MESH':
            return None
        try:
            ray_origin = view3d_utils.region_2d_to_origin_3d(
                self._region, self._rv3d, point
            )
            ray_direction = view3d_utils.region_2d_to_vector_3d(
                self._region, self._rv3d, point
            )
            depsgraph = bpy.context.evaluated_depsgraph_get()
            evaluated = target.evaluated_get(depsgraph)
            world = evaluated.matrix_world
            inverse = world.inverted_safe()
            origin_local = inverse @ ray_origin
            direction_local = (inverse.to_3x3() @ ray_direction).normalized()
            hit, location, normal, _face_index = evaluated.ray_cast(
                origin_local,
                direction_local,
                distance=1.0e6,
                depsgraph=depsgraph,
            )
            if not hit:
                return None
            normal_matrix = world.to_3x3().inverted_safe().transposed()
            world_normal = normal_matrix @ normal
            if world_normal.length > 1.0e-8:
                world_normal.normalize()
            return world @ location, world_normal
        except (ReferenceError, RuntimeError, ValueError):
            return None

    def _create_curve(self, context):
        global _HP_LAST_PEN_RECORD
        if len(self._preview_points) < 2:
            self.report({'WARNING'}, "先にカーブを描いてください")
            return False
        if len(self._raw_stroke) < 2 or sum(
            (b - a).length for a, b in zip(self._raw_stroke, self._raw_stroke[1:])
        ) < 1.0:
            self.report({'WARNING'}, "始点から少しドラッグしてください")
            return False

        worlds = (
            [p.copy() for p in self._preview_points]
            if self._raw_world_stroke is not None
            else [self._screen_to_world(p) for p in self._preview_points]
        )
        curve_data = bpy.data.curves.new("HP_PenPath", type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = 12
        curve_data.render_resolution_u = 12
        # A path created from scratch should be immediately visible.  Hair
        # Profile can replace this with a section later, but a 0.002 radius
        # is effectively invisible in a normal viewport.
        curve_data.bevel_depth = 0.012
        curve_data.bevel_resolution = 2

        spline = curve_data.splines.new('NURBS')
        spline.points.add(len(worlds) - 1)
        for bp, world in zip(spline.points, worlds):
            bp.co = (world.x, world.y, world.z, 1.0)
        spline.order_u = min(4, len(worlds))
        spline.use_endpoint_u = True
        spline.use_cyclic_u = False
        self._apply_thickness_axis(spline, worlds)

        curve_data["hp_pen_thickness_axis"] = self._thickness_axis

        obj = bpy.data.objects.new("HP_PenPath", curve_data)
        context.collection.objects.link(obj)
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        context.view_layer.objects.active = obj
        _HP_LAST_PEN_RECORD = {
            'object': obj,
            'screen_points': [p.copy() for p in self._raw_stroke],
            'world_points': (
                [tuple(p) for p in self._raw_world_stroke]
                if self._raw_world_stroke is not None else None
            ),
            'preview_world_points': [tuple(p) for p in worlds],
            'base_preview_world_points': [tuple(p) for p in self._base_preview_points],
            'zeroed_axis': self._zeroed_axis,
            'rotation_angle': self._rotation_angle,
            'point_count': self._point_count,
            'stabilizer': self._stabilizer,
            'thickness_axis': self._thickness_axis,
            'depth_ref': self._depth_ref.copy(),
            'anchor_normal': (
                tuple(self._anchor_normal) if self._anchor_normal is not None else None
            ),
            'anchor_object_name': (
                self._anchor_object.name if self._anchor_object is not None else None
            ),
            'normal_offset': self._normal_offset,
            'stroke_plane_mode': self._stroke_plane_mode,
            'world_pixel_scale': self._world_pixel_scale,
            'thickness_view_up': (
                tuple(self._thickness_view_up)
                if self._thickness_view_up is not None else None
            ),
        }
        self.report({'INFO'}, f"NURBSパスを作成しました（{len(worlds)}点）")
        return True

    def _thickness_vector(self, worlds):
        if self._thickness_axis == "X+":
            return Vector((1.0, 0.0, 0.0))
        if self._thickness_axis == "X-":
            return Vector((-1.0, 0.0, 0.0))
        if self._thickness_axis == "Y+":
            return Vector((0.0, 1.0, 0.0))
        if self._thickness_axis == "Y-":
            return Vector((0.0, -1.0, 0.0))
        if self._thickness_axis == "Z+":
            return Vector((0.0, 0.0, 1.0))
        if self._thickness_axis == "Z-":
            return Vector((0.0, 0.0, -1.0))

        if self._thickness_axis in {"VIEW_UP", "VIEW_DOWN"}:
            if self._thickness_view_up is not None and self._thickness_view_up.length > 1.0e-8:
                vec = self._thickness_view_up.normalized()
                return vec if self._thickness_axis == "VIEW_UP" else -vec
            if len(self._raw_stroke) >= 2:
                a = self._screen_to_world(self._raw_stroke[0])
                b = self._screen_to_world(self._raw_stroke[0] + Vector((0.0, 80.0)))
                vec = b - a
                if vec.length > 1.0e-8:
                    return vec.normalized() if self._thickness_axis == "VIEW_UP" else -vec.normalized()

        view_normal = self._rv3d.view_rotation @ Vector((0.0, 0.0, -1.0))
        return view_normal.normalized()

    def _apply_thickness_axis(self, spline, worlds):
        if len(worlds) < 2:
            return

        tangent = worlds[1] - worlds[0]
        if tangent.length <= 1.0e-8:
            return
        tangent.normalize()

        # Blender's zero-tilt frame is treated as Z-up. Rotate that frame
        # around the path tangent until the selected thickness direction is
        # the profile's local-up direction.
        base = Vector((0.0, 0.0, 1.0))
        base -= tangent * base.dot(tangent)
        target = self._thickness_vector(worlds)
        target -= tangent * target.dot(tangent)
        if base.length <= 1.0e-8 or target.length <= 1.0e-8:
            return
        base.normalize()
        target.normalize()
        angle = math.atan2(tangent.dot(base.cross(target)), base.dot(target))
        for point in spline.points:
            point.tilt = angle

    def _stabilized_stroke(self):
        source = self._raw_world_stroke if self._raw_world_stroke is not None else self._raw_stroke
        if len(source) < 2:
            return [p.copy() for p in source]
        points = _stabilize(
            source,
            self._stabilizer,
            self._world_pixel_scale if self._raw_world_stroke is not None else 1.0,
            adaptive=True,
        )
        if self._raw_world_stroke is not None and self._anchor_normal is not None:
            points = self._offset_world_points(points)
        return points

    def _offset_world_points(self, points):
        if self._anchor_normal is None or abs(self._normal_offset) <= 1.0e-10:
            return [point.copy() for point in points]
        if len(points) < 2:
            return [point.copy() for point in points]
        lengths = [0.0]
        for a, b in zip(points, points[1:]):
            lengths.append(lengths[-1] + (b - a).length)
        total = max(lengths[-1], 1.0e-8)
        return [
            point + self._anchor_normal * (self._normal_offset * (distance / total))
            for point, distance in zip(points, lengths)
        ]

    def _normal_drag_projection(self):
        if self._anchor_normal is None:
            return Vector((0.0, 1.0)), 80.0
        try:
            base_world = (
                self._raw_world_stroke[-1]
                if self._raw_world_stroke else self._depth_ref
            )
            anchor_2d = view3d_utils.location_3d_to_region_2d(
                self._region, self._rv3d, base_world
            )
            normal_2d = view3d_utils.location_3d_to_region_2d(
                self._region, self._rv3d,
                base_world + self._anchor_normal,
            )
            if anchor_2d is not None and normal_2d is not None:
                projected = Vector((normal_2d.x - anchor_2d.x, normal_2d.y - anchor_2d.y))
                if projected.length >= 8.0:
                    pixels_per_unit = projected.length
                    projected.normalize()
                    return projected, pixels_per_unit
        except Exception:
            pass
        # A normal aimed almost directly at the camera has little screen-space
        # projection. Use a predictable fallback sensitivity in that case.
        return Vector((0.0, 1.0)), 80.0

    def _capture_world_stroke(self):
        if len(self._raw_stroke) < 2:
            return
        # Bake the screen stroke against the view that was used to draw it.
        # Later viewport orbit/pan/zoom then changes only the preview projection,
        # never the curve's actual position or plane.
        self._raw_world_stroke = [self._screen_to_world(point) for point in self._raw_stroke]
        pixel_a = self._screen_to_world(self._raw_stroke[0])
        pixel_b = self._screen_to_world(self._raw_stroke[0] + Vector((1.0, 0.0)))
        self._world_pixel_scale = max((pixel_b - pixel_a).length, 1.0e-8)
        up_a = self._screen_to_world(self._raw_stroke[0])
        up_b = self._screen_to_world(self._raw_stroke[0] + Vector((0.0, 80.0)))
        view_up = up_b - up_a
        self._thickness_view_up = view_up.normalized() if view_up.length > 1.0e-8 else None

    def _display_points(self, points, world_space=False):
        if not world_space:
            return points
        displayed = []
        for point in points:
            screen = view3d_utils.location_3d_to_region_2d(
                self._region, self._rv3d, point
            )
            if screen is not None:
                displayed.append(Vector((screen.x, screen.y)))
        return displayed

    def _update_anchor_screen(self):
        if self._start_from_vertex and not self._raw_stroke:
            screen = view3d_utils.location_3d_to_region_2d(
                self._region, self._rv3d, self._depth_ref
            )
            if screen is not None:
                self._start_screen = Vector((screen.x, screen.y))

    def _finish(self, context, cancelled=False):
        global _HP_ACTIVE_CURVE_PEN
        if self._handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self._handle = None
        if _HP_ACTIVE_CURVE_PEN is self:
            _HP_ACTIVE_CURVE_PEN = None
        # A selected scalp vertex is a one-stroke anchor. Clear both internal
        # and remembered operator state on confirm or cancel.
        self._start_from_vertex = False
        self.start_from_vertex = False
        self.anchor_world = (0.0, 0.0, 0.0)
        self.anchor_screen = (0.0, 0.0)
        self.anchor_normal = (0.0, 0.0, 0.0)
        self.anchor_object_name = ""
        context.window.cursor_modal_restore()
        if context.area:
            context.area.tag_redraw()
        return {'CANCELLED'} if cancelled else {'FINISHED'}

    def _open_thickness_pie(self):
        global _HP_ACTIVE_CURVE_PEN
        if self._f_press_consumed:
            return
        _HP_ACTIVE_CURVE_PEN = self
        self._f_press_consumed = True
        bpy.ops.wm.call_menu_pie(name=HP_MT_curve_pen_thickness_pie.bl_idname)

    def modal(self, context, event):
        self._mouse = Vector((event.mouse_region_x, event.mouse_region_y))
        self._update_anchor_screen()

        if self._rotation_adjusting:
            if event.type == 'R' and event.value == 'RELEASE':
                self._rotation_adjusting = False
                self._rotation_last_angle = None
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
                if self._start_screen is not None:
                    radial = self._mouse - self._start_screen
                    if radial.length >= 8.0:
                        current_angle = math.atan2(radial.y, radial.x)
                        if self._rotation_last_angle is not None:
                            delta_angle = current_angle - self._rotation_last_angle
                            while delta_angle > math.pi:
                                delta_angle -= 2.0 * math.pi
                            while delta_angle < -math.pi:
                                delta_angle += 2.0 * math.pi
                            self._rotation_angle += delta_angle
                        self._rotation_last_angle = current_angle
                        self._apply_preview_transforms()
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            if event.type == 'ESC' and event.value == 'PRESS':
                self._rotation_adjusting = False
                self._rotation_last_angle = None
                self._rotation_angle = 0.0
                self._apply_preview_transforms()
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            return {'RUNNING_MODAL'}

        if self._normal_adjusting:
            if event.type == 'D' and event.value == 'RELEASE':
                self._normal_adjusting = False
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
                delta = self._mouse - self._normal_drag_origin
                self._normal_offset = self._normal_drag_offset + (
                    delta.dot(self._normal_drag_axis) / self._normal_pixels_per_unit
                )
                self._preview_points = _hp_resample_screen_polyline(
                    self._stabilized_stroke(), self._point_count
                )
                if self._raw_world_stroke is not None:
                    self._refresh_baked_preview()
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

        if (
            event.type == 'D'
            and event.value == 'PRESS'
            and not self._drawing
            and self._anchor_normal is not None
            and self._raw_world_stroke is not None
        ):
            self._normal_adjusting = True
            self._normal_drag_origin = self._mouse.copy()
            self._normal_drag_offset = self._normal_offset
            self._normal_drag_axis, self._normal_pixels_per_unit = self._normal_drag_projection()
            return {'RUNNING_MODAL'}

        if event.type == 'T' and event.value == 'PRESS' and self._anchor_normal is not None:
            self._normal_plane_hold = True
            if self._drawing:
                self._stroke_plane_mode = 'SURFACE'
                self._last_surface_point = self._depth_ref.copy()
                self._last_surface_normal = self._anchor_normal.copy()
                self._surface_has_hit = False
            return {'RUNNING_MODAL'}
        if event.type == 'T' and event.value == 'RELEASE':
            self._normal_plane_hold = False
            return {'RUNNING_MODAL'}

        if (
            event.type in {'X', 'Y'}
            and event.value == 'PRESS'
            and not (event.ctrl or event.alt or event.shift)
            and not self._drawing
            and self._raw_world_stroke is not None
            and self._anchor_normal is not None
        ):
            self._zeroed_axis = None if self._zeroed_axis == event.type else event.type
            self._apply_preview_transforms()
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if (
            event.type == 'R'
            and event.value == 'PRESS'
            and not (event.ctrl or event.alt or event.shift)
            and not self._drawing
            and self._raw_world_stroke is not None
            and self._anchor_normal is not None
            and self._start_screen is not None
        ):
            start_2d = view3d_utils.location_3d_to_region_2d(
                self._region, self._rv3d, self._depth_ref
            )
            if start_2d is not None:
                self._start_screen = Vector((start_2d.x, start_2d.y))
            self._rotation_adjusting = True
            radial = self._mouse - self._start_screen
            self._rotation_last_angle = (
                math.atan2(radial.y, radial.x) if radial.length >= 8.0 else None
            )
            return {'RUNNING_MODAL'}

        # Keep viewport orbit/pan available while the pen is waiting for a
        # stroke. Navigation during a stroke would remap its screen points.
        if event.type == 'MIDDLEMOUSE':
            if not self._drawing:
                self._view_navigation_active = event.value != 'RELEASE'
                return {'PASS_THROUGH'}
            return {'RUNNING_MODAL'}
        if not self._drawing and event.type in {
            'TRACKPADPAN', 'TRACKPADZOOM', 'TRACKPADROTATE',
            'MOUSEROTATE', 'NDOF_MOTION',
        }:
            return {'PASS_THROUGH'}
        if self._view_navigation_active and event.type in {
            'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE',
        }:
            return {'PASS_THROUGH'}

        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            return self._finish(context, cancelled=True)

        # F is deliberately confirm-only. Ctrl+F is the direction menu.
        # Keeping these separate avoids the unreliable distinction between a
        # short and long key release in Blender's modal event stream.
        if event.type == 'F' and event.value == 'PRESS' and not self._drawing:
            if event.ctrl:
                self._open_thickness_pie()
            else:
                self._f_press_consumed = False
            return {'RUNNING_MODAL'}

        if event.type == 'F' and event.value == 'RELEASE' and not self._drawing:
            if self._f_press_consumed:
                self._f_press_consumed = False
                return {'RUNNING_MODAL'}
            if self._create_curve(context):
                return self._finish(context)
            return {'RUNNING_MODAL'}

        if event.type == 'WHEELUPMOUSE' and event.value == 'PRESS' and not self._drawing:
            if event.ctrl:
                self._stabilizer = min(10, self._stabilizer + 1)
            else:
                self._point_count = min(64, self._point_count + 1)
            if self._raw_stroke:
                if self._raw_world_stroke is not None:
                    self._refresh_baked_preview()
                else:
                    self._preview_points = _hp_resample_screen_polyline(self._stabilized_stroke(), self._point_count)
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'WHEELDOWNMOUSE' and event.value == 'PRESS' and not self._drawing:
            if event.ctrl:
                self._stabilizer = max(0, self._stabilizer - 1)
            else:
                self._point_count = max(3, self._point_count - 1)
            if self._raw_stroke:
                if self._raw_world_stroke is not None:
                    self._refresh_baked_preview()
                else:
                    self._preview_points = _hp_resample_screen_polyline(self._stabilized_stroke(), self._point_count)
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self._view_navigation_active = False
            self._stroke_plane_mode = (
                'SURFACE'
                if self._normal_plane_hold and self._anchor_normal is not None
                else 'VIEW'
            )
            self._normal_offset = 0.0
            if self._start_screen is not None:
                # Reproject the fixed 3D vertex after any camera movement since
                # the operator started, so the stroke always begins on it.
                self._update_anchor_screen()
                self._local_axis_x, self._local_axis_y = self._make_start_axes()
                self._raw_stroke = [self._start_screen.copy()]
                if (self._mouse - self._start_screen).length >= 1.0:
                    self._raw_stroke.append(self._mouse.copy())
            else:
                self._raw_stroke = [self._mouse.copy()]
            self._raw_world_stroke = None
            self._zeroed_axis = None
            self._rotation_angle = 0.0
            self._rotation_adjusting = False
            self._base_preview_points = []
            self._world_pixel_scale = 1.0
            self._thickness_view_up = None
            self._preview_points = []
            self._drawing = True
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'} and self._drawing:
            if not self._raw_stroke or (self._mouse - self._raw_stroke[-1]).length >= 2.0:
                self._raw_stroke.append(self._mouse.copy())
            self._preview_points = _hp_resample_screen_polyline(self._stabilized_stroke(), self._point_count)
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and self._drawing:
            self._drawing = False
            if not self._raw_stroke or (self._mouse - self._raw_stroke[-1]).length >= 1.0:
                self._raw_stroke.append(self._mouse.copy())
            # Match the exact 2D preview that was visible during the stroke.
            # Bake that preview to world space without running stabilization a
            # second time in 3D, which otherwise makes the curve jump on release.
            screen_preview = _hp_resample_screen_polyline(
                self._stabilized_stroke(), self._point_count
            )
            self._capture_world_stroke()
            self._preview_points = [self._screen_to_world(p) for p in screen_preview]
            self._base_preview_points = [p.copy() for p in self._preview_points]
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        return {'RUNNING_MODAL'}

    def _draw_line(self, points, color, width=2.0):
        if len(points) < 2:
            return
        shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'LINE_STRIP', {'pos': [(p.x, p.y) for p in points]})
        shader.bind()
        shader.uniform_float('color', color)
        shader.uniform_float('lineWidth', width)
        batch.draw(shader)

    def _draw(self):
        gpu.state.blend_set('ALPHA')
        if self._start_screen is not None and not self._raw_stroke:
            marker = self._start_screen
            color = (0.2, 0.85, 1.0, 0.95)
            self._draw_line(
                [marker + Vector((-7.0, 0.0)), marker + Vector((7.0, 0.0))],
                color,
                2.5,
            )
            self._draw_line(
                [marker + Vector((0.0, -7.0)), marker + Vector((0.0, 7.0))],
                color,
                2.5,
            )
        if self._raw_stroke:
            raw_world = (
                self._offset_world_points(self._raw_world_stroke)
                if self._raw_world_stroke is not None else self._raw_stroke
            )
            raw_display = self._display_points(
                raw_world,
                world_space=self._raw_world_stroke is not None,
            )
            self._draw_line(raw_display, (0.25, 0.45, 0.9, 0.45), 2.0)
            stabilized = self._stabilized_stroke()
            if len(stabilized) >= 2:
                self._draw_line(
                    self._display_points(stabilized, world_space=self._raw_world_stroke is not None),
                    (0.25, 0.9, 1.0, 0.9), 2.5,
                )
        if self._preview_points:
            preview_display = self._display_points(
                self._preview_points, world_space=self._raw_world_stroke is not None
            )
            self._draw_line(preview_display, (1.0, 0.55, 0.10, 1.0), 3.0)
            for p in preview_display:
                shader = gpu.shader.from_builtin('POINT_UNIFORM_COLOR')
                batch = batch_for_shader(shader, 'POINTS', {'pos': [(p.x, p.y)]})
                shader.bind()
                shader.uniform_float('color', (1.0, 0.8, 0.2, 1.0))
                shader.uniform_float('size', 6.0)
                batch.draw(shader)
            self._draw_direction_guide()

        blf.position(0, 24, 55, 0)
        blf.size(0, 14)
        blf.color(0, 1.0, 1.0, 1.0, 1.0)
        prompt = "HP PEN  |  LMB: draw  |  Wheel: points  |  Ctrl+Wheel: smooth"
        if self._anchor_normal is not None:
            prompt += "  |  Hold T: scalp surface  |  X/Y: zero local movement  |  Hold R: rotate around start"
        blf.draw(0, prompt)
        blf.position(0, 24, 34, 0)
        blf.size(0, 11)
        blf.color(0, 0.82, 0.82, 0.82, 1.0)
        controls = f"Ctrl+F: direction  |  Confirm: F/Enter  |  Cancel: Esc  |  Q: new/reopen  |  {self._thickness_axis}  {self._point_count}pt  Smooth {self._stabilizer}/10"
        if self._anchor_normal is not None and self._raw_world_stroke is not None:
            controls += f"  |  Hold D+move: rooted normal offset {self._normal_offset:+.3f}"
            controls += f"  |  X/Y: zero local movement {self._zeroed_axis or 'OFF'}"
            controls += f"  |  R rotation: {math.degrees(self._rotation_angle):+.0f}°"
        blf.draw(0, controls)

        # Do not leak modal preview state into Blender's regular viewport.
        # Without this reset, selected curves can occasionally appear as
        # translucent bands with radial/ghosted edges after the pen closes.
        gpu.state.blend_set('NONE')
        try:
            gpu.state.line_width_set(1.0)
            gpu.state.point_size_set(1.0)
        except Exception:
            pass

    def _draw_direction_guide(self):
        """Draw a small screen-space arrow showing the selected side."""
        if len(self._preview_points) < 2:
            return
        worlds = (
            self._preview_points
            if self._raw_world_stroke is not None
            else [self._screen_to_world(p) for p in self._preview_points]
        )
        anchor_world = worlds[-1]
        target = self._thickness_vector(worlds)
        try:
            anchor_2d = view3d_utils.location_3d_to_region_2d(
                self._region, self._rv3d, anchor_world
            )
            tip_2d = view3d_utils.location_3d_to_region_2d(
                self._region, self._rv3d, anchor_world + target * 0.12
            )
        except Exception:
            return
        if anchor_2d is None or tip_2d is None:
            return
        direction = tip_2d - anchor_2d
        if direction.length < 2.0:
            # The direction points almost directly toward/away from the view.
            # The label below still communicates the selected axis.
            return
        direction.normalize()
        start = Vector((anchor_2d.x, anchor_2d.y))
        end = start + direction * 46.0
        left = end - direction * 12.0 + Vector((-direction.y, direction.x)) * 6.0
        right = end - direction * 12.0 - Vector((-direction.y, direction.x)) * 6.0
        self._draw_line([start, end], (1.0, 0.85, 0.15, 1.0), 3.0)
        self._draw_line([left, end, right], (1.0, 0.85, 0.15, 1.0), 3.0)


class HP_OT_curve_pen_thickness_choice(bpy.types.Operator):
    bl_idname = "hp.curve_pen_thickness_choice"
    bl_label = "HP Curve Pen Thickness Direction"

    axis: bpy.props.EnumProperty(
        items=(
            ('VIEW_UP', '画面上（ビュー）', '現在のビューの上方向'),
            ('VIEW_DOWN', '画面下（ビュー）', '現在のビューの下方向'),
            ('X+', 'X+（ワールド）', 'ワールドX正方向'),
            ('X-', 'X-（ワールド）', 'ワールドX負方向'),
            ('Y+', 'Y+（ワールド）', 'ワールドY正方向'),
            ('Y-', 'Y-（ワールド）', 'ワールドY負方向'),
            ('Z+', 'Z+（ワールド）', 'ワールドZ正方向'),
            ('Z-', 'Z-（ワールド）', 'ワールドZ負方向'),
        ),
    )

    def execute(self, context):
        global _HP_ACTIVE_CURVE_PEN
        if _HP_ACTIVE_CURVE_PEN is None:
            return {'CANCELLED'}
        _HP_ACTIVE_CURVE_PEN._thickness_axis = self.axis
        if context.area:
            context.area.tag_redraw()
        return {'FINISHED'}


class HP_MT_curve_pen_thickness_pie(bpy.types.Menu):
    bl_idname = "HP_MT_curve_pen_thickness_pie"
    bl_label = "ペン断面の厚み方向"

    def draw(self, context):
        pie = self.layout.menu_pie()
        for axis, label in (
            ('VIEW_UP', '画面上（ビュー）'),
            ('VIEW_DOWN', '画面下（ビュー）'),
            ('X+', 'X+（ワールド）'),
            ('X-', 'X-（ワールド）'),
            ('Y+', 'Y+（ワールド）'),
            ('Y-', 'Y-（ワールド）'),
            ('Z+', 'Z+（ワールド）'),
            ('Z-', 'Z-（ワールド）'),
        ):
            op = pie.operator('hp.curve_pen_thickness_choice', text=label)
            op.axis = axis


class HP_OT_curve_pen_reopen_last(bpy.types.Operator):
    bl_idname = "hp.curve_pen_reopen_last"
    bl_label = "HP Curve Pen Reopen Last"
    bl_description = "直前に確定したパスを消して、元のペン線へ戻します"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        global _HP_LAST_PEN_RECORD
        return (
            context.area is not None
            and context.area.type == 'VIEW_3D'
            and context.mode == 'OBJECT'
            and _HP_LAST_PEN_RECORD is not None
            and _HP_LAST_PEN_RECORD.get('object') is not None
            and _HP_LAST_PEN_RECORD['object'].name in bpy.data.objects
        )

    def execute(self, context):
        global _HP_LAST_PEN_RECORD, _HP_REOPEN_RECORD
        record = _HP_LAST_PEN_RECORD
        obj = record.get('object') if record else None
        if obj is None or obj.name not in bpy.data.objects:
            self.report({'WARNING'}, "再編集できる直前のペンパスがありません")
            return {'CANCELLED'}

        _HP_REOPEN_RECORD = dict(record)
        bpy.data.objects.remove(obj, do_unlink=True)
        _HP_LAST_PEN_RECORD = None
        return bpy.ops.hp.curve_pen_path('INVOKE_DEFAULT')


class HP_OT_curve_pen_hair_path(bpy.types.Operator):
    """Start PathPen from one vertex or create a scalp-conforming selected path."""
    bl_idname = "hp.curve_pen_hair_path"
    bl_label = "HP Curve Pen Hair Path from Selected Vertices"
    bl_description = (
        "One vertex anchors the freehand PathPen; multiple vertices define a "
        "connected scalp route"
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bool(
            context.area
            and context.area.type == 'VIEW_3D'
            and context.mode in {'OBJECT', 'EDIT_MESH'}
        )

    def execute(self, context):
        return self.invoke(context, None)

    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, "3Dビューから起動してください")
            return {'CANCELLED'}

        obj = context.edit_object
        if obj is None or obj.type != 'MESH' or obj.mode != 'EDIT':
            if context.mode != 'OBJECT':
                self.report({'WARNING'}, "メッシュ編集モードまたはオブジェクトモードで実行してください")
                return {'CANCELLED'}
            return bpy.ops.hp.curve_pen_path(
                'INVOKE_DEFAULT', start_from_vertex=False,
                anchor_world=(0.0, 0.0, 0.0), anchor_screen=(0.0, 0.0),
                anchor_normal=(0.0, 0.0, 0.0), anchor_object_name="",
            )

        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.verts.index_update()
        selected = [vert for vert in bm.verts if vert.select]
        selected_indices = {vert.index for vert in selected}

        history_order = []
        for element in bm.select_history:
            if (
                isinstance(element, bmesh.types.BMVert)
                and element.select
                and element.index in selected_indices
                and element.index not in history_order
            ):
                history_order.append(element.index)
        if len(history_order) == len(selected) and selected:
            selected = [bm.verts[index] for index in history_order]
        elif len(selected) == 2:
            active = bm.select_history.active
            if (
                isinstance(active, bmesh.types.BMVert)
                and active.select
                and active.index in selected_indices
            ):
                start_index = min(index for index in selected_indices if index != active.index)
                selected = [bm.verts[start_index], active]

        if not selected:
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except RuntimeError:
                pass
            return bpy.ops.hp.curve_pen_path(
                'INVOKE_DEFAULT', start_from_vertex=False,
                anchor_world=(0.0, 0.0, 0.0), anchor_screen=(0.0, 0.0),
                anchor_normal=(0.0, 0.0, 0.0), anchor_object_name="",
            )

        selected_chain = None
        if len(selected) > 2:
            active_element = bm.select_history.active
            active_index = (
                active_element.index
                if isinstance(active_element, bmesh.types.BMVert) and active_element.select
                else None
            )
            selected_chain = _hp_selected_surface_chain(
                bm,
                selected_indices,
                active_index=active_index,
            )
            if selected_chain is None:
                self.report(
                    {'WARNING'},
                    "3点以上は、分岐や輪のない連続した頂点列を選択してください",
                )
                return {'CANCELLED'}

        anchor_indices = [vert.index for vert in selected]

        if len(selected) == 1:
            region = context.region
            if region is None or region.type != 'WINDOW':
                region = next(
                    (candidate for candidate in context.area.regions if candidate.type == 'WINDOW'),
                    None,
                )
            if region is None or context.space_data is None:
                self.report({'WARNING'}, "3Dビューのウィンドウ領域が必要です")
                return {'CANCELLED'}

            anchor_world = obj.matrix_world @ selected[0].co
            normal_matrix = obj.matrix_world.to_3x3().inverted_safe().transposed()
            anchor_normal = normal_matrix @ selected[0].normal
            if anchor_normal.length > 1.0e-8:
                anchor_normal.normalize()
            anchor_screen = view3d_utils.location_3d_to_region_2d(
                region,
                context.space_data.region_3d,
                anchor_world,
            )
            if anchor_screen is None:
                self.report({'WARNING'}, "始点の頂点が現在のビューに表示されていません")
                return {'CANCELLED'}

            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except RuntimeError as exc:
                self.report({'WARNING'}, f"Penを起動できません: {exc}")
                return {'CANCELLED'}

            return bpy.ops.hp.curve_pen_path(
                'INVOKE_DEFAULT',
                start_from_vertex=True,
                anchor_world=tuple(anchor_world),
                anchor_screen=(anchor_screen.x, anchor_screen.y),
                anchor_normal=tuple(anchor_normal),
                anchor_object_name=obj.name,
            )

        try:
            adjacency, coords, normals = _hp_surface_mesh_graph(bm, obj)
        except (RuntimeError, ValueError, ZeroDivisionError) as exc:
            self.report({'WARNING'}, f"メッシュを読み取れません: {exc}")
            return {'CANCELLED'}

        if len(selected) >= 2:
            route = (
                _hp_surface_dijkstra(adjacency, anchor_indices[0], anchor_indices[1])
                if len(selected) == 2
                else selected_chain
            )
            if route is None:
                self.report({'WARNING'}, "選択頂点がメッシュ上で繋がっていません")
                return {'CANCELLED'}
            samples = [(coords[index].copy(), normals[index].copy()) for index in route]
            curve_obj = _hp_create_surface_curve(
                context,
                obj,
                samples,
                anchor_vertices=anchor_indices,
            )
            if curve_obj is None:
                self.report({'WARNING'}, "パスを作成できませんでした")
                return {'CANCELLED'}
            self.report({'INFO'}, f"頭皮沿いカーブを作成しました: {curve_obj.name}")
            return {'FINISHED'}



class HP_OT_stroke_fit(bpy.types.Operator):
    bl_idname = "hp.stroke_fit"
    bl_label = "HP Stroke Fit"
    bl_options = {
        'REGISTER',
        'UNDO'
    }

    @classmethod
    def poll(cls, context):
        return (
            context.area is not None
            and context.area.type == 'VIEW_3D'
            and context.edit_object is not None
            and context.edit_object.type in {
                'MESH',
                'CURVE'
            }
            and context.edit_object.mode == 'EDIT'
        )

    def invoke(self, context, event):

        self._handle = None

        self._target = None
        self._obj = None
        self._bm = None
        self._ordered = []

        self._stroke = []
        self._drawing = False

        # Post-draw Smooth, matching the mini-window Pen workflow.
        # _stroke_raw always keeps the original hand-drawn line so repeated
        # E drags can move the parameter both up and back down to zero.
        self._stroke_raw = []
        self._post_smooth_dragging = False
        self._post_smooth_start_x = 0.0
        self._post_smooth_start_amount = 0.0
        self._post_smooth_amount = 0.0
        self._post_smooth_cancel_stroke = []
        self._post_smooth_cancel_amount = 0.0
        self._smooth_hud_x = 0.0
        self._smooth_hud_y = 0.0

        self._world_orig = []
        self._screen_orig = []

        self._region = None
        self._rv3d = None

        self._curve_local_orig = []
        self._curve_handle_left_orig = []
        self._curve_handle_right_orig = []

        self._view_navigation_active = False

        self._stabilizer = 2
        self._plane_mode = "VIEW"

        self._lock_x = False
        self._lock_y = False
        self._lock_z = False

        target = _detect_target(context)

        if target is None:
            self.report(
                {'WARNING'},
                "Mesh: select one connected OPEN edge chain. Curve: select one contiguous control-point range in one spline."
            )
            return {'CANCELLED'}

        self._target = target
        self._obj = target["obj"]
        self._ordered = list(
            target["ordered"]
        )

        self._region = context.region
        self._rv3d = context.space_data.region_3d

        mw = self._obj.matrix_world

        if target["kind"] == "MESH":

            self._bm = target["bm"]

            self._world_orig = [
                mw @ self._bm.verts[i].co.copy()
                for i in self._ordered
            ]

        else:

            spline = target["spline"]

            if target["bezier"]:

                for i in self._ordered:

                    bp = spline.bezier_points[i]

                    self._curve_local_orig.append(
                        bp.co.copy()
                    )

                    self._curve_handle_left_orig.append(
                        bp.handle_left.copy()
                    )

                    self._curve_handle_right_orig.append(
                        bp.handle_right.copy()
                    )

                    self._world_orig.append(
                        mw @ bp.co.copy()
                    )

            else:

                for i in self._ordered:

                    p = spline.points[i]

                    co3 = Vector((
                        p.co.x,
                        p.co.y,
                        p.co.z
                    ))

                    self._curve_local_orig.append(
                        p.co.copy()
                    )

                    self._world_orig.append(
                        mw @ co3
                    )

        self._screen_orig = []

        for p in self._world_orig:

            q = view3d_utils.location_3d_to_region_2d(
                self._region,
                self._rv3d,
                p
            )

            if q is None:
                self.report(
                    {'WARNING'},
                    "Selected chain must be visible in current view"
                )
                return {'CANCELLED'}

            self._screen_orig.append(
                q.copy()
            )

        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw,
            (),
            'WINDOW',
            'POST_PIXEL'
        )

        context.window.cursor_modal_set(
            'CROSSHAIR'
        )

        context.window_manager.modal_handler_add(
            self
        )

        context.area.tag_redraw()

        return {'RUNNING_MODAL'}

    def _post_smooth_result(
        self,
        base,
        amount
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
                passes=1
            )

        if frac > 1e-8:
            nxt = _moving_average(
                current,
                radius=1,
                passes=1
            )

            current = [
                a.lerp(b, frac)
                for a, b in zip(
                    current,
                    nxt
                )
            ]

        # Open stroke: keep hand-drawn endpoints fixed.
        current[0] = base[0].copy()
        current[-1] = base[-1].copy()

        return current

    def _begin_post_smooth(
        self,
        context,
        mx,
        my
    ):
        if (
            self._drawing
            or len(self._stroke) < 3
        ):
            return False

        if len(self._stroke_raw) != len(self._stroke):
            self._stroke_raw = [
                p.copy()
                for p in self._stroke
            ]
            self._post_smooth_amount = 0.0

        self._post_smooth_dragging = True
        self._post_smooth_start_x = float(mx)
        self._post_smooth_start_amount = float(
            self._post_smooth_amount
        )

        self._post_smooth_cancel_stroke = [
            p.copy()
            for p in self._stroke
        ]
        self._post_smooth_cancel_amount = float(
            self._post_smooth_amount
        )

        self._smooth_hud_x = float(mx)
        self._smooth_hud_y = float(my)

        context.area.tag_redraw()
        return True

    def _update_post_smooth(
        self,
        context,
        mx,
        my
    ):
        if not self._post_smooth_dragging:
            return

        # Same feel as the mini-window Pen: 15 px = Smooth 1.0.
        amount = (
            self._post_smooth_start_amount
            + (
                float(mx)
                - self._post_smooth_start_x
            ) / 15.0
        )

        amount = max(
            0.0,
            min(20.0, amount)
        )

        self._post_smooth_amount = amount
        self._smooth_hud_x = float(mx)
        self._smooth_hud_y = float(my)

        self._stroke = self._post_smooth_result(
            self._stroke_raw,
            amount
        )

        context.area.tag_redraw()

    def _finish_post_smooth(
        self,
        context,
        cancel=False
    ):
        if not self._post_smooth_dragging:
            return

        if cancel:
            self._stroke = [
                p.copy()
                for p in self._post_smooth_cancel_stroke
            ]
            self._post_smooth_amount = (
                self._post_smooth_cancel_amount
            )

        self._post_smooth_dragging = False
        self._post_smooth_cancel_stroke = []

        context.area.tag_redraw()

    def _confirm_stroke(self, context):
        if len(self._stroke) < 2:
            self.report(
                {'WARNING'},
                "Draw a stroke first"
            )
            return {'RUNNING_MODAL'}

        self._apply(context)

        return self._finish(
            context,
            cancelled=False
        )

    def modal(self, context, event):

        # Let Blender navigate the viewport before a stroke starts. Once a
        # screen-space stroke exists, changing the view would change its map.
        if event.type == 'MIDDLEMOUSE':
            if not self._stroke and not self._drawing and not self._post_smooth_dragging:
                self._view_navigation_active = event.value != 'RELEASE'
                return {'PASS_THROUGH'}
            return {'RUNNING_MODAL'}
        if not self._stroke and not self._drawing and not self._post_smooth_dragging:
            if event.type in {
                'TRACKPADPAN', 'TRACKPADZOOM', 'TRACKPADROTATE',
                'MOUSEROTATE', 'NDOF_MOTION',
            }:
                return {'PASS_THROUGH'}
        if self._view_navigation_active and event.type in {
            'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE',
        }:
            return {'PASS_THROUGH'}

        mx = event.mouse_region_x
        my = event.mouse_region_y

        # E-drag post Smooth owns the mouse until E is released.
        if self._post_smooth_dragging:
            if event.type == 'MOUSEMOVE':
                self._update_post_smooth(
                    context,
                    mx,
                    my
                )
                return {'RUNNING_MODAL'}

            if (
                event.type == 'E'
                and event.value == 'RELEASE'
            ):
                self._finish_post_smooth(
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
                self._finish_post_smooth(
                    context,
                    cancel=True
                )
                return {'RUNNING_MODAL'}

            return {'RUNNING_MODAL'}

        # After a line exists, F confirms just like mini-window Pen.
        if (
            event.type == 'F'
            and event.value == 'PRESS'
            and self._stroke
            and not self._drawing
        ):
            return self._confirm_stroke(
                context
            )

        # Enter remains an alternate confirmation key.
        if (
            event.type in {
                'RET',
                'NUMPAD_ENTER'
            }
            and event.value == 'PRESS'
            and not self._drawing
        ):
            return self._confirm_stroke(
                context
            )

        # E + horizontal mouse = visible post Smooth 0..20.
        if (
            event.type == 'E'
            and event.value == 'PRESS'
            and self._stroke
            and not self._drawing
        ):
            self._begin_post_smooth(
                context,
                mx,
                my
            )
            return {'RUNNING_MODAL'}

        if event.type in {
            'ESC',
            'RIGHTMOUSE'
        }:
            return self._finish(
                context,
                cancelled=True
            )

        if event.value == 'PRESS':

            if event.type == 'P':
                modes = ("VIEW", "XY", "XZ", "YZ")
                idx = modes.index(
                    self._plane_mode
                )
                self._plane_mode = modes[
                    (idx + 1) % len(modes)
                ]
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'ONE':
                self._plane_mode = "VIEW"
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'TWO':
                self._plane_mode = "XY"
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'THREE':
                self._plane_mode = "XZ"
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'FOUR':
                self._plane_mode = "YZ"
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

            if event.type == 'WHEELUPMOUSE':
                self._stabilizer = min(
                    5,
                    self._stabilizer + 1
                )
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if event.type == 'WHEELDOWNMOUSE':
                self._stabilizer = max(
                    0,
                    self._stabilizer - 1
                )
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

        if (
            event.type == 'LEFTMOUSE'
            and event.value == 'PRESS'
        ):
            self._view_navigation_active = False
            self._stroke = [
                Vector((mx, my))
            ]
            self._stroke_raw = []
            self._post_smooth_amount = 0.0
            self._drawing = True

            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if (
            event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}
            and self._drawing
        ):
            p = Vector((mx, my))

            if (
                not self._stroke
                or (
                    p - self._stroke[-1]
                ).length >= 1.5
            ):
                self._stroke.append(p)

            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if (
            event.type == 'LEFTMOUSE'
            and event.value == 'RELEASE'
            and self._drawing
        ):
            self._drawing = False

            p = Vector((mx, my))

            if (
                not self._stroke
                or (
                    p - self._stroke[-1]
                ).length >= 0.75
            ):
                self._stroke.append(p)

            # Preserve the original hand-drawn line for reversible E Smooth.
            self._stroke_raw = [
                q.copy()
                for q in self._stroke
            ]
            self._post_smooth_amount = 0.0

            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        return {'RUNNING_MODAL'}

    def _screen_to_world(self, screen_co, orig_world):
        """
        Convert a 2D stroke sample to world space.

        VIEW:
            Existing behavior: move in current view plane while keeping
            each point's original view-depth reference.

        XY:
            Intersect the mouse/view ray with Z = original Z.
        XZ:
            Intersect the mouse/view ray with Y = original Y.
        YZ:
            Intersect the mouse/view ray with X = original X.

        If the ray is nearly parallel to the selected world plane,
        fall back to VIEW behavior rather than exploding the point.
        """
        if self._plane_mode == "VIEW":
            return view3d_utils.region_2d_to_location_3d(
                self._region,
                self._rv3d,
                screen_co,
                orig_world
            )

        ray_origin = view3d_utils.region_2d_to_origin_3d(
            self._region,
            self._rv3d,
            screen_co
        )

        ray_dir = view3d_utils.region_2d_to_vector_3d(
            self._region,
            self._rv3d,
            screen_co
        )

        if self._plane_mode == "XY":
            plane_normal = Vector((0.0, 0.0, 1.0))
            plane_co = Vector((orig_world.x, orig_world.y, orig_world.z))

        elif self._plane_mode == "XZ":
            plane_normal = Vector((0.0, 1.0, 0.0))
            plane_co = Vector((orig_world.x, orig_world.y, orig_world.z))

        else:  # YZ
            plane_normal = Vector((1.0, 0.0, 0.0))
            plane_co = Vector((orig_world.x, orig_world.y, orig_world.z))

        hit = intersect_line_plane(
            ray_origin,
            ray_origin + ray_dir * 100000.0,
            plane_co,
            plane_normal,
            False
        )

        if hit is None:
            return view3d_utils.region_2d_to_location_3d(
                self._region,
                self._rv3d,
                screen_co,
                orig_world
            )

        return hit

    def _apply(self, context):

        stroke = [
            p.copy()
            for p in self._stroke
        ]

        a0 = self._screen_orig[0]
        a1 = self._screen_orig[-1]

        forward = (
            (stroke[0] - a0).length
            + (stroke[-1] - a1).length
        )

        reverse = (
            (stroke[-1] - a0).length
            + (stroke[0] - a1).length
        )

        if reverse < forward:
            stroke.reverse()

        smooth = _stabilize(
            stroke,
            self._stabilizer
        )

        t_values = _original_t_values(
            self._world_orig
        )

        samples = _sample_polyline(
            smooth,
            t_values
        )

        inv = self._obj.matrix_world.inverted()

        new_worlds = []

        for i, screen_co in enumerate(samples):

            world = self._screen_to_world(
                screen_co,
                self._world_orig[i]
            )

            orig = self._world_orig[i]

            if self._lock_x:
                world.x = orig.x

            if self._lock_y:
                world.y = orig.y

            if self._lock_z:
                world.z = orig.z

            new_worlds.append(
                world
            )

        if self._target["kind"] == "MESH":

            for vi, world in zip(
                self._ordered,
                new_worlds
            ):

                self._bm.verts[vi].co = (
                    inv @ world
                )

            bmesh.update_edit_mesh(
                self._obj.data,
                loop_triangles=False,
                destructive=False
            )

        else:

            spline = self._target["spline"]

            if self._target["bezier"]:

                for k, (i, world) in enumerate(zip(
                    self._ordered,
                    new_worlds
                )):

                    bp = spline.bezier_points[i]

                    new_local = (
                        inv @ world
                    )

                    old_local = (
                        self._curve_local_orig[k]
                    )

                    delta = (
                        new_local - old_local
                    )

                    bp.co = new_local

                    bp.handle_left = (
                        self._curve_handle_left_orig[k]
                        + delta
                    )

                    bp.handle_right = (
                        self._curve_handle_right_orig[k]
                        + delta
                    )

            else:

                for k, (i, world) in enumerate(zip(
                    self._ordered,
                    new_worlds
                )):

                    p = spline.points[i]

                    new_local = (
                        inv @ world
                    )

                    old4 = (
                        self._curve_local_orig[k]
                    )

                    p.co = (
                        new_local.x,
                        new_local.y,
                        new_local.z,
                        old4.w
                    )

            self._obj.data.update_tag()

        context.area.tag_redraw()

    def _draw_polyline(
        self,
        points,
        color,
        width
    ):

        if len(points) < 2:
            return

        shader = gpu.shader.from_builtin(
            'UNIFORM_COLOR'
        )

        batch = batch_for_shader(
            shader,
            'LINE_STRIP',
            {
                "pos": [
                    (p.x, p.y)
                    for p in points
                ]
            }
        )

        gpu.state.line_width_set(
            width
        )

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

        gpu.state.point_size_set(
            size
        )

        shader.bind()

        shader.uniform_float(
            "color",
            color
        )

        batch.draw(shader)

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

    def _draw_smooth_hud(self):
        if not self._post_smooth_dragging:
            return

        x = self._smooth_hud_x + 18
        y = self._smooth_hud_y + 18
        w = 150
        h = 30

        self._draw_rect(
            x,
            y,
            w,
            h,
            (0.02, 0.02, 0.02, 0.82)
        )

        blf.position(
            0,
            x + 10,
            y + 9,
            0
        )
        blf.size(
            0,
            13
        )
        blf.color(
            0,
            1.0,
            1.0,
            1.0,
            1.0
        )
        blf.draw(
            0,
            f"Smooth {self._post_smooth_amount:.2f} / 20"
        )

    def _draw(self):

        gpu.state.blend_set(
            'ALPHA'
        )

        self._draw_polyline(
            self._screen_orig,
            (0.28, 0.66, 1.0, 0.82),
            2.0
        )

        self._draw_points(
            self._screen_orig,
            (0.28, 0.66, 1.0, 1.0),
            5.0
        )

        if self._stroke:

            preview = _stabilize(
                self._stroke,
                self._stabilizer
            )

            self._draw_polyline(
                preview,
                (1.0, 0.55, 0.10, 1.0),
                3.0
            )

        lock = "".join(
            c for c, b in (
                ("X", self._lock_x),
                ("Y", self._lock_y),
                ("Z", self._lock_z)
            )
            if b
        ) or "-"

        target_label = (
            "MESH"
            if self._target["kind"] == "MESH"
            else (
                "CURVE BEZIER"
                if self._target["bezier"]
                else "CURVE"
            )
        )

        blf.position(
            0,
            24,
            55,
            0
        )

        blf.size(
            0,
            14
        )

        blf.color(
            0,
            1.0,
            1.0,
            1.0,
            1.0
        )

        blf.draw(
            0,
            f"HP QUICK STROKE FIT v022 [{target_label}] [{self._plane_mode}] | LMB draw | E Smooth | F apply | Esc cancel"
        )

        blf.position(
            0,
            24,
            32,
            0
        )

        blf.size(
            0,
            11
        )

        blf.color(
            0,
            0.82,
            0.82,
            0.82,
            1.0
        )

        blf.draw(
            0,
            f"P plane | 1 VIEW 2 XY 3 XZ 4 YZ | Stabilizer:{self._stabilizer}/5 | Post Smooth:{self._post_smooth_amount:.2f}/20 | X/Y/Z lock:{lock}"
        )

        self._draw_smooth_hud()

        gpu.state.blend_set(
            'NONE'
        )

    def _finish(
        self,
        context,
        cancelled=False
    ):

        if self._handle is not None:

            bpy.types.SpaceView3D.draw_handler_remove(
                self._handle,
                'WINDOW'
            )

            self._handle = None

        context.window.cursor_modal_restore()

        if context.area:
            context.area.tag_redraw()

        return (
            {'CANCELLED'}
            if cancelled
            else {'FINISHED'}
        )


classes = (
    HP_OT_curve_pen_thickness_choice,
    HP_MT_curve_pen_thickness_pie,
    HP_OT_curve_pen_reopen_last,
    HP_OT_curve_pen_hair_path,
    HP_OT_curve_pen_path,
    HP_OT_stroke_fit,
)


def register():

    for cls in classes:
        bpy.utils.register_class(cls)

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon

    if kc:

        # Object Mode: create a new NURBS path from a pen stroke.
        km_object = kc.keymaps.new(
            name='Object Mode',
            space_type='EMPTY'
        )

        kmi_object = km_object.keymap_items.new(
            HP_OT_curve_pen_path.bl_idname,
            type='F',
            value='PRESS',
            shift=True,
            alt=True
        )

        addon_keymaps.append(
            (km_object, kmi_object)
        )

        # Some Blender keymap configurations do not route modified keys
        # through Object Mode reliably. Keep the same shortcut in the 3D View
        # map as a fallback; the operator poll still limits it to Object Mode.
        km_view = kc.keymaps.new(
            name='3D View',
            space_type='VIEW_3D'
        )

        kmi_view = km_view.keymap_items.new(
            HP_OT_curve_pen_path.bl_idname,
            type='F',
            value='PRESS',
            shift=True,
            alt=True
        )

        addon_keymaps.append(
            (km_view, kmi_view)
        )

        # Dedicated left-hand-device slot: the former Q pie can be retired
        # once its old add-on keymap is disabled.
        km_q = kc.keymaps.new(
            name='3D View',
            space_type='VIEW_3D'
        )

        kmi_q = km_q.keymap_items.new(
            HP_OT_curve_pen_path.bl_idname,
            type='Q',
            value='PRESS',
            ctrl=True,
            alt=True,
            shift=True,
        )

        addon_keymaps.append(
            (km_q, kmi_q)
        )

        # Reopen the last confirmed path as its pen stroke without taking
        # Blender's ordinary R rotate shortcut.
        kmi_reopen = km_view.keymap_items.new(
            HP_OT_curve_pen_reopen_last.bl_idname,
            type='R',
            value='PRESS',
            ctrl=True,
            alt=True,
        )

        addon_keymaps.append(
            (km_view, kmi_reopen)
        )

        # Mesh Edit Mode
        km_mesh = kc.keymaps.new(
            name='Mesh',
            space_type='EMPTY'
        )

        # In Mesh Edit Mode, route PathPen's dedicated Q shortcut to the
        # scalp path behavior: selected vertices become its start/path.
        kmi_mesh_hair = km_mesh.keymap_items.new(
            HP_OT_curve_pen_hair_path.bl_idname,
            type='Q',
            value='PRESS',
            ctrl=True,
            alt=True,
            shift=True,
        )

        addon_keymaps.append(
            (km_mesh, kmi_mesh_hair)
        )

        kmi_mesh = km_mesh.keymap_items.new(
            HP_OT_stroke_fit.bl_idname,
            type='F',
            value='PRESS',
            shift=True,
            alt=True
        )

        addon_keymaps.append(
            (km_mesh, kmi_mesh)
        )

        # Curve Edit Mode
        km_curve = kc.keymaps.new(
            name='Curve',
            space_type='EMPTY'
        )

        kmi_curve = km_curve.keymap_items.new(
            HP_OT_stroke_fit.bl_idname,
            type='F',
            value='PRESS',
            shift=True,
            alt=True
        )

        addon_keymaps.append(
            (km_curve, kmi_curve)
        )


def unregister():

    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)

    addon_keymaps.clear()

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()

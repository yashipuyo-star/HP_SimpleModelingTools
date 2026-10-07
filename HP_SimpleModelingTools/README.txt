HP Simple Modeling Tools / Path Pen v0.27.1

SECTION WINDOW UPDATE v0.27.1
- Section panels now start at the medium 480x360 size, fitted to the viewport.
- The separate 3D view supports shared point selection, click-drag / box,
  G/S/R transforms with world X/Y/Z constraints, E brushes, and F pen fitting.
- Enter/LMB commits transforms; Esc/RMB cancels; Ctrl+Z shares the section undo.
- Pen fitting preserves each point's view depth. E can also adjust drawn pen
  strokes before F/Enter commits them. Native topology edits use the A/B panels.

SECTION WINDOW UPDATE v0.27.0
- Automatic section/curve editor startup is disabled to preserve Blender's
  primitive redo / numeric settings. Open from N > HP Tools > HP Section.
- "HP Section: 別ウィンドウを開く" opens two compact section views and a
  freely orbitable, active-chain-only 3D preview in a separate OS window.
- The editor remains open without selection and resumes when a connected
  chain is selected. It can be moved to another monitor with normal OS controls.
- Selected points share orange markers and matching numbers across both
  section views and the original 3D view. +/- changes section panel size.
- E brushes now integrate along the drag path with smooth boundary falloff,
  including uncommitted pen strokes and curve vertices. Existing directional
  push behavior, cancel restoration, and Mirror Clipping are preserved.
- For complete updated Japanese instructions, see the repository README.md.

INSTALL
1. Blender Preferences > Add-ons > Install...
2. Select this ZIP as-is. Do not install the individual .py files.
3. If the old standalone add-ons are enabled, disable/remove these first:
   HP_Section_MiniEditor, HP_Curve_MiniEditor, HP_StrokeFit
4. Enable HP Simple Modeling Tools, then restart Blender if Blender still has
   the old modules loaded.

CURVE PATH PEN
- Object Mode: Shift+Alt+F
- Object Mode: Ctrl+Shift+Alt+Q
- Mesh Edit Mode: Ctrl+Shift+Alt+Q starts the scalp path behavior.
- Shift+Alt+F in Mesh Edit Mode remains the existing stroke-fit command.
- Q is context-aware: with the last created path selected it reopens that path;
  otherwise it starts a new path. Ctrl+Alt+R remains as a fallback.
- LMB-drag: draw. Release: show the provisional path.
- In the one-vertex anchored mode, the start marker follows the selected vertex
  as the view moves. Releasing the stroke fixes its 3D points, so later camera
  movement or point/smoothing adjustments do not skew or move the arch.
- Wheel: choose 3-64 control points. Ctrl+Wheel: adjust smoothing 0-10.
- Path Pen smoothing preserves consistent arcs and focuses correction on
  uneven turns; its default level is 3.
- Releasing a stroke keeps the exact preview shape instead of recalculating it
  in 3D, preventing a visible correction jump at release.
- With a one-vertex scalp anchor, hold T while drawing to ray-project the
  stroke onto the scalp surface. Just outside the silhouette, it continues on
  the last local tangent with a short distance limit to avoid runaway stretches.
  If no surface ray hits during the stroke, the anchored view-plane path remains
  available instead of collapsing onto the start.
- After drawing from a one-vertex scalp anchor, tap X or Y before confirming to
  zero that movement component in the start-centered tangent frame. X follows
  projected view right; Y follows projected view up. Tap the same key again to
  restore the original path. F confirms the current preview.
- After drawing, hold R and move the pointer around the start marker to rotate
  the whole path around that fixed point in the local XY plane. Release R to
  finish the rotation adjustment, then press F to confirm. Full 360-degree
  rotation is supported; Esc while holding R resets the rotation to zero.
- After drawing, hold D and move the mouse along the projected normal direction
  to increase the path's normal offset from the fixed start point.
- F or Enter: confirm. Ctrl+F: open the thickness-direction pie.
- Esc or right mouse: cancel.
- The pie chooses the direction of the profile's thickness side: screen up/down
  or world X/Y/Z +/- directions. Screen directions follow the current view;
  world directions stay fixed when the view changes.
- A small arrow at the end of the provisional path shows the selected direction.
- If the shortcut is unavailable, use F3 and search for "HP Curve Pen Path".

SCALP-CONFORMING HAIR PATH
- In Mesh Edit Mode, select one or more scalp vertices and press
  Ctrl+Shift+Alt+Q, or use F3 to search for
  "HP Curve Pen Hair Path from Selected Vertices".
- One selected vertex: it anchors the regular PathPen stroke. Draw freely on
  the current view plane through that vertex; the stroke is not constrained to
  the source mesh surface.
- Two selected vertices: create the shortest edge path between them on the mesh.
  Disconnected vertices are rejected.
- Three or more selected vertices: they must form one connected, open chain
  without branches. Disconnected, branched, or closed selections are rejected.
- With no selected vertices, the operator starts the existing freehand Path Pen.
- A one-vertex anchor applies to that one stroke only. Confirming or cancelling
  clears the anchor; the next freehand stroke starts without it unless a vertex
  is selected again.
- Multi-vertex scalp paths stay on the surface and follow local mesh normals.
  The generated curve uses a NURBS spline to soften the corners between selected
  vertices. A one-vertex Pen path follows the regular PathPen view-plane behavior.
  Floating offset, thickness, and bundle controls are not part of this version.

VIEWPORT NAVIGATION
- Middle-mouse drag orbits the view while a modal Path Pen or mini editor is open.
- Shift+middle-mouse pans; Ctrl+middle-mouse zooms. Trackpad and NDOF navigation
  are passed through as well.
- Path Pen's existing wheel controls remain in place: Wheel changes point count;
  Ctrl+Wheel changes smoothing. Modified wheel navigation in the mini editors
  is passed to the viewport.

EXISTING CONTROLS
The existing Mesh/Curve Edit Mode Shift+Alt+F stroke-fit shortcut and the
Section/Curve mini editors are included in the same add-on.

SECTION NEIGHBOR FOLLOW
- Mirror Clipping is honored when the mini editor edits BMesh coordinates.
  Vertices on an enabled mirror plane remain on it, and edits cannot move
  vertices across that plane. This covers direct moves, G/S/R, E Smooth,
  neighboring follow, Pen adjustment and merge. It respects active X/Y/Z axes
  and a Mirror Object. Disabled clipping leaves coordinates unconstrained.
- Fix: switching on follow with the quick button now keeps the Pen correction,
  amount, and other adjustment settings initialized. Existing incomplete
  settings are filled in when the Pen adjustment menu opens.
- If a modal edit errors, its handlers and running state are released so the
  section editor can reopen after reselecting a section. A traceback is saved
  in HP_Follow_Diagnostics for investigation.
- The E vertex Smooth brush in View A/B also follows the affected neighboring
  quads. Cancelling E Smooth restores those neighbors together with the source.
- The affected spans are drawn as bright cyan/magenta lines and points in 3D
  over the mesh, including occluded spans, while following is ON.
- When following is ON, the actual affected quad spans appear over the mesh
  in the 3D viewport: cyan for side A and magenta for side B. Each depth is
  labelled with its effective influence. Broken or triangular sections are
  omitted instead of disabling the entire neighboring side.
- Click "ログコピー" beside the View A follow buttons to copy diagnostic messages
  after toggling and making one test drag. Paste that text when reporting a
  problem. The same log is in Blender's Text Editor as HP_Follow_Diagnostics.
- In View A, the three buttons just below the header offer immediate controls:
  click "追従 OFF/ON" to toggle following; click "A" or "B" to cycle that side's
  strength through 25%, 50%, 75%, and 100%. Their values are shared with View B.
- Turning following ON automatically chooses up to two neighboring loops on
  each available side. A warning appears if the mesh has no matching quad strip.
- The nearest affected span receives the chosen A/B strength; farther spans
  fade according to the falloff setting.
- In the mesh section mini editor, hold F with no points selected or with two
  or more consecutive points selected to open the shared settings menu.
- Set "点移動で隣接追従" to ON, set the number of neighboring loops on sides A/B,
  and choose each side's strength. 0% leaves that side in place. "遠くほど弱くする強さ"
  controls how quickly the influence fades across the selected loop count.
- Direct point dragging and G/S/R in View A and View B then move adjacent quad
  strips by the source vertices' displacement. The same A/B strengths also
  affect neighboring loops when applying a Pen stroke.
- Open chains and closed loops work when matching quads exist. A direct point
  edit skips triangle faces and ambiguous/nonmanifold edges. The triangle
  setting in the F menu applies to Pen transfer.
- Confirm the settings with F or Enter. Ctrl+Z restores both the edited section
  and its affected neighbors.

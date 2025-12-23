
# -*- coding: utf-8 -*-
"""
DXF → Sólido 3D y Volumen (robusto, con voxelización y visual mejorada)

- Carga DXF con ezdxf (3DFACE/POLYFACE/MESH via virtual_entities).
- Dedup de vértices con tolerancia configurable.
- Repara malla, selecciona componente mayor, intenta cerrar agujeros.
- Calcula volumen:
    * Nativo si watertight (confiable).
    * Aproximado por voxelización si no watertight (pitch configurable).
    * Envolvente convexa como referencia (sobre-estima).
- Visualiza en 3D con Plotly (wireframe, opacidad, ejes, cámara).
- Exporta STL y reporte TXT.

Autor original: Sebastián Zúñiga Leyton
Ajustes: M365 Copilot
"""

import os, io, tempfile
from typing import Optional, Tuple, List
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import trimesh

# Dependencia opcional: ezdxf
try:
    import ezdxf
    HAS_EZDXF = True
except Exception:
    HAS_EZDXF = False

# ========= Unidades (INSUNITS) =========
UNITS_MAP = {
    0: ("unitless", 1.0), 1: ("in", 0.0254), 2: ("ft", 0.3048), 3: ("mi", 1609.344),
    4: ("mm", 1e-3), 5: ("cm", 1e-2), 6: ("m", 1.0), 7: ("km", 1000.0),
    8: ("uin", 0.0254e-6), 9: ("mil", 25.4e-6), 10: ("yd", 0.9144),
    11: ("Å", 1e-10), 12: ("nm", 1e-9), 13: ("µm", 1e-6), 14: ("dm", 0.1),
    15: ("dam", 10.0), 16: ("hm", 100.0), 17: ("Gm", 1e9), 18: ("AU", 1.495978707e11),
    19: ("ly", 9.460730472e15), 20: ("pc", 3.085677581e16),
}
ASSUME_NAME_TO_METERS = {
    "unitless":1.0, "m":1.0, "meter":1.0, "meters":1.0, "mm":1e-3, "millimeter":1e-3,
    "millimeters":1e-3, "cm":1e-2, "centimeter":1e-2, "centimeters":1e-2,
    "dm":1e-1, "in":0.0254, "inch":0.0254, "inches":0.0254, "ft":0.3048, "feet":0.3048,
    "yd":0.9144, "yard":0.9144, "km":1000.0,
}

def cubic_suffix(u:str)->str:
    m={"unitless":"u³","in":"in³","ft":"ft³","mi":"mi³","mm":"mm³","cm":"cm³","m":"m³",
       "km":"km³","yd":"yd³","Å":"Å³","nm":"nm³","µm":"µm³","dm":"dm³","dam":"dam³",
       "hm":"hm³","Gm":"Gm³","AU":"AU³","ly":"ly³","pc":"pc³","uin":"uin³","mil":"mil³"}
    return m.get(u,f"{u}³")

def read_insunits(dxf_path:str, assume_units:Optional[str])->Tuple[str,float]:
    if assume_units:
        u=assume_units.strip().lower()
        return (u, ASSUME_NAME_TO_METERS.get(u,1.0)) if u in ASSUME_NAME_TO_METERS else ("unitless",1.0)
    if not HAS_EZDXF:
        return "unitless",1.0
    try:
        doc=ezdxf.readfile(dxf_path)
        code=doc.header.get("$INSUNITS",0)
        return UNITS_MAP.get(code,("unitless",1.0))
    except Exception:
        return "unitless",1.0

# ========= Carga robusta desde DXF con ezdxf =========
def _collect_faces_ezdxf(dxf_path:str, layer_regex:Optional[str]=None, dedup_tol:float=1e-6):
    """
    Devuelve listas (verts, faces) desde 3DFACE (y derivados) usando ezdxf.
    Aplica virtual_entities() para desanidar INSERT/BLOCK. Dedup con tolerancia.
    """
    import re
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    layer_ok = (lambda layer: True)
    if layer_regex:
        rx = re.compile(layer_regex, flags=re.IGNORECASE)
        layer_ok = lambda layer: bool(rx.search(layer or ""))

    verts: List[Tuple[float,float,float]] = []
    faces: List[Tuple[int,int,int]] = []
    # hash por grid snapping para agrupar cercanos
    vmap = {}  # key -> index

    def add_vertex(v):
        # tolerancia (en unidades DXF)
        q = max(float(dedup_tol), 1e-12)
        k = (round(v.x/q), round(v.y/q), round(v.z/q))
        idx = vmap.get(k)
        if idx is None:
            idx = len(verts)
            verts.append((float(v.x), float(v.y), float(v.z)))
            vmap[k] = idx
        return idx

    def add_triangle(a,b,c):
        # descartar triángulos degenerados
        va=np.array(verts[a]); vb=np.array(verts[b]); vc=np.array(verts[c])
        if np.linalg.norm(np.cross(vb-va, vc-va)) < 1e-16:
            return
        faces.append((a,b,c))

    def handle_face3d(face):
        try:
            ws = list(face.wcs_vertices())
        except Exception:
            ws = [face.dxf.vtx0, face.dxf.vtx1, face.dxf.vtx2, face.dxf.vtx3]
        pts = [p for p in ws if p is not None]
        # filtrar puntos consecutivos idénticos
        uniq = []
        for p in pts:
            if not uniq or (p != uniq[-1]):
                uniq.append(p)
        if len(uniq) < 3:
            return
        if len(uniq) == 3:
            i0=add_vertex(uniq[0]); i1=add_vertex(uniq[1]); i2=add_vertex(uniq[2])
            add_triangle(i0,i1,i2); return
        if len(uniq) >= 4:
            i0=add_vertex(uniq[0]); i1=add_vertex(uniq[1]); i2=add_vertex(uniq[2]); i3=add_vertex(uniq[3])
            add_triangle(i0,i1,i2)
            add_triangle(i0,i2,i3)

    for e in msp:
        try:
            layer = e.dxf.layer
        except Exception:
            layer = ""
        if not layer_ok(layer):
            continue

        processed = False
        # intentar virtual_entities
        try:
            for ve in e.virtual_entities():
                if ve.dxftype() == "3DFACE":
                    handle_face3d(ve)
                    processed = True
        except Exception:
            pass
        if processed:
            continue

        # 3DFACE directo
        try:
            if e.dxftype() == "3DFACE":
                handle_face3d(e)
                continue
        except Exception:
            pass

        # Fallback: explotar a 3DFACE
        try:
            if e.dxftype() in ("POLYFACE", "POLYLINE", "MESH"):
                from ezdxf import explode
                tmp = list(explode.virtual_entities(e))
                for ve in tmp:
                    if ve.dxftype() == "3DFACE":
                        handle_face3d(ve)
                continue
        except Exception:
            pass

    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)

def load_mesh_from_dxf_robust(dxf_path:str, thickness:Optional[float], layer_regex:Optional[str], dedup_tol:float)->trimesh.Trimesh:
    """
    (A) Parser ezdxf → malla; si falla, (B) trimesh.load(force='mesh').
    thickness: si no hay malla 3D y existen perfiles 2D, extruye (requiere shapely).
    """
    if HAS_EZDXF:
        try:
            V,F = _collect_faces_ezdxf(dxf_path, layer_regex=layer_regex, dedup_tol=dedup_tol)
            if V.size and F.size:
                mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
                if not mesh.is_empty:
                    return mesh
        except Exception as e:
            st.info(f"Parser ezdxf falló, probando fallback trimesh: {e}")

    # Fallback con trimesh
    geom = trimesh.load(dxf_path, force='mesh')
    if isinstance(geom, trimesh.Scene):
        meshes = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if meshes:
            return trimesh.util.concatenate(meshes)

    # ¿2D? → extruir si se pidió
    if thickness and thickness != 0:
        paths_2d = []
        if isinstance(geom, trimesh.Trimesh):
            pass
        elif hasattr(geom, "geometry"):
            paths_2d = [g for g in geom.geometry.values() if hasattr(g, "polygons_full")]
        elif hasattr(geom, "polygons_full"):
            paths_2d = [geom]
        polygons=[]
        for p in paths_2d: polygons.extend(p.polygons_full)
        if polygons:
            from shapely.ops import unary_union
            merged = unary_union(polygons)
            return trimesh.creation.extrude_polygon(merged, height=float(thickness))

    if isinstance(geom, trimesh.Trimesh):
        return geom

    raise ValueError("DXF no reconocido como malla ni perfiles 2D.")

# ========= Repair / Volumen / Plot =========
def repair_mesh(mesh:trimesh.Trimesh, weld_tol:Optional[float])->trimesh.Trimesh:
    # Merge de vértices cercano
    try:
        if weld_tol and weld_tol > 0:
            trimesh.repair.merge_vertices(mesh, weld_tol)
    except Exception:
        pass

    # Limpiezas
    for fn in ('remove_duplicate_faces', 'remove_degenerate_faces', 'remove_unreferenced_vertices'):
        try:
            getattr(mesh, fn)()
        except Exception:
            pass

    # Normales y orientación
    try: trimesh.repair.fix_normals(mesh)
    except Exception: pass
    try: trimesh.repair.fix_winding(mesh)
    except Exception: pass

    # Intento de cerrar agujeros
    try: trimesh.repair.fill_holes(mesh)
    except Exception: pass

    # Seleccionar componente mayor para evitar fragmentos
    try:
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            parts = sorted(parts, key=lambda m: (len(m.faces), m.bounding_box_oriented.volume if m.vertices.size else 0), reverse=True)
            mesh = parts[0]
    except Exception:
        pass

    try: mesh.process()
    except Exception: pass

    return mesh

def compute_volumes(mesh:trimesh.Trimesh, to_meters:float, voxel_pitch:Optional[float]=None):
    wt = bool(mesh.is_watertight)
    vol_native = None
    vol_m3 = None
    vol_voxel_m3 = None
    hull_m3 = None

    # Volumen nativo (si watertight)
    try:
        if wt:
            vol_native = float(mesh.volume)
            vol_m3 = vol_native * (to_meters**3)
    except Exception:
        wt = False

    # Envolvente convexa
    try:
        hull_m3 = float(mesh.convex_hull.volume) * (to_meters**3)
    except Exception:
        hull_m3 = None

    # Voxelización (aprox para no-watertight)
    if (not wt) and voxel_pitch and voxel_pitch > 0:
        try:
            vg = mesh.voxelized(voxel_pitch)  # VoxelGrid
            # VoxelGrid no siempre expone .volume; estimamos por número de celdas llenas * pitch^3
            filled = getattr(vg, 'filled_count', None)
            if filled is None:
                # fallback: contar ocupación por matriz
                mat = vg.matrix if hasattr(vg, 'matrix') else None
                filled = int(np.count_nonzero(mat)) if mat is not None else None
            if filled is not None:
                vol_voxel_m3 = float(filled) * float(voxel_pitch**3) * (to_meters**3)
        except Exception:
            vol_voxel_m3 = None

    return {
        "wt": wt,
        "vol_native": vol_native,
        "vol_m3": vol_m3,
        "vol_voxel_m3": vol_voxel_m3,
        "hull_m3": hull_m3
    }

def plot_trimesh(mesh:trimesh.Trimesh, title="Sólido DXF", opacity=0.85, show_axes=True, camera='data')->go.Figure:
    V,F=mesh.vertices, mesh.faces
    fig = go.Figure([
        go.Mesh3d(
            x=V[:,0], y=V[:,1], z=V[:,2], i=F[:,0], j=F[:,1], k=F[:,2],
            color="#4d88ff", opacity=float(opacity), flatshading=True,
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2)
        )
    ])
    if show_axes:
        # Ejes simples en el origen
        axes_len = float(np.ptp(V, axis=0).max()) if V.size else 1.0
        ax = [
            go.Scatter3d(x=[0, axes_len], y=[0,0], z=[0,0], mode="lines", line=dict(color="red", width=3), name="X"),
            go.Scatter3d(x=[0,0], y=[0, axes_len], z=[0,0], mode="lines", line=dict(color="green", width=3), name="Y"),
            go.Scatter3d(x=[0,0], y=[0,0], z=[0, axes_len], mode="lines", line=dict(color="blue", width=3), name="Z"),
        ]
        for a in ax: fig.add_trace(a)

    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        margin=dict(l=0,r=0,t=36,b=0)
    )
    if camera == 'xy':
        fig.update_layout(scene_camera=dict(eye=dict(x=0., y=0., z=2.5)))
    elif camera == 'xz':
        fig.update_layout(scene_camera=dict(eye=dict(x=0., y=2.5, z=0.)))
    elif camera == 'yz':
        fig.update_layout(scene_camera=dict(eye=dict(x=2.5, y=0., z=0.)))
    else:
        fig.update_layout(scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)))
    return fig

# ========= UI =========
st.set_page_config(page_title="DXF → Sólido y Volumen (robusto)", layout="wide")
st.title("DXF → Sólido 3D y Volumen")
st.subheader("Conversión robusta de DXF a sólido y cálculo de volumen — Herramienta creada por Sebastián Zúñiga Leyton")

with st.sidebar:
    st.header("Parámetros")
    assume_units = st.selectbox("Asumir unidades (si el DXF no trae $INSUNITS):",
        ["(auto)","mm","cm","m","in","ft","unitless"], index=0)
    layer_regex = st.text_input("Filtro por capa (regex opcional)", value="")
    dedup_tol = st.number_input("Tolerancia de deduplicado (unidades DXF)", min_value=0.0, value=1e-6, step=1e-6, format="%.9f")
    weld_tol = st.number_input("Tolerancia de soldado (unidades DXF)", min_value=0.0, value=0.0, step=0.0001, format="%.6f")
    thickness = st.number_input("Espesor extrusión (solo si DXF 2D)", min_value=0.0, value=0.0, step=0.1)
    voxel_pitch = st.number_input("Pitch voxel (aprox volumen si no watertight)", min_value=0.0, value=0.0, step=0.001, format="%.6f")
    simplify_for_view = st.checkbox("Simplificar solo para visualizar", value=True)
    target_faces = st.number_input("Caras objetivo visualización", min_value=1000, value=30000, step=1000)
    opacity = st.slider("Opacidad malla", 0.1, 1.0, 0.85, 0.05)
    show_axes = st.checkbox("Mostrar ejes XYZ", value=True)
    camera = st.selectbox("Cámara (vista)", ["data","xy","xz","yz"], index=0)

uploaded = st.file_uploader("Sube tu DXF", type=["dxf"])
if not uploaded:
    st.info("📁 Sube un archivo DXF para comenzar.")
    st.stop()

with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = os.path.join(tmpdir, uploaded.name)
    with open(tmp_path, "wb") as f: f.write(uploaded.read())

    unit_name, to_meters = read_insunits(tmp_path, None if assume_units=="(auto)" else assume_units)
    st.write(f"**Unidades DXF:** `{unit_name}` (1 {unit_name} = {to_meters} m)")

    # Carga robusta
    try:
        mesh = load_mesh_from_dxf_robust(
            tmp_path,
            thickness=(thickness if thickness>0 else None),
            layer_regex=(layer_regex or None),
            dedup_tol=dedup_tol
        )
    except Exception as e:
        st.error(f"Error al cargar el DXF (parser robusto + fallback): {e}")
        st.stop()

    # Reparación
    mesh = repair_mesh(mesh, weld_tol=(weld_tol if weld_tol>0 else None))

    # Métricas
    parts = mesh.split(only_watertight=False)
    main = parts[0] if len(parts) else mesh
    st.write({"componentes": len(parts), "vertices_main": int(len(main.vertices)), "faces_main": int(len(main.faces))})

    # Volúmenes
    vols = compute_volumes(main, to_meters, voxel_pitch=(voxel_pitch if voxel_pitch>0 else None))
    cubic = cubic_suffix(unit_name)

    colA,colB,colC,colD = st.columns(4)
    colA.metric("Vértices", f"{main.vertices.shape[0]:,}")
    colB.metric("Caras", f"{main.faces.shape[0]:,}")
    colC.metric("Watertight", "Sí" if vols["wt"] else "No")
    colD.metric("Volumen (m³)" if vols["wt"] else "Aprox (voxel/convex) m³",
                f"{(vols['vol_m3'] if vols['wt'] and vols['vol_m3'] is not None else (vols['vol_voxel_m3'] or vols['hull_m3'] or 0.0)):,.6f}")

    st.markdown("### Resultados")
    if vols["wt"] and vols["vol_native"] is not None:
        st.success(f"Volumen (nativo): `{vols['vol_native']:,.6f} {cubic}`")
        st.success(f"Volumen (m³): `{vols['vol_m3']:,.6f} m³`")
    else:
        if vols["vol_voxel_m3"] is not None:
            st.warning(f"Volumen aproximado por voxelización (m³): `{vols['vol_voxel_m3']:,.6f}` (malla abierta)")
        if vols["hull_m3"] is not None:
            st.info(f"Envolvente convexa (m³): `{vols['hull_m3']:,.6f}` (sobre-estima)")

    # Visualización (opcionalmente simplificada)
    mesh_view = main.copy()
    if simplify_for_view and mesh_view.faces.shape[0] > target_faces:
        step = max(int(mesh_view.faces.shape[0]/target_faces), 1)
        mesh_view.update_faces(np.arange(0, mesh_view.faces.shape[0], step))
        mesh_view.remove_unreferenced_vertices()

    # Wireframe opcional
    fig = plot_trimesh(mesh_view, title=os.path.basename(uploaded.name), opacity=opacity, show_axes=show_axes, camera=camera)
    st.plotly_chart(fig, use_container_width=True)
    if st.checkbox("Mostrar wireframe", value=False):
        edges = mesh_view.edges_unique
        ex = mesh_view.vertices[edges].reshape(-1, 2, 3)
        fig_w = go.Figure([
            go.Scatter3d(
                x=seg[:,0], y=seg[:,1], z=seg[:,2],
                mode="lines", line=dict(color="gray", width=1),
                name="Wireframe", showlegend=False
            ) for seg in ex
        ])
        fig_w.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0,r=0,t=36,b=0), title="Wireframe")
        st.plotly_chart(fig_w, use_container_width=True)

    # Exportaciones
    st.markdown("### Exportar")
    stl_bytes = io.BytesIO()
    try:
        mesh_view.export(stl_bytes, file_type="stl"); stl_bytes.seek(0)
        st.download_button("⬇️ Descargar STL", stl_bytes, os.path.splitext(uploaded.name)[0]+".stl", "application/sla")
    except Exception as e:
        st.info(f"No se pudo generar STL: {e}")

    rep = io.StringIO()
    rep.write(f"Archivo DXF: {uploaded.name}\n")
    rep.write(f"Unidades DXF: {unit_name} (1 {unit_name} = {to_meters} m)\n")
    rep.write(f"Vertices (main): {main.vertices.shape[0]}, Caras (main): {main.faces.shape[0]}\n")
    rep.write(f"Watertight: {vols['wt']}\n")
    if vols["wt"] and vols["vol_native"] is not None:
        rep.write(f"Volumen nativo: {vols['vol_native']:.9f} {cubic}\n")
        rep.write(f"Volumen m3: {vols['vol_m3']:.9f} m^3\n")
    else:
        if vols["vol_voxel_m3"] is not None:
            rep.write(f"Volumen voxel (m3): {vols['vol_voxel_m3']:.9f}\n")
        if vols["hull_m3"] is not None:
            rep.write(f"Convex hull (m3): {vols['hull_m3']:.9f}\n")
    st.download_button("⬇️ Descargar Reporte TXT", rep.getvalue(),
                       os.path.splitext(uploaded.name)[0]+"_reporte_volumen.txt", "text/plain")

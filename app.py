
# -*- coding: utf-8 -*-
"""
DXF → Sólido 3D y Volumen (simple)
- Asume volumen en m³ (convierte desde INSUNITS si existe).
- Sólido 3D siempre visible; wireframe opcional (activado por defecto).
- Sin ejes, sin cámara XY/XZ/YZ, sin parámetros avanzados en la UI.
- 'Calcular' dispara todo el pipeline.
"""

import os, io, tempfile, traceback
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import trimesh

# ezdxf (opcional) para parse robusto
try:
    import ezdxf
    HAS_EZDXF = True
except Exception:
    HAS_EZDXF = False

# --- Unidades DXF → metros (INSUNITS) ---
UNITS_MAP = {
    0: ("unitless", 1.0), 1: ("in", 0.0254), 2: ("ft", 0.3048), 3: ("mi", 1609.344),
    4: ("mm", 1e-3), 5: ("cm", 1e-2), 6: ("m", 1.0), 7: ("km", 1000.0)
}

def read_insunits_to_meters(dxf_path: str) -> float:
    """Lee $INSUNITS si existe y retorna factor a metros. Si falla, asume 1.0."""
    if not HAS_EZDXF:
        return 1.0
    try:
        doc = ezdxf.readfile(dxf_path)
        code = doc.header.get("$INSUNITS", 0)
        return UNITS_MAP.get(code, ("unitless", 1.0))[1]
    except Exception:
        return 1.0

# --- Parser DXF mínimo (3DFACE + virtual_entities) ---
def _collect_faces_ezdxf(dxf_path: str):
    """Extrae (vertices, faces) desde 3DFACE usando ezdxf; expande virtual_entities si es posible."""
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    verts = []
    faces = []
    vmap = {}

    def add_vertex(v):
        # dedup interno con tolerancia fija pequeña (no expuesta en UI)
        q = 1e-6
        k = (round(v.x / q), round(v.y / q), round(v.z / q))
        idx = vmap.get(k)
        if idx is None:
            idx = len(verts)
            verts.append((float(v.x), float(v.y), float(v.z)))
            vmap[k] = idx
        return idx

    def add_triangle(a, b, c):
        va = np.array(verts[a]); vb = np.array(verts[b]); vc = np.array(verts[c])
        if np.linalg.norm(np.cross(vb - va, vc - va)) < 1e-16:
            return
        faces.append((a, b, c))

    def handle_face3d(face):
        try:
            ws = list(face.wcs_vertices())
        except Exception:
            ws = [face.dxf.vtx0, face.dxf.vtx1, face.dxf.vtx2, face.dxf.vtx3]
        pts = [p for p in ws if p is not None]
        if len(pts) < 3:
            return
        if len(pts) == 3:
            i0 = add_vertex(pts[0]); i1 = add_vertex(pts[1]); i2 = add_vertex(pts[2])
            add_triangle(i0, i1, i2); return
        # cuadrilátero -> dos triángulos
        i0 = add_vertex(pts[0]); i1 = add_vertex(pts[1]); i2 = add_vertex(pts[2]); i3 = add_vertex(pts[3])
        add_triangle(i0, i1, i2)
        add_triangle(i0, i2, i3)

    for e in msp:
        processed = False
        try:
            for ve in e.virtual_entities():
                if ve.dxftype() == "3DFACE":
                    handle_face3d(ve)
                    processed = True
        except Exception:
            pass
        if processed:
            continue
        try:
            if e.dxftype() == "3DFACE":
                handle_face3d(e)
        except Exception:
            pass

    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)

def load_mesh_from_dxf_simple(dxf_path: str) -> trimesh.Trimesh:
    """Primero intenta ezdxf; si falla, usa trimesh.load(force='mesh')."""
    if HAS_EZDXF:
        try:
            V, F = _collect_faces_ezdxf(dxf_path)
            if V.size and F.size:
                mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
                if not mesh.is_empty:
                    return mesh
        except Exception:
            pass

    geom = trimesh.load(dxf_path, force='mesh')
    if isinstance(geom, trimesh.Scene):
        meshes = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if meshes:
            return trimesh.util.concatenate(meshes)
    if isinstance(geom, trimesh.Trimesh):
        return geom
    raise ValueError("No se encontró una malla 3D en el DXF.")

# --- Reparación mínima ---
def repair_mesh_simple(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    for fn in ('remove_duplicate_faces', 'remove_degenerate_faces', 'remove_unreferenced_vertices'):
        try:
            getattr(mesh, fn)()
        except Exception:
            pass
    try: trimesh.repair.fix_normals(mesh)
    except Exception: pass
    try: trimesh.repair.fill_holes(mesh)
    except Exception: pass
    try: mesh.process()
    except Exception: pass
    return mesh

# --- Split seguro (networkx/scipy) ---
def split_components_safe(mesh: trimesh.Trimesh):
    try:
        return mesh.split(only_watertight=False, engine='networkx')
    except Exception:
        try:
            return mesh.split(only_watertight=False)
        except Exception as e:
            st.info(f"Split no disponible. Usando malla completa. Detalle: {e}")
            return [mesh]

# --- Volumen (m³) ---
def compute_volume_m3(mesh: trimesh.Trimesh, to_meters: float) -> (float, bool):
    """Retorna (volumen_m3, es_watertight). Si no watertight, usa convex hull como aproximación."""
    wt = bool(mesh.is_watertight)
    if wt:
        try:
            vol_native = float(mesh.volume)
            return vol_native * (to_meters ** 3), True
        except Exception:
            wt = False
    # aproximación con convex hull
    try:
        vol_hull_m3 = float(mesh.convex_hull.volume) * (to_meters ** 3)
        return vol_hull_m3, False
    except Exception:
        return 0.0, False

# --- Visualización (sólido siempre, wireframe opcional) ---
def render_mesh_scene(mesh: trimesh.Trimesh, title: str, opacity: float = 0.9, show_wireframe: bool = True) -> go.Figure:
    V, F = mesh.vertices, mesh.faces
    fig = go.Figure()
    # Sólido
    if F is not None and len(F) > 0:
        fig.add_trace(go.Mesh3d(
            x=V[:, 0].tolist(), y=V[:, 1].tolist(), z=V[:, 2].tolist(),
            i=F[:, 0].tolist(), j=F[:, 1].tolist(), k=F[:, 2].tolist(),
            color="#4d88ff", opacity=float(opacity), flatshading=True,
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2),
            name="Sólido"
        ))
    else:
        hull = mesh.convex_hull
        HV, HF = hull.vertices, hull.faces
        fig.add_trace(go.Mesh3d(
            x=HV[:, 0].tolist(), y=HV[:, 1].tolist(), z=HV[:, 2].tolist(),
            i=HF[:, 0].tolist(), j=HF[:, 1].tolist(), k=HF[:, 2].tolist(),
            color="#4d88ff", opacity=float(opacity), flatshading=True,
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2),
            name="Hull (fallback)"
        ))
    # Wireframe opcional
    if show_wireframe:
        try:
            edges = mesh.edges_unique
            # limitar por rendimiento
            max_segments = 20000
            if len(edges) > max_segments:
                idx = np.random.choice(len(edges), size=max_segments, replace=False)
                edges = edges[idx]
            segs = mesh.vertices[edges].reshape(-1, 2, 3)
            xs = segs[:, :, 0].ravel(order="C")
            ys = segs[:, :, 1].ravel(order="C")
            zs = segs[:, :, 2].ravel(order="C")
            n = len(segs)
            xw = np.empty(n * 3, dtype=float); yw = np.empty(n * 3, dtype=float); zw = np.empty(n * 3, dtype=float)
            xw[0::3] = xs[0::2]; xw[1::3] = xs[1::2]; xw[2::3] = np.nan
            yw[0::3] = ys[0::2]; yw[1::3] = ys[1::2]; yw[2::3] = np.nan
            zw[0::3] = zs[0::2]; zw[1::3] = zs[1::2]; zw[2::3] = np.nan
            fig.add_trace(go.Scatter3d(
                x=xw.tolist(), y=yw.tolist(), z=zw.tolist(),
                mode="lines", line=dict(color="gray", width=2),
                name="Wireframe", showlegend=True
            ))
        except Exception:
            pass

    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=36, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    )
    return fig

# --- UI mínima ---
st.set_page_config(page_title="DXF → Sólido 3D y Volumen (m³)", layout="wide")
st.title("DXF → Sólido 3D y Volumen (m³)")
st.caption("Sube tu DXF, elige si ver wireframe y presiona **Calcular**. Herramienta creada por Sebastián Zúñiga Leyton")

uploaded = st.file_uploader("Sube tu DXF", type=["dxf"])
show_wireframe = st.checkbox("Wireframe (superponer)", value=True)  # activado por defecto
if st.button("Calcular"):
    if not uploaded:
        st.warning("Primero sube un archivo DXF.")
        st.stop()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, uploaded.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())

        # Unidades → metros (m³)
        to_meters = read_insunits_to_meters(tmp_path)

        try:
            mesh = load_mesh_from_dxf_simple(tmp_path)
        except Exception as e:
            st.error(f"Error al cargar el DXF: {e}")
            st.code(traceback.format_exc())
            st.stop()

        mesh = repair_mesh_simple(mesh)

        parts = split_components_safe(mesh)
        main = parts[0] if len(parts) else mesh

        vol_m3, wt = compute_volume_m3(main, to_meters)

        # Métricas
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Vértices", f"{main.vertices.shape[0]:,}")
        colB.metric("Caras", f"{main.faces.shape[0]:,}")
        colC.metric("Watertight", "Sí" if wt else "No")
        colD.metric("Volumen (m³)" if wt else "Aprox (m³)", f"{vol_m3:,.6f}")

        # Visual
        try:
            mesh_view = main.copy()
            fig = render_mesh_scene(mesh_view, title=os.path.basename(uploaded.name), show_wireframe=show_wireframe)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.error("Ocurrió un error al renderizar el sólido.")
            st.code(traceback.format_exc())

        # Exportar
        st.markdown("### Exportar")
        # STL
        stl_bytes = io.BytesIO()
        try:
            mesh_view.export(stl_bytes, file_type="stl"); stl_bytes.seek(0)
            st.download_button("⬇️ Descargar STL", stl_bytes, os.path.splitext(uploaded.name)[0] + ".stl", "application/sla")
        except Exception as e:
            st.info(f"No se pudo generar STL: {e}")
        # Reporte
        rep = io.StringIO()
        rep.write(f"Archivo DXF: {uploaded.name}\n")
        rep.write(f"Watertight: {wt}\n")
        rep.write(f"Volumen (m3): {vol_m3:.9f}\n")
        st.download_button("⬇️ Descargar Reporte TXT", rep.getvalue(),
                           os.path.splitext(uploaded.name)[0] + "_reporte_volumen.txt", "text/plain")

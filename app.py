
# -*- coding: utf-8 -*-
"""
DXF → Sólido 3D y Volumen (robusto y mínimo-invasivo)

- Carga DXF con ezdxf (prioriza 3DFACE; hooks tolerantes para MESH/POLYFACE).
- Convierte cualquier entrada (dict/(V,F)/Trimesh) a trimesh.Trimesh.
- Repara malla, selecciona componente mayor, calcula y muestra volumen SIEMPRE.
- Visualiza en 3D con Plotly.
- Exporta STL y reporte TXT.

Autor original: Sebastián Zúñiga Leyton
Ajustes: M365 Copilot
"""

import io
import os
import numpy as np
import streamlit as st
import tempfile
import trimesh
import plotly.graph_objects as go
import ezdxf

# -------------------- Utilidades de conversión y reparación --------------------

def to_trimesh_safe(mesh_like):
    """Convierte dict/(V,F)/Trimesh a trimesh.Trimesh de forma segura."""
    if isinstance(mesh_like, trimesh.Trimesh):
        return mesh_like
    if isinstance(mesh_like, dict) and 'vertices' in mesh_like and 'faces' in mesh_like:
        return trimesh.Trimesh(
            vertices=np.asarray(mesh_like['vertices'], dtype=float),
            faces=np.asarray(mesh_like['faces'], dtype=np.int64),
            process=False
        )
    if isinstance(mesh_like, (tuple, list)) and len(mesh_like) == 2:
        v, f = mesh_like
        return trimesh.Trimesh(
            vertices=np.asarray(v, dtype=float),
            faces=np.asarray(f, dtype=np.int64),
            process=False
        )
    raise TypeError(f"Formato inesperado de malla: {type(mesh_like)}")

def pick_largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Devuelve el componente conectado con más caras (representativo)."""
    try:
        comps = mesh.split(only_watertight=False)
        return max(comps, key=lambda m: len(m.faces)) if comps else mesh
    except Exception:
        return mesh

def repair_mesh(mesh_like, weld_tol: float | None = None) -> trimesh.Trimesh:
    """
    Conversión + reparaciones básicas. Mantiene tu firma y comportamiento.
    """
    mesh = to_trimesh_safe(mesh_like)

    # Soldar vértices si hay tolerancia > 0
    if weld_tol is not None and weld_tol > 0:
        try:
            trimesh.repair.merge_vertices(mesh, weld_tol)
        except Exception:
            pass

    # Limpiezas tolerantes a errores
    for fn in ('remove_duplicate_faces', 'remove_degenerate_faces', 'remove_unreferenced_vertices'):
        try:
            getattr(mesh, fn)()
        except Exception:
            pass

    # Normales y procesado
    try: trimesh.repair.fix_normals(mesh)
    except Exception: pass
    try: mesh.process()
    except Exception: pass

    return mesh

def compute_volume_safe(mesh: trimesh.Trimesh, try_fill_holes: bool = True):
    """
    Calcula volumen de forma robusta y devuelve (volumen, estado, nota).
    estado: 'ok' | 'approx' | 'error'
    """
    try:
        m = pick_largest_component(mesh.copy())

        # Limpiezas suaves
        for fn in ('remove_degenerate_faces', 'remove_duplicate_faces', 'remove_unreferenced_vertices'):
            try:
                getattr(m, fn)()
            except Exception:
                pass

        # Normales y procesado
        try: trimesh.repair.fix_normals(m)
        except Exception: pass
        try: m.process()
        except Exception: pass

        # Intentar cerrar agujeros (opcional)
        if try_fill_holes and not m.is_watertight:
            try:
                trimesh.repair.fill_holes(m)
                m.process()
            except Exception:
                pass

        # Volumen
        if m.is_watertight:
            return float(m.volume), 'ok', 'Malla watertight: volumen confiable.'
        else:
            # Volumen aproximado (puede ser inexacto)
            try:
                vol = float(m.volume)
                return vol, 'approx', 'Malla no watertight: volumen aproximado (podría ser inexacto).'
            except Exception:
                # Fallback: envolvente convexa (sobre-estima)
                try:
                    hull = m.convex_hull
                    return float(hull.volume), 'approx', 'Usando envolvente convexa como aproximación (sobre-estima).'
                except Exception:
                    return None, 'error', 'No fue posible calcular el volumen (malla abierta o degenerada).'
    except Exception as e:
        return None, 'error', f'Error inesperado: {e}'

# -------------------- Parseo DXF (3DFACE + hooks tolerantes) --------------------

def dxf_to_mesh_data(dxf_path: str):
    """
    Convierte entidades 3DFACE de un DXF a (vertices, faces).
    Incluye hooks tolerantes para MESH y POLYFACE (si el DXF lo permite).
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    def get_index(v: np.ndarray, table: list[list[float]], atol=1e-9):
        # Busca si un vértice igual (aprox) existe; si no, lo agrega.
        for i, t in enumerate(table):
            if np.allclose(v, t, atol=atol):
                return i
        table.append([float(v[0]), float(v[1]), float(v[2])])
        return len(table) - 1

    # --- 3DFACE ---
    count_3dface = 0
    for e in msp.query("3DFACE"):
        v0 = np.array(e.dxf.vtx0, dtype=float)
        v1 = np.array(e.dxf.vtx1, dtype=float)
        v2 = np.array(e.dxf.vtx2, dtype=float)
        v3 = np.array(e.dxf.vtx3, dtype=float) if hasattr(e.dxf, "vtx3") else v2

        idx = [get_index(v0, vertices), get_index(v1, vertices), get_index(v2, vertices)]
        faces.append(idx)
        count_3dface += 1

        if not np.allclose(v3, v2):
            idx2 = [get_index(v0, vertices), get_index(v2, vertices), get_index(v3, vertices)]
            faces.append(idx2)

    # --- Hooks simples para MESH ---
    count_mesh = 0
    for e in msp.query("MESH"):
        try:
            vs = np.asarray(e.vertices, dtype=float)
            fs = np.asarray(e.faces, dtype=np.int64)
            if vs.size > 0 and fs.size > 0:
                base = len(vertices)
                for v in vs:
                    vertices.append([float(v[0]), float(v[1]), float(v[2])])
                for f in fs:
                    faces.append([int(base + f[0]), int(base + f[1]), int(base + f[2])])
                count_mesh += 1
        except Exception:
            pass

    # --- Hooks simples para POLYFACE (dependiente de DXF) ---
    count_polyface = 0
    for pl in msp.query("POLYLINE"):
        try:
            if hasattr(pl, "is_poly_face_mesh") and pl.is_poly_face_mesh:
                vlist = []
                flist = []
                for v in pl.vertices():
                    if hasattr(v, "is_face_record") and v.is_face_record:
                        idxs = [v.dxf.vtx0, v.dxf.vtx1, v.dxf.vtx2, v.dxf.vtx3]
                        idxs = [i for i in idxs if i not in (0, None)]
                        if len(idxs) >= 3:
                            flist.append(idxs[:3])
                    else:
                        loc = np.array(v.dxf.location, dtype=float)
                        vlist.append([float(loc[0]), float(loc[1]), float(loc[2])])

                if len(vlist) > 0 and len(flist) > 0:
                    base = len(vertices)
                    vertices.extend(vlist)
                    for f in flist:  # 1-based → 0-based
                        faces.append([base + (int(i) - 1) for i in f[:3]])
                    count_polyface += 1
        except Exception:
            pass

    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)

    meta = {
        "count_3dface": count_3dface,
        "count_mesh": count_mesh,
        "count_polyface": count_polyface,
        "total_vertices": int(len(vertices)),
        "total_faces": int(len(faces))
    }
    return vertices, faces, meta

# -------------------- App Streamlit --------------------

st.set_page_config(page_title="DXF → Volumen 3D", layout="wide")
st.title("DXF → Sólido 3D y Volumen")

st.markdown(
    "Sube un **DXF** con entidades 3D (idealmente `3DFACE`). "
    "La app convertirá a malla, intentará repararla, mostrará la vista 3D y calculará el volumen."
)

uploaded = st.file_uploader("Sube tu archivo DXF", type=["dxf"])

col_opts = st.columns(4)
with col_opts[0]:
    weld_tol = st.number_input("Tolerancia de soldadura (unidades DXF)", min_value=0.0, value=0.0, step=0.001, format="%.6f")
with col_opts[1]:
    scale = st.number_input("Factor de escala (ej: mm→m = 0.001)", min_value=0.0, value=1.0, step=0.001, format="%.6f")
with col_opts[2]:
    try_fill_holes = st.checkbox("Intentar cerrar agujeros (fill_holes)", value=True)
with col_opts[3]:
    show_wireframe = st.checkbox("Wireframe", value=False)

mesh_opacity = st.slider("Opacidad de la malla", 0.1, 1.0, 0.6, 0.05)

if uploaded is not None:
    try:
        # Guardar a archivo temporal para ezdxf
        with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tmp:
            tmp.write(uploaded.read())
            dxf_path = tmp.name

        st.info("Leyendo DXF…")
        vertices, faces, meta = dxf_to_mesh_data(dxf_path)

        if len(vertices) == 0 or len(faces) == 0:
            st.error("No se encontraron caras triangulables. Necesitas entidades 3D (ej. `3DFACE`).")
            st.stop()

        # Aplicar factor de escala (mm→m, etc.)
        if scale > 0 and scale != 1.0:
            vertices = vertices * float(scale)

        st.write("Resumen DXF:", meta)

        # Reparación mínima (mantiene tu flujo)
        st.info("Reparando malla…")
        mesh = repair_mesh((vertices, faces), weld_tol=(weld_tol if weld_tol > 0 else None))

        # Cálculo robusto de volumen (SIEMPRE muestra algo)
        vol, state, note = compute_volume_safe(mesh, try_fill_holes=try_fill_holes)
        if vol is not None:
            if state == 'ok':
                st.success(f"Volumen: {vol:.6f} u³")
            else:
                st.warning(f"Volumen (aprox): {vol:.6f} u³")
            st.caption(note)
        else:
            st.error("No se pudo calcular el volumen.")
            st.caption(note)
        vol_text = "n/a" if vol is None else f"{vol:.6f}"

        # Visualización: componente mayor para evitar fragmentos
        st.info("Renderizando vista 3D…")
        mesh_draw = pick_largest_component(mesh)
        x, y, z = mesh_draw.vertices[:, 0], mesh_draw.vertices[:, 1], mesh_draw.vertices[:, 2]
        i, j, k = mesh_draw.faces[:, 0], mesh_draw.faces[:, 1], mesh_draw.faces[:, 2]

        data = [
            go.Mesh3d(
                x=x, y=y, z=z, i=i, j=j, k=k,
                opacity=float(mesh_opacity), color="lightskyblue", name="Malla"
            )
        ]
        if show_wireframe:
            edges = mesh_draw.edges_unique
            ex = mesh_draw.vertices[edges].reshape(-1, 2, 3)
            for seg in ex:
                data.append(
                    go.Scatter3d(
                        x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
                        mode="lines", line=dict(color="gray", width=1),
                        name="Wireframe", showlegend=False
                    )
                )

        fig = go.Figure(data=data)
        fig.update_layout(scene=dict(aspectmode="data"),
                          margin=dict(l=0, r=0, t=30, b=0),
                          title="Malla 3D (componente mayor)")
        st.plotly_chart(fig, use_container_width=True)

        # Diagnóstico breve
        st.write({
            "n_vertices": int(len(mesh.vertices)),
            "n_faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight)
        })

        # Exportaciones
        c1, c2 = st.columns(2)
        with c1:
            try:
                stl_bytes = mesh.export(file_type="stl")
                st.download_button("Descargar STL", data=stl_bytes,
                                   file_name="malla.stl", mime="application/octet-stream")
            except Exception:
                st.error("No se pudo exportar STL.")
        with c2:
            reporte = io.StringIO()
            reporte.write("=== REPORTE MALLA DXF ===\n")
            reporte.write(f"Vertices: {len(mesh.vertices)}\n")
            reporte.write(f"Faces: {len(mesh.faces)}\n")
            reporte.write(f"Watertight: {mesh.is_watertight}\n")
            reporte.write(f"Volumen (u³): {vol_text}\n")
            reporte.write("\n--- Meta DXF ---\n")
            for k, v in meta.items():
                reporte.write(f"{k}: {v}\n")
            st.download_button("Descargar reporte TXT",
                               data=reporte.getvalue(),
                               file_name="reporte.txt", mime="text/plain")

    except Exception as e:
        import traceback
        st.error("Ocurrió un error procesando la malla. Detalle debajo o revisa los logs (Manage app → Logs).")
        st.code(traceback.format_exc())
        st.stop()
    finally:
        # Limpiar el archivo temporal
        try:
            if 'dxf_path' in locals() and os.path.exists(dxf_path):
                os.remove(dxf_path)
        except Exception:
            pass
else:
    st.info("Carga un archivo DXF para comenzar.")

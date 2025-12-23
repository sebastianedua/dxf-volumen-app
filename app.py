
# -*- coding: utf-8 -*-
"""
DXF → Sólido 3D y Volumen (robusto)

- Lee DXF con ezdxf (3DFACE prioritario; hooks para MESH/POLYFACE).
- Convierte a trimesh.Trimesh de forma segura.
- Repara malla, verifica watertight y calcula volumen.
- Visualiza en 3D con Plotly.
- Exporta STL y Reporte TXT.

Autor: Sebastián Zúñiga Leyton 
"""

import io
import os
import numpy as np
import streamlit as st
import tempfile
import trimesh
import plotly.graph_objects as go

# ---------- Utilidades de conversión y reparación ----------

def to_trimesh(mesh_like):
    """
    Convierte diversas representaciones (Trimesh, dict, (V,F)) a trimesh.Trimesh.
    Lanza TypeError si no reconoce el formato.
    """
    if isinstance(mesh_like, trimesh.Trimesh):
        return mesh_like

    # dict con 'vertices' y 'faces'
    if isinstance(mesh_like, dict) and 'vertices' in mesh_like and 'faces' in mesh_like:
        return trimesh.Trimesh(
            vertices=np.asarray(mesh_like['vertices'], dtype=float),
            faces=np.asarray(mesh_like['faces'], dtype=np.int64),
            process=False
        )

    # tupla/lista (vertices, faces)
    if isinstance(mesh_like, (tuple, list)) and len(mesh_like) == 2:
        vertices, faces = mesh_like
        return trimesh.Trimesh(
            vertices=np.asarray(vertices, dtype=float),
            faces=np.asarray(faces, dtype=np.int64),
            process=False
        )

    raise TypeError(
        f"Formato de malla no soportado: {type(mesh_like)}. "
        "Usa trimesh.Trimesh, dict {'vertices','faces'} o (vertices, faces)."
    )


def repair_mesh(mesh_like, weld_tol: float | None = None) -> trimesh.Trimesh:
    """
    Convierte a Trimesh, aplica reparaciones básicas y opcionalmente suelda vértices con tolerancia.
    """
    mesh = to_trimesh(mesh_like)

    # Soldar vértices cercanos si se define tolerancia positiva
    if weld_tol is not None and weld_tol > 0:
        try:
            trimesh.repair.merge_vertices(mesh, weld_tol)
        except Exception:
            # Si merge falla por geometría rara, continuamos
            pass

    # Limpiezas
    try:
        mesh.remove_duplicate_faces()
    except Exception:
        pass
    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    # Normales y procesado general
    try:
        trimesh.repair.fix_normals(mesh)
    except Exception:
        pass
    try:
        mesh.process()
    except Exception:
        pass

    # Intento opcional de cerrar agujeros
    if not mesh.is_watertight:
        try:
            trimesh.repair.fill_holes(mesh)
            mesh.process()
        except Exception:
            pass

    return mesh


# ---------- Parseo DXF (3DFACE principal; hooks para MESH/POLYFACE) ----------

import ezdxf

def dxf_to_mesh_data(dxf_path: str):
    """
    Convierte entidades 3DFACE de un DXF a (vertices, faces).
    Si no hay 3DFACE, intenta hooks básicos para MESH/POLYFACE (silenciosos si fallan).
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
        # 3DFACE normalmente trae vtx0, vtx1, vtx2, vtx3 (el cuarto puede duplicar el 3º)
        v0 = np.array(e.dxf.vtx0, dtype=float)
        v1 = np.array(e.dxf.vtx1, dtype=float)
        v2 = np.array(e.dxf.vtx2, dtype=float)
        v3 = np.array(e.dxf.vtx3, dtype=float) if hasattr(e.dxf, "vtx3") else v2

        # Triangulamos como cara triangular (0,1,2)
        idx = [get_index(v0, vertices), get_index(v1, vertices), get_index(v2, vertices)]
        faces.append(idx)
        count_3dface += 1

        # Si el cuarto vértice es distinto, podemos crear otra cara (0,2,3)
        if not np.allclose(v3, v2):
            idx2 = [get_index(v0, vertices), get_index(v2, vertices), get_index(v3, vertices)]
            faces.append(idx2)

    # --- Hooks simples para MESH (si existe y expone vertices/faces) ---
    count_mesh = 0
    for e in msp.query("MESH"):
        try:
            # Algunas exportaciones traen .vertices y .faces directamente
            vs = np.asarray(e.vertices, dtype=float)  # puede fallar si no existe
            fs = np.asarray(e.faces, dtype=np.int64)
            if vs.size > 0 and fs.size > 0:
                # Reindexamos para mezclar con los ya acumulados (si hubiera)
                base = len(vertices)
                for v in vs:
                    vertices.append([float(v[0]), float(v[1]), float(v[2])])
                for f in fs:
                    faces.append([int(base + f[0]), int(base + f[1]), int(base + f[2])])
                count_mesh += 1
        except Exception:
            # Ignorar si la estructura de MESH no es accesible en esta versión
            pass

    # --- Hooks simples para POLYFACE (muy dependiente de versión DXF; dejamos silencioso) ---
    count_polyface = 0
    for pl in msp.query("POLYLINE"):
        # En algunos DXF POLYLINE representa mallas tipo Polyface; intentar lectura básica
        try:
            if hasattr(pl, "is_poly_face_mesh") and pl.is_poly_face_mesh:
                # Nota: El acceso a vértices y caras depende fuertemente del DXF y ezdxf.
                # Aquí intentamos una lectura defensiva. Si falla, se ignora silenciosamente.
                vlist = []
                flist = []
                for v in pl.vertices():  # VERTEX records
                    if hasattr(v, "is_face_record") and v.is_face_record:
                        # Cara: índices referenciando vértices (a menudo 1-based)
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
                    # Ajustar 1-based → 0-based y reindexar con base
                    for f in flist:
                        faces.append([base + (i - 1) for i in f[:3]])
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


# ---------- App Streamlit ----------

st.set_page_config(page_title="DXF → Volumen 3D", layout="wide")
st.title("DXF → Sólido 3D y Volumen")

st.markdown(
    "Sube un archivo **DXF** con entidades 3D (idealmente `3DFACE`). "
    "La app convertirá a malla, intentará repararla, mostrará la vista 3D y calculará el volumen. Herramiemnta creada por Sebastián Zúñiga Leyton"
)

uploaded = st.file_uploader("Sube tu archivo DXF", type=["dxf"])
weld_tol = st.number_input(
    "Tolerancia de soldadura (unidades DXF) — opcional",
    min_value=0.0, value=0.0, step=0.001, format="%.6f"
)

col_opts = st.columns(3)
with col_opts[0]:
    do_fill_holes = st.checkbox("Intentar cerrar agujeros (fill_holes)", value=True)
with col_opts[1]:
    show_wireframe = st.checkbox("Mostrar wireframe adicional", value=False)
with col_opts[2]:
    mesh_opacity = st.slider("Opacidad de la malla", 0.1, 1.0, 0.6, 0.05)

if uploaded is not None:
    try:
        # Guardar a archivo temporal para ezdxf (necesita ruta)
        with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tmp:
            tmp.write(uploaded.read())
            dxf_path = tmp.name

        st.info("Leyendo y convirtiendo DXF…")
        vertices, faces, meta = dxf_to_mesh_data(dxf_path)
        st.write("Resumen DXF:", meta)

        if len(vertices) == 0 or len(faces) == 0:
            st.error(
                "No se encontraron caras triangulables. "
                "Asegúrate de que el DXF tenga entidades 3D (por ejemplo `3DFACE`)."
            )
            st.stop()

        # Reparación
        st.info("Reparando malla…")
        mesh = repair_mesh((vertices, faces), weld_tol=(weld_tol if weld_tol > 0 else None))

        # Intento adicional de cerrar agujeros si está habilitado
        if do_fill_holes and not mesh.is_watertight:
            try:
                trimesh.repair.fill_holes(mesh)
                mesh.process()
            except Exception:
                pass

        # Estado de la malla
        st.write({
            "tipo_mesh": str(type(mesh)),
            "n_vertices": int(len(mesh.vertices)),
            "n_faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight)
        })

        # Volumen
        vol_text = "No disponible"
        try:
            vol_value = float(mesh.volume)
            vol_text = f"{vol_value:.6f}"
            if mesh.is_watertight:
                st.success(f"Volumen: {vol_text} (u³)")
            else:
                st.warning(f"Volumen (malla no watertight): {vol_text} (u³) — puede ser inexacto.")
        except Exception:
            st.error("No fue posible calcular el volumen. Revisa que la malla esté cerrada (watertight).")

        # Visualización 3D con Plotly
        st.info("Renderizando vista 3D…")
        x, y, z = mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2]
        i, j, k = mesh.faces[:, 0], mesh.faces[:, 1], mesh.faces[:, 2]

        data = [
            go.Mesh3d(
                x=x, y=y, z=z, i=i, j=j, k=k,
                opacity=float(mesh_opacity),
                color="lightskyblue",
                name="Malla"
            )
        ]
        if show_wireframe:
            # Wireframe básico usando segmentos de la malla
            edges = mesh.edges_unique
            ex = mesh.vertices[edges].reshape(-1, 2, 3)
            for seg in ex:
                data.append(
                    go.Scatter3d(
                        x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
                        mode="lines", line=dict(color="gray", width=1),
                        name="Wireframe", showlegend=False
                    )
                )

        fig = go.Figure(data=data)
        fig.update_layout(
            scene=dict(aspectmode="data"),
            margin=dict(l=0, r=0, t=30, b=0),
            title="Malla 3D"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Exportaciones
        c1, c2 = st.columns(2)
        with c1:
            try:
                stl_bytes = mesh.export(file_type="stl")
                st.download_button(
                    "Descargar STL",
                    data=stl_bytes,
                    file_name="malla.stl",
                    mime="application/octet-stream"
                )
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
            st.download_button(
                "Descargar reporte TXT",
                data=reporte.getvalue(),
                file_name="reporte.txt",
                mime="text/plain"
            )

    except Exception as e:
        import traceback
        st.error("Ocurrió un error procesando la malla. Detalle abajo o revisa los logs en Streamlit Cloud (Manage app → Logs).")
        st.code(traceback.format_exc())
        st.stop()
    finally:
        # Limpia el archivo temporal si existe
        try:
            if 'dxf_path' in locals() and os.path.exists(dxf_path):
                os.remove(dxf_path)
        except Exception:
            pass
else:
    st.info("Carga un archivo DXF para comenzar.")

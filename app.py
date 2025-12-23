# app.py
# -*- coding: utf-8 -*-
"""
DXF → Sólido 3D y Volumen (robusto)
- Lee DXF con ezdxf (3DFACE/POLYFACE/MESH via virtual_entities), triangula y
  descarta caras degeneradas.
- Repara malla, verifica watertight y calcula volumen (nativo y m³).
- Visualiza en 3D con Plotly.
- Exporta STL y Reporte TXT.

Autor: Sebastián Zúñiga Leyton - Ingeniero Civil de Minas
"""

import os, io, tempfile
from typing import Optional, Tuple, List

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import trimesh

# ---- dependencias opcionales / lectura de DXF
try:
    import ezdxf
    HAS_EZDXF = True
except Exception:
    HAS_EZDXF = False

# ========= Unidades =========
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
def _collect_faces_ezdxf(dxf_path:str, layer_regex:Optional[str]=None):
    """
    Devuelve listas (verts, faces) desde 3DFACE (y derivados) usando ezdxf.
    Aplica virtual_entities() para desanidar INSERT/BLOCK.
    - layer_regex: si se entrega, solo incluye entidades cuya capa haga match (regex).
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

    # índice de vertices deduplicados con tolerancia
    # hash por grid snapping pequeño para agrupar
    TOL = 1e-10  # en unidades DXF (muy pequeño)
    vmap = {}    # key -> index

    def add_vertex(v):
        k = (round(v.x/TOL), round(v.y/TOL), round(v.z/TOL))
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
        # usar coordenadas en WCS si existen
        try:
            ws = list(face.wcs_vertices())
        except Exception:
            ws = [face.dxf.vtx0, face.dxf.vtx1, face.dxf.vtx2, face.dxf.vtx3]
        # Muchos DXF ponen v2==v3 para triángulos
        pts = [p for p in ws if p is not None]
        # filtrar puntos repetidos exactos
        uniq = []
        for p in pts:
            if not uniq or (p != uniq[-1]):
                uniq.append(p)
        if len(uniq) < 3:
            return
        # Triángulo directo
        if len(uniq) == 3:
            i0=add_vertex(uniq[0]); i1=add_vertex(uniq[1]); i2=add_vertex(uniq[2])
            add_triangle(i0,i1,i2); return
        # Cuadrilátero -> 2 triángulos
        if len(uniq) >= 4:
            i0=add_vertex(uniq[0]); i1=add_vertex(uniq[1]); i2=add_vertex(uniq[2]); i3=add_vertex(uniq[3])
            add_triangle(i0,i1,i2)
            add_triangle(i0,i2,i3)

    # Recorremos entidades, expandiendo virtuales (POLYFACE/MESH → 3DFACE)
    for e in msp:
        try:
            layer = e.dxf.layer
        except Exception:
            layer = ""
        if not layer_ok(layer):
            continue

        processed = False
        # intentar virtual_entities si existe
        try:
            for ve in e.virtual_entities():
                if ve.dxftype() == "3DFACE":
                    handle_face3d(ve)
                    processed = True
        except Exception:
            pass

        if processed:
            continue

        # caso 3DFACE directo
        try:
            if e.dxftype() == "3DFACE":
                handle_face3d(e)
                continue
        except Exception:
            pass

        # Algunas MESH antiguas pueden no exponer virtual; intentar fallback específico
        try:
            if e.dxftype() in ("POLYFACE", "POLYLINE", "MESH"):
                # último recurso: explotar a 3DFACE usando auditor
                from ezdxf import explode
                tmp = list(explode.virtual_entities(e))
                for ve in tmp:
                    if ve.dxftype() == "3DFACE":
                        handle_face3d(ve)
                continue
        except Exception:
            pass
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)

def load_mesh_from_dxf_robust(dxf_path:str, thickness:Optional[float], layer_regex:Optional[str]=None)->trimesh.Trimesh:
    """
    Intenta: (A) parser ezdxf → malla; si falla, (B) trimesh.load(force='mesh').
    thickness solo se usa si no hay malla 3D y existen perfiles 2D (requiere shapely).
    """
    if HAS_EZDXF:
        try:
            V,F = _collect_faces_ezdxf(dxf_path, layer_regex=layer_regex)
            if V.size and F.size:
                mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
                if not mesh.is_empty:
                    return mesh
        except Exception as e:
            # Continuar al fallback
            st.info(f"Parser ezdxf falló, probando fallback trimesh: {e}")

    # ---- Fallback con trimesh ----
    geom = trimesh.load(dxf_path, force='mesh')
    if isinstance(geom, trimesh.Scene):
        meshes = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if meshes:
            return trimesh.util.concatenate(meshes)

        # ¿2D? → extruir si se pidió
        if thickness and thickness != 0:
            paths_2d=[g for g in geom.geometry.values() if hasattr(g,"polygons_full")]
            polygons=[]
            for p in paths_2d: polygons.extend(p.polygons_full)
            if polygons:
                from shapely.ops import unary_union
                merged = unary_union(polygons)
                return trimesh.creation.extrude_polygon(merged, height=float(thickness))
        raise ValueError("No se encontraron mallas 3D en el DXF.")
    if isinstance(geom, trimesh.Trimesh):
        return geom
    if hasattr(geom, "polygons_full") and thickness and thickness>0:
        from shapely.ops import unary_union
        merged = unary_union(list(geom.polygons_full))
        return trimesh.creation.extrude_polygon(merged, height=float(thickness))
    raise ValueError("DXF no reconocido como malla ni perfiles 2D.")

# ========= Repair / Volumen / Plot =========
def repair_mesh(mesh:trimesh.Trimesh, weld_tol:Optional[float])->trimesh.Trimesh:
    mesh.remove_duplicate_faces(); mesh.remove_degenerate_faces(); mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(mesh)
    try: trimesh.repair.fix_normals(mesh)
    except Exception: pass
    try: trimesh.repair.fill_holes(mesh)
    except Exception: pass
    if weld_tol and weld_tol>0:
        v = mesh.vertices.copy()
        q = max(float(weld_tol), 1e-12)
        vq = np.round(v/q)*q
        mesh.vertices = vq
        mesh.remove_duplicate_faces(); mesh.remove_unreferenced_vertices()
    parts = mesh.split(only_watertight=False)
    if len(parts)>1:
        parts = sorted(parts, key=lambda m: (m.bounding_box_oriented.volume if m.vertices.size else 0), reverse=True)
        mesh = parts[0]
    return mesh

def compute_volumes(mesh:trimesh.Trimesh, to_meters:float):
    wt = bool(mesh.is_watertight)
    vol_native = float(mesh.volume) if wt else 0.0
    vol_m3 = vol_native * (to_meters**3)
    hull_m3 = float(mesh.convex_hull.volume) * (to_meters**3)
    return vol_native, vol_m3, wt, hull_m3

def plot_trimesh(mesh:trimesh.Trimesh, title="Sólido DXF")->go.Figure:
    V,F=mesh.vertices, mesh.faces
    fig = go.Figure([go.Mesh3d(
        x=V[:,0], y=V[:,1], z=V[:,2], i=F[:,0], j=F[:,1], k=F[:,2],
        color="#4d88ff", opacity=1.0, flatshading=True,
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2)
    )])
    fig.update_layout(title=title, scene=dict(aspectmode="data"),
                      margin=dict(l=0,r=0,t=36,b=0))
    return fig

# ========= UI =========
st.set_page_config(page_title="DXF → Sólido y Volumen (robusto)", layout="wide")
st.title("DXF → Sólido 3D y Volumen")
st.subheader("Conversión robusta de DXF a sólido y cálculo de volumen - Herramienta creada por Sebastián Zúñiga Leyton")

with st.sidebar:
    st.header("Parámetros")
    assume_units = st.selectbox("Asumir unidades (si el DXF no trae $INSUNITS):",
                                ["(auto)","mm","cm","m","in","ft","unitless"], index=0)
    layer_regex = st.text_input("Filtro por capa (regex opcional)", value="")
    weld_tol = st.number_input("Tolerancia de soldado (unidades DXF)", min_value=0.0, value=0.0, step=0.0001, format="%.6f")
    thickness = st.number_input("Espesor extrusión (solo si DXF 2D)", min_value=0.0, value=0.0, step=0.1)
    simplify_for_view = st.checkbox("Simplificar solo para visualizar", value=True)
    target_faces = st.number_input("Caras objetivo visualización", min_value=1000, value=50000, step=1000)

uploaded = st.file_uploader("Sube tu DXF", type=["dxf"])
if not uploaded:
    st.info("📁 Sube un archivo DXF para comenzar.")
    st.stop()

with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = os.path.join(tmpdir, uploaded.name)
    with open(tmp_path, "wb") as f: f.write(uploaded.read())

    unit_name, to_meters = read_insunits(tmp_path, None if assume_units=="(auto)" else assume_units)
    st.write(f"**Unidades DXF:** `{unit_name}` (1 {unit_name} = {to_meters} m)")

    try:
        mesh = load_mesh_from_dxf_robust(
            tmp_path,
            thickness=(thickness if thickness>0 else None),
            layer_regex=(layer_regex or None)
        )
    except Exception as e:
        st.error(f"Error al cargar el DXF (parser robusto + fallback): {e}")
        st.stop()

    mesh = repair_mesh(mesh, weld_tol=(weld_tol if weld_tol>0 else None))

    vol_native, vol_m3, wt, hull_m3 = compute_volumes(mesh, to_meters)
    cubic = cubic_suffix(unit_name)

    colA,colB,colC,colD = st.columns(4)
    colA.metric("Vértices", f"{mesh.vertices.shape[0]:,}")
    colB.metric("Caras", f"{mesh.faces.shape[0]:,}")
    colC.metric("Watertight", "Sí" if wt else "No")
    colD.metric("Volumen (m³)" if wt else "Envolvente convexa (m³)",
                f"{(vol_m3 if wt else hull_m3):,.6f}")

    st.markdown("### Resultados")
    if wt:
        st.write(f"**Volumen (nativo):** `{vol_native:,.6f} {cubic}`")
        st.write(f"**Volumen (m³):** `{vol_m3:,.6f} m³`")
    else:
        st.warning("La malla no es cerrada. El volumen exacto no es confiable.")
        st.write(f"**Envolvente convexa (m³):** `{hull_m3:,.6f}`")

    # Visualización (opcionalmente simplificada)
    mesh_view = mesh.copy()
    if simplify_for_view and mesh_view.faces.shape[0] > target_faces:
        step = max(int(mesh_view.faces.shape[0]/target_faces), 1)
        mesh_view.update_faces(np.arange(0, mesh_view.faces.shape[0], step))
        mesh_view.remove_unreferenced_vertices()
    st.plotly_chart(plot_trimesh(mesh_view, title=os.path.basename(uploaded.name)), use_container_width=True)

    # Exports
    st.markdown("### Exportar")
    # STL
    stl_bytes = io.BytesIO()
    try:
        mesh.export(stl_bytes, file_type="stl"); stl_bytes.seek(0)
        st.download_button("⬇️ Descargar STL", stl_bytes, os.path.splitext(uploaded.name)[0]+".stl", "application/sla")
    except Exception as e:
        st.info(f"No se pudo generar STL: {e}")

    # Reporte
    rep = io.StringIO()
    rep.write(f"Archivo DXF: {uploaded.name}\n")
    rep.write(f"Unidades DXF: {unit_name} (1 {unit_name} = {to_meters} m)\n")
    rep.write(f"Vertices: {mesh.vertices.shape[0]}, Caras: {mesh.faces.shape[0]}\n")
    rep.write(f"Watertight: {wt}\n")
    if wt:
        rep.write(f"Volumen nativo: {vol_native:.9f} {cubic}\n")
        rep.write(f"Volumen m3: {vol_m3:.9f} m^3\n")
    else:
        rep.write("La malla no es cerrada (no-watertight)\n")
        rep.write(f"Convex hull m3: {hull_m3:.9f}\n")
    st.download_button("⬇️ Descargar Reporte TXT", rep.getvalue(),
                       os.path.splitext(uploaded.name)[0]+"_reporte_volumen.txt", "text/plain")
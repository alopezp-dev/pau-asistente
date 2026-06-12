
# app.py — PAU Asistente · Interfaz Streamlit
# Ejecutar con: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import re
import faiss
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="PAU Asistente",
    page_icon="🎓",
    layout="wide"
)

# ── Carga de recursos (cacheados para no recargar en cada consulta) ───────────
@st.cache_data
def cargar_datos():
    df = pd.read_csv("notas_corte_espana_2026.csv", sep=";", decimal=",", encoding="utf-8-sig")
    return df

@st.cache_resource
def cargar_modelo_e_indice(df):
    modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    textos = [
        f"{r.titulacion} en {r.universidad}, {r.provincia}, {r.comunidad_autonoma}. "
        f"Nota de corte {r.nota_actual:.3f}. Facultad: {r.facultad}."
        for _, r in df.iterrows()
    ]
    emb = modelo.encode(textos, batch_size=64, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(emb)
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)
    return modelo, idx, textos

# ── Lógica del sistema ────────────────────────────────────────────────────────
def buscar(consulta, nota, comunidad, top_k, df, modelo, idx):
    vec = modelo.encode([consulta], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vec)
    scores, indices = idx.search(vec, 150)
    candidatos = df.iloc[indices[0]].copy()
    candidatos["score"] = scores[0]
    if nota:
        candidatos = candidatos[candidatos["nota_actual"].isna() | (candidatos["nota_actual"] <= nota + 0.5)]
    if comunidad != "Todas":
        candidatos["score"] += (candidatos["comunidad_autonoma"] == comunidad).astype(float) * 0.1
    return candidatos.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🎓 PAU Asistente")
st.markdown("*Sistema experto de orientación universitaria para estudiantes de selectividad*")
st.divider()

df = cargar_datos()
modelo, idx, _ = cargar_modelo_e_indice(df)

col1, col2 = st.columns([2, 1])
with col1:
    consulta = st.text_input("💬 ¿Qué carrera buscas?",
                             placeholder="Ej: Quiero estudiar Medicina en Madrid")
with col2:
    nota = st.number_input("📊 Tu nota (0 = sin filtro)", 0.0, 14.0, 0.0, 0.001, format="%.3f")

col3, col4 = st.columns([2, 1])
with col3:
    comunidades = ["Todas"] + sorted(df["comunidad_autonoma"].dropna().unique().tolist())
    comunidad = st.selectbox("📍 Comunidad autónoma", comunidades)
with col4:
    top_k = st.slider("Nº de resultados", 3, 20, 8)

if st.button("🔍 Buscar carreras", type="primary") and consulta:
    with st.spinner("Buscando las mejores opciones para ti..."):
        nota_val = nota if nota > 0 else None
        resultados = buscar(consulta, nota_val, comunidad, top_k, df, modelo, idx)

    if resultados.empty:
        st.warning("No se encontraron resultados. Prueba a ampliar los criterios.")
    else:
        st.success(f"✅ {len(resultados)} carreras encontradas")
        for _, row in resultados.iterrows():
            nota_corte = f"{row['nota_actual']:.3f}" if pd.notna(row["nota_actual"]) else "N/D"
            accesible = ""
            if nota_val and pd.notna(row["nota_actual"]):
                diff = nota_val - row["nota_actual"]
                accesible = f" ✅ +{diff:.2f}" if diff >= 0 else f" ⚠️ {diff:.2f}"
            with st.expander(f"🎓 {row['titulacion']} · Nota: {nota_corte}{accesible}"):
                c1, c2 = st.columns(2)
                c1.markdown(f"**Universidad:** {row['universidad']}")
                c1.markdown(f"**Facultad:** {row['facultad']}")
                c2.markdown(f"**Provincia:** {row['provincia']}")
                c2.markdown(f"**Comunidad:** {row['comunidad_autonoma']}")
                if pd.notna(row.get("nota_anterior")):
                    st.markdown(f"*Nota anterior: {row['nota_anterior']:.3f}*")

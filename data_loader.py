import os
import pandas as pd
import io
import streamlit as st
from pathlib import Path

# leitura de arquivo em disco, cacheada
@st.cache_data
def read_csv_from_path(path):
    return pd.read_csv(path)

# leitura a partir de bytes (uploaded), cacheada
@st.cache_data
def read_csv_from_bytes(bts):
    return pd.read_csv(io.BytesIO(bts))

# função que apenas tenta localizar arquivo em disco (sem widgets)
def load_data_from_disk():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = [
        os.path.join(base_dir, "data", "raw", "Climate_Change_Real_Physics.csv"),
        os.path.join(base_dir, "src", "data", "raw", "Climate_Change_Real_Physics.csv"),
        os.path.join(base_dir, "data", "Climate_Change_Real_Physics.csv"),
        os.path.join(base_dir, "kaggleml", "data", "raw", "Climate_Change_Real_Physics.csv"),
    ]
    # debug: mostra no log quais caminhos são verificados
    for p in candidates:
        print("DEBUG: verificando:", p, "->", Path(p).exists())
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None  # sinaliza que não encontrou

# --- ALTERE APENAS ESTA LINHA ---
DEFAULT_PATH = Path("data/raw/Climate_Change_Real_Physics.csv")
# --------------------------------

@st.cache_data
def load_data():
    import os
    import pandas as pd
    import streamlit as st

    base_dir = os.path.dirname(os.path.abspath(__file__))  # src/
    project_root = os.path.abspath(os.path.join(base_dir, ".."))

    candidates = [
        os.path.join(project_root, "data", "raw", "Climate_Change_Real_Physics.csv"),
        os.path.join(project_root, "src", "data", "raw", "Climate_Change_Real_Physics.csv"),
        os.path.join(project_root, "data", "Climate_Change_Real_Physics.csv"),
        os.path.join(base_dir, "data", "raw", "Climate_Change_Real_Physics.csv"),
    ]

    for path in candidates:
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception as e:
                st.error(f"Falha ao ler {path}: {e}")
                return pd.DataFrame()

    st.error("❌ Arquivo não encontrado em nenhum dos caminhos esperados.")
    st.info("Caminhos verificados:\n" + "\n".join(candidates))

    uploaded = st.file_uploader("Envie o CSV Climate_Change_Real_Physics.csv", type="csv")
    if uploaded is not None:
        try:
            return pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Falha ao ler o arquivo enviado: {e}")
    return pd.DataFrame()

import os
from pathlib import Path
import streamlit as st

# DEBUG TEMPORÁRIO: mostrar informações de caminhos e existência do arquivo
st.write("DEBUG: cwd =", os.getcwd())
st.write("DEBUG: __file__ =", Path(__file__).resolve())
root = Path(__file__).resolve().parents[1]  # pasta do projeto (assumida)
st.write("DEBUG: projeto root =", root)
candidate = root / "data" / "raw" / "Climate_Change_Real_Physics.csv"
st.write("DEBUG: verificando:", candidate, "->", candidate.exists())

# lista arquivos top-level (limitar saída)
files = [p.relative_to(root) for p in root.rglob("*") if p.is_file()][:200]
st.write("DEBUG: arquivos no repo (até 200):", files)

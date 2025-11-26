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
    root = Path(__file__).resolve().parents[1]  # assume project root = parent of src/
    filename = "Climate_Change_Real_Physics.csv"

    # procura recursivamente no repositório pela primeira ocorrência do arquivo
    matches = list(root.rglob(filename))
    st.write("DEBUG: projeto root =", root)
    st.write("DEBUG: matches encontrados =", [str(p) for p in matches][:20])

    if matches:
        path = matches[0]
        st.write("DEBUG: carregando arquivo encontrado em:", path)
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.error(f"Falha ao ler {path}: {e}")
            return None

    # fallback: caminhos comuns verificados anteriormente (mantém compatibilidade)
    candidates = [
        root / "data" / "raw" / filename,
        root / "kaggleml" / "data" / "raw" / filename,
        root / "kaggleml" / filename,
    ]
    for p in candidates:
        st.write("DEBUG: verificando:", p, "->", p.exists())
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception as e:
                st.error(f"Falha ao ler {p}: {e}")
                return None

    return None  # sinaliza que não encontrou

# --- ALTERE APENAS ESTA LINHA ---
DEFAULT_PATH = Path("data/raw/Climate_Change_Real_Physics.csv")
# --------------------------------

@st.cache_data
def load_data(path: str | Path | None = None) -> pd.DataFrame:
    root = Path(__file__).resolve().parents[1]  # raiz do projeto
    filename = "Climate_Change_Real_Physics.csv"

    # 1) se path fornecido usar ele
    if path:
        p = Path(path)
        if p.exists():
            return pd.read_csv(p)

    # 2) procurar recursivamente no repo
    matches = list(root.rglob(filename))
    st.write("DEBUG: projeto root =", root)
    st.write("DEBUG: matches encontrados =", [str(p) for p in matches][:20])
    if matches:
        return pd.read_csv(matches[0])

    # 3) caminhos padrão (fallback)
    candidates = [
        root / "data" / "raw" / filename,
        root / "kaggleml" / "data" / "raw" / filename,
        root / filename,
    ]
    for c in candidates:
        st.write("DEBUG: verificando:", c, "->", c.exists())
        if c.exists():
            return pd.read_csv(c)

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

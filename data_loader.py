import os
import pandas as pd
import streamlit as st
from pathlib import Path

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

import os
import pandas as pd
import streamlit as st
from pathlib import Path

# --- ALTERE APENAS ESTA LINHA ---
DEFAULT_PATH = Path("data/raw/Climate_Change_Real_Physics.csv")
# --------------------------------

@st.cache_data
def load_data(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """
    Carrega os dados CSV e guarda na memória (cache) para o site ficar rápido.
    """
    path = Path(path)
    
    # Verifica se o arquivo existe
    if not path.exists():
        st.error(f"❌ Arquivo não encontrado em: {path}")
        # Retorna um DataFrame vazio para não quebrar o app
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(path)
        # Limpeza básica: remove duplicatas se houver
        df = df.drop_duplicates()
        return df
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return pd.DataFrame()

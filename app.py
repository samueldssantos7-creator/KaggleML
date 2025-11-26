import sys
import os
# Adiciona o diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False
    import streamlit as st
    st.warning("Biblioteca 'shap' não disponível — funcionalidades de interpretabilidade desativadas.")
import matplotlib.pyplot as plt
import io
# usamos Matplotlib para SHAP com render sob demanda (fallback a imagem)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # <--- MUDANÇA 1: Modelo Matemático
from sklearn.ensemble import RandomForestRegressor
from data_loader import load_data
import numpy as np

# Configuração da página
st.set_page_config(page_title="Climate Change Analytics", layout="wide")

# --- DEFINIÇÃO DE CORES PERSONALIZADAS ---
COLOR_MAP = {
    'Africa':        '#2ca02c',
    'South America': '#FFD700',
    'North America': '#808080',
    'Europe':        '#00008B',
    'Antarctica':    '#87CEEB',
    'Oceania':       '#006400',
    'Asia':          '#4169E1'
}

def main():
    st.title("🌍 Global Climate Change Analytics")
    st.markdown("**Dashboard Interativo.** Utilize os filtros laterais para segmentar a análise.")

    # 1. Carregamento
    df_raw = load_data()

    if df_raw.empty:
        st.warning("Nenhum dado carregado.")
        return

    # ==========================================
    # --- FILTROS GLOBAIS (SIDEBAR) ---
    # ==========================================
    st.sidebar.header("🔍 Filtros de Dados")
    df = df_raw.copy()

    # Filtros (Continente, País, Ano)
    cont_cols = [c for c in df.columns if 'continent' in c.lower()]
    selected_continent_col = cont_cols[0] if cont_cols else None
    if selected_continent_col:
        continents = sorted(df[selected_continent_col].unique().astype(str))
        sel_continents = st.sidebar.multiselect("🌎 Filtrar por Continente", continents, default=continents)
        if sel_continents: df = df[df[selected_continent_col].isin(sel_continents)]

    country_cols = [c for c in df.columns if 'country' in c.lower() or 'location' in c.lower()]
    if selected_continent_col and selected_continent_col in country_cols: country_cols.remove(selected_continent_col)
    selected_country_col = country_cols[0] if country_cols else None
    if selected_country_col:
        countries = sorted(df[selected_country_col].unique().astype(str))
        sel_countries = st.sidebar.multiselect("🏳️ Filtrar por País/Local", countries, default=countries[:5] if len(countries) > 5 else countries)
        if sel_countries: df = df[df[selected_country_col].isin(sel_countries)]

    year_col = 'Year'
    if year_col in df.columns:
        df = df.dropna(subset=[year_col])
        df[year_col] = df[year_col].astype(int)
        min_year, max_year = int(df[year_col].min()), int(df[year_col].max())
        if min_year != max_year:
            selected_years = st.sidebar.slider("📅 Filtrar por Período", min_year, max_year, (min_year, max_year))
            df = df[(df[year_col] >= selected_years[0]) & (df[year_col] <= selected_years[1])]

    st.sidebar.markdown("---")
    st.sidebar.info(f"**Registros encontrados:** {len(df)}")
    if df.empty:
        st.error("⚠️ Nenhum dado encontrado.")
        return

    # ==========================================
    # --- ABAS DO DASHBOARD ---
    # ==========================================
    tab1, tab2, tab3 = st.tabs(["📊 Análise Interativa", "🔗 Correlações", "🤖 Modelagem & Previsão"])

    # --- ABA 1: EDA ---
    with tab1:
        st.subheader("Análise Visual")
        col1, col2 = st.columns(2)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        color_col = selected_continent_col if selected_continent_col else selected_country_col

        with col1:
            st.markdown("### 🌡️ Temperatura Média por Continente")
            if selected_continent_col and 'Avg_Temperature(°C)' in df.columns:
                df_avg = df.groupby(selected_continent_col)['Avg_Temperature(°C)'].mean().reset_index().sort_values(by='Avg_Temperature(°C)')
                fig_bar = px.bar(df_avg, x=selected_continent_col, y='Avg_Temperature(°C)', color=selected_continent_col, color_discrete_map=COLOR_MAP, text_auto='.1f', title="Média Geral de Temperatura")
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.markdown("### 📈 Evolução Temporal")
            if year_col in df.columns:
                y_col_line = st.selectbox("Indicador de Evolução", numeric_cols, index=min(1, len(numeric_cols)-1))
                group_cols = [year_col]
                if color_col: group_cols.append(color_col)
                df_grouped = df.groupby(group_cols)[y_col_line].mean().reset_index()
                fig_line = px.line(df_grouped, x=year_col, y=y_col_line, color=color_col, color_discrete_map=COLOR_MAP, title=f"Evolução de {y_col_line}")
                st.plotly_chart(fig_line, use_container_width=True)

    # --- ABA 2: CORRELAÇÕES ---
    with tab2:
        st.subheader("Matriz de Correlação")
        df_numeric = df.select_dtypes(include=['number'])
        if year_col in df_numeric.columns: df_numeric = df_numeric.drop(columns=[year_col]) 
        if not df_numeric.empty and df_numeric.shape[1] > 1:
            fig_corr = px.imshow(df_numeric.corr(), text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_corr, use_container_width=True)

    # --- ABA 3: APENAS SHAP (Matplotlib seguro) ---
    with tab3:
        st.subheader("Como você pensa? (SHAP — Matplotlib seguro)")

        # Treinar / Carregar modelo rápido (necessário para mostrar o gráfico Matplotlib)
        features = ['Avg_Temperature(°C)', 'CO2_Emissions(Mt)', 'Sea_Level_Rise(mm)']
        if all(f in df.columns for f in features) and 'Climate_Risk_Index' in df.columns:
            if st.button("Treinar Regressão Linear", key="train_lr"):
                X = df[features].dropna()
                y = df.loc[X.index, 'Climate_Risk_Index']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression().fit(X_train, y_train)
                r2 = model.score(X_test, y_test)
                # salva na sessão para a seção SHAP usar
                st.session_state['model'] = model
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test.reset_index(drop=True).tolist()
                st.session_state['feats'] = features
                st.session_state['r2'] = r2
                st.success(f"Modelo treinado — R² (validação): {r2:.2%}")

            # botão para gerar SHAP Matplotlib após o treino
            shap_container = st.container()
            if 'model' in st.session_state:
                # filtros para o SHAP: escolher features a mostrar
                selected_feats = st.multiselect(
                    "Selecione as features para exibir no gráfico SHAP",
                    options=features,
                    default=features
                )

                # número fixo seguro de amostras (até 100)
                X_test_full = st.session_state.get('X_test')
                sample_count = int(min(100, X_test_full.shape[0])) if X_test_full is not None else 50

                if st.button("Gerar SHAP (Matplotlib)", key="gen_shap_mpl"):
                    try:
                        plt.close('all')
                        model = st.session_state['model']
                        X_train = st.session_state.get('X_train')
                        X_test = X_test_full
                        if X_test is None:
                            st.warning("X_test ausente — treine o modelo antes.")
                        else:
                            # calcula shap values para o conjunto de teste completo
                            masker = shap.maskers.Independent(X_train if X_train is not None else X_test, max_samples=100)
                            explainer = shap.LinearExplainer(model, masker)
                            shap_values = explainer.shap_values(X_test)

                            # normaliza formato de array: (n_samples, n_features)
                            sv = np.array(shap_values)
                            if sv.shape[0] == len(features) and sv.shape[1] == X_test.shape[0]:
                                sv = sv.T

                            # escolhe amostras aleatórias para mostrar
                            rng = np.random.RandomState(42)
                            idxs = rng.choice(np.arange(sv.shape[0]), size=min(sample_count, sv.shape[0]), replace=False)
                            X_plot = X_test.reset_index(drop=True).iloc[idxs].copy()
                            sv_plot = sv[idxs, :]

                            # filtra colunas (features) selecionadas
                            feat_idx = [features.index(f) for f in selected_feats]
                            sv_plot = sv_plot[:, feat_idx]
                            feats_plot = selected_feats
                            X_plot = X_plot[feats_plot]

                            # desenha com matplotlib (summary_plot)
                            plt.figure(figsize=(10, 4))
                            shap.summary_plot(sv_plot, X_plot, feature_names=feats_plot, show=False)
                            fig = plt.gcf()
                            with shap_container:
                                st.pyplot(fig, clear_figure=True)
                            plt.close(fig)
                    except Exception as e:
                        with shap_container:
                            st.error(f"Erro ao gerar SHAP: {e}")

if __name__ == "__main__":
    main()

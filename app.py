import sys
import os
# Adiciona o diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_squared_error

# IMPORTS ADICIONADOS
from data_loader import load_data_from_disk, read_csv_from_bytes

# Configuração da página
st.set_page_config(page_title="Climate Change Analytics", layout="wide")

COLOR_MAP = {
    'Africa':        '#2ca02c',
    'South America': '#FFD700',
    'North America': '#808080',
    'Europe':        '#00008B',
    'Antarctica':    '#87CEEB',
    'Oceania':       '#006400',
    'Asia':          '#4169E1'
}

def train_model(model_name, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if model_name == "Linear Regression":
        model = LinearRegression().fit(X_train, y_train)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    # substituir uso de `squared=False` por sqrt do MSE (compatível com todas as versões do sklearn)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return model, X_train, X_test, y_train, y_test, preds, r2, rmse

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False
    import streamlit as st
    st.warning("Biblioteca 'shap' não disponível — funcionalidades de interpretabilidade desativadas.")

def show_permutation_importance(model, X, y, feature_names, n_repeats=10):
    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1)
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': r.importances_mean,
        'importance_std': r.importances_std
    }).sort_values('importance_mean', ascending=False)
    fig = px.bar(fi_df, x='feature', y='importance_mean', error_y='importance_std', title='Permutation Importance')
    st.plotly_chart(fig, use_container_width=True)
    # se houver figura matplotlib antes/ depois, garantir fechamento
    try:
        plt.close('all')
    except Exception:
        pass
    return fi_df

def show_partial_dependence(model, X, features):
    try:
        fig, ax = plt.subplots(figsize=(6,4))
        PartialDependenceDisplay.from_estimator(model, X, features, ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Falha ao gerar Partial Dependence: {e}")
    finally:
        plt.close('all')

def main():
    st.title("🌍 Global Climate Change Analytics")
    # primeiro, tenta carregar do disco
    df_raw = load_data_from_disk()
    if df_raw is None:
        st.warning("❌ Arquivo não encontrado em disco. Faça upload do CSV abaixo.")
        uploaded = st.file_uploader("Envie o CSV Climate_Change_Real_Physics.csv", type="csv")
        if uploaded is not None:
            df_raw = read_csv_from_bytes(uploaded.getvalue())
    if df_raw is None or df_raw.empty:
        st.error("Nenhum dado carregado.")
        return

    st.markdown("**Dashboard Interativo.** Utilize os filtros laterais para segmentar a análise.")

    st.sidebar.header("🔍 Filtros de Dados")
    df = df_raw.copy()

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

    tab1, tab2, tab3 = st.tabs(["📊 Análise Interativa", "🔗 Correlações", "🤖 Modelagem & Previsão"])

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
            if year_col in df.columns and len(numeric_cols)>0:
                y_col_line = st.selectbox("Indicador de Evolução", numeric_cols, index=min(1, len(numeric_cols)-1))
                group_cols = [year_col]
                if color_col: group_cols.append(color_col)
                df_grouped = df.groupby(group_cols)[y_col_line].mean().reset_index()
                fig_line = px.line(df_grouped, x=year_col, y=y_col_line, color=color_col, color_discrete_map=COLOR_MAP, title=f"Evolução de {y_col_line}")
                st.plotly_chart(fig_line, use_container_width=True)

    with tab2:
        st.subheader("Matriz de Correlação")
        df_numeric = df.select_dtypes(include=['number'])
        if year_col in df_numeric.columns: df_numeric = df_numeric.drop(columns=[year_col])
        if not df_numeric.empty and df_numeric.shape[1] > 1:
            fig_corr = px.imshow(df_numeric.corr(), text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_corr, use_container_width=True)

    with tab3:
        st.subheader("Modelagem & Previsão")
        st.markdown("Treine modelos e visualize métricas, predições e importâncias de features.")

        # features & target defaults
        default_features = ['Avg_Temperature(°C)', 'CO2_Emissions(Mt)', 'Sea_Level_Rise(mm)']
        target_col = 'Climate_Risk_Index'

        available_feats = [c for c in default_features if c in df.columns]
        if not available_feats or target_col not in df.columns:
            st.info("Colunas para modelagem ausentes. Verifique se as colunas de features e target existem no dataset.")
        else:
            model_choice = st.radio("Escolha o modelo", ("Linear Regression", "Random Forest"))
            feats_selected = st.multiselect("Features", options=available_feats, default=available_feats)
            if not feats_selected:
                st.warning("Selecione ao menos 1 feature.")
            else:
                X = df[feats_selected].dropna()
                y = df.loc[X.index, target_col]

                colA, colB = st.columns([1,2])
                with colA:
                    if st.button("Treinar Modelo"):
                        model, X_train, X_test, y_train, y_test, preds, r2, rmse = train_model(model_choice, X, y)
                        st.session_state['model'] = model
                        st.session_state['X_test'] = X_test
                        st.session_state['y_test'] = y_test
                        st.session_state['preds'] = preds
                        st.session_state['feats'] = feats_selected
                        st.session_state['r2'] = r2
                        st.session_state['rmse'] = rmse
                        st.success(f"Treinado: {model_choice} — R²: {r2:.3f} — RMSE: {rmse:.3f}")

                with colB:
                    if 'model' in st.session_state:
                        st.markdown("### Métricas")
                        st.write(f"Modelo: **{model_choice}**")
                        st.write(f"R²: **{st.session_state['r2']:.3f}**")
                        st.write(f"RMSE: **{st.session_state['rmse']:.3f}**")

                        # plot preds vs actual
                        df_pred = pd.DataFrame({
                            'y_true': st.session_state['y_test'].reset_index(drop=True),
                            'y_pred': st.session_state['preds']
                        })
                        # scatter com linha identidade e linha de ajuste
                        x = df_pred['y_true'].to_numpy()
                        y = df_pred['y_pred'].to_numpy()
                        # métricas
                        r2 = r2_score(x, y)
                        rmse = np.sqrt(mean_squared_error(x, y))
                        st.write(f"R²: {r2:.3f} — RMSE: {rmse:.3f}")

                        fig = px.scatter(df_pred, x='y_true', y='y_pred', title="Predições vs Observado")
                        # identidade
                        fig.add_trace(go.Scatter(x=[x.min(), x.max()], y=[x.min(), x.max()],
                                                 mode='lines', name='y = x', line=dict(color='black', dash='dash')))
                        # linha de ajuste (opcional se já tiver)
                        m, b = np.polyfit(x, y, 1)
                        x_line = np.linspace(x.min(), x.max(), 100)
                        fig.add_trace(go.Scatter(x=x_line, y=m*x_line+b, mode='lines', name=f'Fit: y={m:.3f}x+{b:.3f}', line=dict(color='red')))
                        st.plotly_chart(fig, use_container_width=True)

                        # resíduos
                        res = x - y
                        fig2 = px.scatter(x=y, y=res, labels={'x':'y_pred','y':'resíduo (y_true - y_pred)'}, title="Resíduos vs y_pred")
                        fig2.add_hline(y=0, line_dash="dash", line_color="black")
                        st.plotly_chart(fig2, use_container_width=True)
                        # histograma de resíduos
                        st.plotly_chart(px.histogram(pd.DataFrame({'res':res}), x='res', nbins=40, title="Histograma dos resíduos"), use_container_width=True)

                        # feature importances for RF
                        if model_choice == "Random Forest" and hasattr(st.session_state['model'], "feature_importances_"):
                            importances = st.session_state['model'].feature_importances_
                            fi_df = pd.DataFrame({'feature': st.session_state['feats'], 'importance': importances})
                            fi_df = fi_df.sort_values('importance', ascending=False)
                            fig_fi = px.bar(fi_df, x='feature', y='importance', title="Feature Importances (Random Forest)")
                            st.plotly_chart(fig_fi, use_container_width=True)

                        # Permutation Importance (não depende de SHAP)
                        if model_choice == "Random Forest" and 'model' in st.session_state:
                            st.markdown("### Importância das Features (Permutation)")
                            fi_df_perm = show_permutation_importance(
                                st.session_state['model'],
                                st.session_state['X_test'],
                                st.session_state['y_test'],
                                st.session_state['feats']
                            )
                            st.write(fi_df_perm)

                        # Partial Dependence Plots (usar session_state)
                        if 'model' in st.session_state:
                            st.markdown("### Dependência Parcial")
                            num_cols_to_plot = min(len(st.session_state['feats']), 3)
                            selected_partial_feats = st.multiselect(
                                "Selecione as features para Dependência Parcial",
                                st.session_state['feats'],
                                max_selections=num_cols_to_plot
                            )
                            if selected_partial_feats:
                                show_partial_dependence(
                                    st.session_state['model'],
                                    st.session_state['X_test'],
                                    selected_partial_feats
                                )

                        # permitir download de predições
                        csv = df_pred.to_csv(index=False).encode('utf-8')
                        st.download_button("Download predições (CSV)", data=csv, file_name="predictions.csv", mime="text/csv")

if __name__ == "__main__":
    main()

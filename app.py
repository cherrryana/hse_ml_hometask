import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import r2_score, mean_squared_error
from preprocessing import preprocess_data, prepare_features, align_features
from matplotlib.patches import Patch

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

st.set_page_config(
    page_title="Прогноз цен на автомобили", 
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    with open('models/best_model_car_prices.pkl', 'rb') as f:
        data = pickle.load(f)
    return data if isinstance(data, dict) else {'model': data}

@st.cache_data
def load_data():
    return pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')

def show_eda():
    st.header("EDA")
    
    df = load_data()
        
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Количество записей", len(df))
    with col2:
        st.metric("Количество признаков", len(df.columns))
    with col3:
        st.metric("Пропуски", df.isnull().sum().sum())
    
    chart_type = st.selectbox(
        "Выберите тип графика", 
        [
            "Распределение цен",
            "Зависимость цены от года",
            "Корреляционная матрица"
        ]
    )
    
    if chart_type == "Распределение цен":
        fig, ax = plt.subplots(figsize=(6, 4))
        prices = df['selling_price'][df['selling_price'] > 0]
        ax.hist(prices, bins=60, edgecolor='white', linewidth=1.2, alpha=0.85)
        ax.set_xlabel('Цена (рупии)', fontweight='bold')
        ax.set_ylabel('Количество автомобилей', fontweight='bold')
        ax.set_title('Распределение цен на автомобили', fontweight='bold')
        ax.ticklabel_format(style='plain', axis='x', useOffset=False)
        plt.tight_layout()
        st.pyplot(fig)
        
    elif chart_type == "Корреляционная матрица":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr = df[numeric_cols].corr().dropna(axis=0, how='all').dropna(axis=1, how='all')
        size = max(8, len(corr) * 0.8)
        fig, ax = plt.subplots(figsize=(size, size))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0, 
                    square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
        ax.set_title('Корреляционная матрица признаков', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
            
    elif chart_type == "Зависимость цены от года":
        fig, ax = plt.subplots(figsize=(6, 4))
        scatter = ax.scatter(df['year'], df['selling_price'], alpha=0.6, s=50, 
                            c=df['year'], cmap='plasma', edgecolors='white', linewidth=0.5)
        ax.set_xlabel('Год выпуска', fontweight='bold')
        ax.set_ylabel('Цена (рупии)', fontweight='bold')
        ax.set_title('Зависимость цены от года выпуска', fontweight='bold')
        ax.ticklabel_format(style='plain', axis='y', useOffset=False)
        plt.colorbar(scatter, ax=ax).set_label('Год выпуска', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)


def show_prediction():
    st.header("Предсказание цены")
    
    data = load_model()
    model = data['model']
    scaler = data.get('scaler')
    ohe = data.get('ohe')
    
    input_method = st.selectbox("Способ ввода", ["Ручной ввод", "CSV"])
    
    if input_method == "Ручной ввод":
        col1, col2 = st.columns(2)
        with col1:
            year = st.number_input("Год выпуска", 1985, 2025, 2015)
            km_driven = st.number_input("Пробег (км)", 0, 1000000, 5000)
            mileage = st.number_input("Расход (kmpl)", 0.0, 50.0, 20.0)
            engine = st.number_input("Объем (CC)", 0, 5000, 1200)
            max_power = st.number_input("Мощность (bhp)", 0.0, 500.0, 80.0)
            seats = st.number_input("Места", 2, 15, 5)
        with col2:
            torque = st.text_input("Крутящий момент", "150 kgm @ 3500 rpm")
            fuel = st.selectbox("Топливо", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
            transmission = st.selectbox("Трансмиссия", ["Manual", "Automatic"])
            seller_type = st.selectbox("Тип продавца", ["Individual", "Dealer", "Trustmark Dealer"])
            owner = st.selectbox("Владелец", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
            car_name = st.text_input("Название автомобиля (например: Maruti Swift Dzire VDI)", 
                                    value="Maruti Swift Dzire VDI")
        
        if st.button("Предсказать"):
            try:
                input_data = pd.DataFrame({
                    'name': [car_name],
                    'year': [year],
                    'km_driven': [km_driven],
                    'mileage': [mileage],
                    'engine': [engine],
                    'max_power': [max_power],
                    'torque': [torque],
                    'seats': [seats],
                    'fuel': [fuel],
                    'transmission': [transmission],
                    'seller_type': [seller_type],
                    'owner': [owner]
                })
                
                df_processed = preprocess_data(input_data)
                X_processed = prepare_features(df_processed, scaler, ohe)
                X_processed = align_features(X_processed, model)
                
                if X_processed.shape[1] != len(model.coef_):
                    st.error(f"Несовпадение количества признаков: ожидалось {len(model.coef_)}, получено {X_processed.shape[1]}")
                    return
                
                prediction = model.predict(X_processed)[0]
                
                if prediction < 0:
                    st.warning(f"Предсказание отрицательное: {prediction:,.0f} рупий")
                    st.warning("Возможная причина: входные данные сильно отличаются от обучающих")
                
                st.metric("Предсказанная цена", f"{prediction:,.0f} рупий")
                
            except Exception as e:
                st.error(f"Ошибка при предсказании: {e}")
            
    else:
        uploaded_file = st.file_uploader("CSV файл", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            if st.button("Предсказать"):
                try:
                    df_processed = preprocess_data(df.copy())
                    
                    X_processed = prepare_features(df_processed, scaler, ohe)
                    X_processed = align_features(X_processed, model)
                    
                    if X_processed.shape[1] != len(model.coef_):
                        st.error(f"Несовпадение количества признаков: ожидалось {len(model.coef_)}, получено {X_processed.shape[1]}")
                        return
                    
                    predictions = model.predict(X_processed)
                    df['predicted_price'] = predictions
                    
                    negative_count = (predictions < 0).sum()
                    if negative_count > 0:
                        st.warning(f"Получено {negative_count} отрицательных предсказаний из {len(df)}")
                        st.warning("Возможная причина: входные данные сильно отличаются от обучающих")
                    
                    display_cols = [
                        'name',
                        'selling_price',
                        'predicted_price'
                    ]
                    display_cols = [c for c in display_cols if c]
                    
                    st.dataframe(df[display_cols].head(20))
                    
                    if negative_count > 0:
                        negative_df = df[df['predicted_price'] < 0][display_cols]
                        st.warning(f"Записи с отрицательными предсказаниями ({len(negative_df)}):")
                        st.dataframe(negative_df)
                    
                    if 'selling_price' in df.columns:
                        st.subheader("Сравнение с реальными ценами")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("R² score", f"{r2_score(df['selling_price'], df['predicted_price']):.4f}")
                        with col2:
                            st.metric("MSE", f"{mean_squared_error(df['selling_price'], df['predicted_price']):,.0f}")
                        
                except Exception as e:
                    st.error(f"Ошибка при предсказании: {e}")

def show_model_weights():
    st.header("Веса модели")
    
    data = load_model()
    model = data['model']
    
    feature_names = list(model.feature_names_in_)
    weights_df = pd.DataFrame({
        'feature': feature_names,
        'weight': model.coef_,
        'weight_abs': np.abs(model.coef_)
    }).sort_values('weight_abs', ascending=False)
    
    st.dataframe(weights_df.head(15), use_container_width=True)
    
    top = weights_df.head(15)
    weights_to_plot = np.clip(top['weight'].values, -1e6, 1e6)
    max_abs = top['weight_abs'].max()
    
    colors = [plt.cm.Reds(0.5 + min(abs(w) / max_abs, 1) * 0.5) if w < 0 
              else plt.cm.Greens(0.5 + min(w / max_abs, 1) * 0.5) 
              for w in weights_to_plot]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(top)), weights_to_plot, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    
    for i, val in enumerate(weights_to_plot):
        offset = abs(val) * 0.05
        x_pos = val + offset if val >= 0 else val - offset
        ax.text(x_pos, i, f'{val:.0f}', va='center', fontweight='bold', fontsize=9)
    
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['feature'])
    ax.set_xlabel('Вес признака', fontweight='bold')
    ax.set_title('Топ-15 наиболее важных признаков модели', fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax.spines[['top', 'right']].set_visible(False)
    
    ax.legend(handles=[
        Patch(facecolor=plt.cm.Greens(0.7), label='Положительное влияние'),
        Patch(facecolor=plt.cm.Reds(0.7), label='Отрицательное влияние')
    ], loc='upper right', framealpha=0.9, fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    st.pyplot(fig, use_container_width=False)


def main():
    st.title("Прогноз цен на автомобили")
    
    page = st.sidebar.selectbox("Выберите раздел", 
                                ["EDA", "Prediction", "Weights"])
    
    if page == "EDA":
        show_eda()
    elif page == "Prediction":
        show_prediction()
    elif page == "Weights":
        show_model_weights()
        

if __name__ == "__main__":
    main()

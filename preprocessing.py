import pandas as pd
import numpy as np
import re
import pickle

TORQUE_CONVERSION_CONST = 9.80665

MODELS = [
    "i20", "City", "Spark", "Scorpio", "Figo Aspire", "Swift Dzire", "Amaze",
    "Innova", "Terrano", "Grand i10", "SX4", "Cruze", "Bolt", "Linea", "Manza",
    "Wagon R", "Safari", "Alto", "800", "Ertiga", "Baleno", "Celerio"
]

def extract_segment(row):
    name = str(row.get('name', '')).lower()
    engine = row.get('engine', 0)
    max_power = row.get('max_power', 0)
    seats = row.get('seats', 5)
    
    suv_keywords = ['suv', 'scorpio', 'safari', 'terrano', 'fortuner', 'endeavour', 'xuv', 'duster']
    if any(kw in name for kw in suv_keywords):
        return 'SUV'
    
    if seats > 7:
        return 'MPV'
    
    if max_power > 150 and engine > 2000:
        return 'Luxury'
    
    if engine > 2000 or max_power > 120:
        return 'Full-size'
    
    if engine < 1000 and max_power < 60:
        return 'Compact'
    
    if engine >= 1000 and engine <= 2000:
        return 'Mid-size'
    
    return 'Standard'

DIESEL_KEYWORDS = ['crdi', 'idtec', 'dicor', 'm2di', 'diesel', 'zdi', 'ddis']
CNG_KEYWORDS = ['cng']
PETROL_KEYWORDS = ['vtec', 'revotron', 'kappa', 'safire']
LPG_KEYWORDS = ['LPG']
ELECTRIC_KEYWORDS = ['Electric', 'EV', 'Hybrid', 'HEV', 'PHEV']

NUMERIC_FEATURES = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'seats', 'max_torque_rpm']
CATEGORICAL_FEATURES = ['brand', 'model', 'name_engine_type', 'segment', 'seats', 'fuel', 'transmission', 
                        'has_airbag', 'is_bs4', 'has_abs', 'is_luxury']


def extract_brand(sample):
    if pd.isna(sample) or not str(sample).split():
        return 'unk'
    return str(sample).split()[0]


def extract_model(sample):
    if not pd.isna(sample):
        name = str(sample).lower()
        for m in MODELS:
            if m.lower() in name:
                return m
    return 'unk'


def extract_engine_type(sample, fuel):
    if pd.isna(sample):
        return 'unk'
    
    name_str = str(sample).lower()
    
    for keywords, engine_type in [(DIESEL_KEYWORDS, 'Diesel'), (PETROL_KEYWORDS, 'Petrol'),
                                  (CNG_KEYWORDS, 'CNG'), (LPG_KEYWORDS, 'LPG'),
                                  (ELECTRIC_KEYWORDS, 'Electric')]:
        if any(kw.lower() in name_str for kw in keywords):
            return engine_type
    
    if fuel is not None and not pd.isna(fuel):
        fuel_str = str(fuel).lower()
        for fuel_type in ['diesel', 'petrol', 'cng', 'lpg']:
            if fuel_type in fuel_str:
                return fuel_type.capitalize()
    
    return 'unk'

def extract_categorical_features(df):
    df = df.copy()
    
    if 'name' in df.columns:
        df['brand'] = df['name'].apply(extract_brand)
        df['model'] = df['name'].apply(extract_model)
        df['name_engine_type'] = df.apply(
            lambda row: extract_engine_type(
                row['name'], 
                row.get('fuel', None)
            ), 
            axis=1
        )
        df['segment'] = df.apply(extract_segment, axis=1)
        df['has_airbag'] = df['name'].str.lower().str.contains('airbag|asta|vxi|zxi|sx', case=False, na=False)
        df['has_abs'] = df['name'].str.contains(r'\bABS\b', case=False, na=False)
        df['is_bs4'] = df['name'].str.lower().str.contains('bs ?iv|bsiv', case=False, na=False)
        df['is_luxury'] = df['name'].str.lower().str.contains('luxury|premium|top|vxi|zxi|sx|vx|zx|vdi|vvt', case=False, na=False)
    else:
        df['brand'] = 'unk'
        df['model'] = 'unk'
        df['name_engine_type'] = df.get('fuel', 'unk')
        df['segment'] = df.apply(extract_segment, axis=1)
        df['has_airbag'] = False
        df['has_abs'] = False
        df['is_bs4'] = False
        df['is_luxury'] = False
    
    return df


def preprocess_numeric_features(df):
    df = df.copy()
    
    if 'mileage' in df.columns:
        if df['mileage'].dtype == 'object':
            df['mileage'] = df['mileage'].str.replace('kmpl', '').str.replace('km/kg', '').str.strip()
            df['mileage'] = pd.to_numeric(df['mileage'])
    
    if 'engine' in df.columns:
        if df['engine'].dtype == 'object':
            df['engine'] = df['engine'].str.replace('CC', '').str.strip()
            df['engine'] = pd.to_numeric(df['engine'])
    
    if 'max_power' in df.columns:
        if df['max_power'].dtype == 'object':
            df['max_power'] = df['max_power'].str.replace('bhp', '').str.strip()
            df['max_power'] = pd.to_numeric(df['max_power'])
    
    if 'torque' in df.columns and df['torque'].dtype == 'object':
        def process_torque(value):
            if pd.isna(value):
                return pd.Series([np.nan, np.nan])

            numbers = list(map(float, re.findall(r'\d+\.?\d*', str(value))))
            if not numbers:
                return pd.Series([np.nan, np.nan])

            torque = numbers[0]
            max_torque_rpm = int(numbers[-1]) if len(numbers) > 1 else np.nan
            if 'kgm' in str(value).lower():
                torque *= TORQUE_CONVERSION_CONST

            return pd.Series([torque, max_torque_rpm])
        
        torque_data = df['torque'].apply(process_torque)
        df['torque'] = torque_data[0]
        df['max_torque_rpm'] = torque_data[1]
    
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            median_val = df[col].median()
            df[col].fillna(median_val if pd.notna(median_val) else 0, inplace=True)
    
    return df


def preprocess_data(df):
    df = df.copy()
    
    df = preprocess_numeric_features(df)    
    df = extract_categorical_features(df)
    
    return df


def load_scaler():
    try:
        with open('models/scaler.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise ValueError("Scaler not found at path 'models/scaler.pkl'")


def load_ohe():
    try:
        with open('models/ohe.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise ValueError("OHE not found at path 'models/ohe.pkl'")


def prepare_features(df, scaler=None, ohe=None):
    X_numeric = df[NUMERIC_FEATURES].copy()
    
    if scaler is None:
        scaler = load_scaler()
        
    X_numeric_scaled = scaler.transform(X_numeric)
    X_numeric_scaled = pd.DataFrame(X_numeric_scaled, columns=NUMERIC_FEATURES, index=X_numeric.index)
    
    for f in CATEGORICAL_FEATURES:
        if f not in df.columns:
            if f in ['has_airbag', 'has_abs', 'is_bs4', 'is_luxury']:
                df[f] = False
            elif f == 'segment':
                df[f] = None
            else:
                df[f] = 'unk'
    
    X_cat = df[CATEGORICAL_FEATURES].copy()
    if 'seats' in X_cat.columns:
        X_cat['seats'] = X_cat['seats'].astype(str)
    for bool_col in ['has_airbag', 'has_abs', 'is_bs4', 'is_luxury']:
        if bool_col in X_cat.columns:
            X_cat[bool_col] = X_cat[bool_col].astype(str)
    X_cat = X_cat.fillna('unk').replace([None, np.nan], 'unk')
    
    if ohe is None:
        ohe = load_ohe()

    X_cat_encoded = ohe.transform(X_cat)
    feature_names = ohe.get_feature_names_out(X_cat.columns)
    X_cat_encoded = pd.DataFrame(X_cat_encoded, columns=feature_names, index=X_cat.index)
    
    return pd.concat([X_numeric_scaled, X_cat_encoded], axis=1)


def align_features(X_combined, model):
    if not hasattr(model, 'feature_names_in_'):
        return X_combined
    
    expected_features = list(model.feature_names_in_)
    missing = [f for f in expected_features if f not in X_combined.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    
    return X_combined[expected_features]

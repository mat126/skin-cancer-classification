from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def prepare_data(df):
    # Columns to exclude from the model
    exclude_columns = [
        'isic_id', 'patient_id', 'lesion_id',
        'iddx_full', 'iddx_1', 'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5',
        'mel_mitotic_index', 'mel_thick_mm', 'attribution', 'copyright_license',
        'tbp lv nevi confidence', 'tbp lv dnn lesion confidence',
        'anatom_site_general', 'sex', 'age_approx'
    ]

    # Define features
    feature_columns = [c for c in df.columns if c not in exclude_columns + ['target', 'image_path']]
    numerical = df[feature_columns].select_dtypes(include=['int', 'float']).columns.tolist()
    categorical = df[feature_columns].select_dtypes(include=['object']).columns.tolist()

    # Define preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)
    ])

    # Separate features and target
    X = df[feature_columns]
    y = df['target'].astype(int)

    return X, y, preprocessor

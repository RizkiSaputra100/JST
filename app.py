import io
import hashlib
import os
import random
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


RANDOM_STATE = 42
TARGET_COLUMN = "Satisfaction Score"

COLUMN_TRANSLATIONS = {
    "age": "Usia",
    "customer id": "ID Pelanggan",
    "customer type": "Tipe Pelanggan",
    "class": "Kelas",
    "type of travel": "Jenis Perjalanan",
    "flight distance": "Jarak Penerbangan",
    "inflight wifi service": "Layanan WiFi di Pesawat",
    "departure/arrival time convenient": "Kenyamanan Waktu Berangkat/Tiba",
    "ease of online booking": "Kemudahan Booking Online",
    "gate location": "Lokasi Gerbang",
    "food and drink": "Makanan dan Minuman",
    "online boarding": "Boarding Online",
    "seat comfort": "Kenyamanan Kursi",
    "inflight entertainment": "Hiburan di Pesawat",
    "on-board service": "Layanan di Pesawat",
    "leg room service": "Ruang Kaki",
    "baggage handling": "Penanganan Bagasi",
    "checkin service": "Layanan Check-in",
    "inflight service": "Layanan Selama Penerbangan",
    "cleanliness": "Kebersihan",
    "departure delay in minutes": "Keterlambatan Berangkat (Menit)",
    "arrival delay in minutes": "Keterlambatan Tiba (Menit)",
    "satisfaction": "Kepuasan",
    "satisfaction score": "Skor Kepuasan",
    "rating": "Rating",
    "service quality": "Kualitas Layanan",
    "delivery time": "Waktu Pengiriman",
    "payment method": "Metode Pembayaran",
    "customer segment": "Segmen Pelanggan",
    "gender": "Jenis Kelamin",
    "loyal customer": "Pelanggan Loyal",
    "disloyal customer": "Pelanggan Tidak Loyal",
}

VALUE_TRANSLATIONS = {
    "satisfied": "Puas",
    "satisfaction": "Puas",
    "neutral": "Netral",
    "neutral or dissatisfied": "Netral/Tidak Puas",
    "dissatisfied": "Tidak Puas",
    "unsatisfied": "Tidak Puas",
    "not satisfied": "Tidak Puas",
    "happy": "Puas",
    "unhappy": "Tidak Puas",
    "yes": "Ya",
    "no": "Tidak",
    "male": "Laki-laki",
    "female": "Perempuan",
    "business": "Bisnis",
    "personal": "Pribadi",
    "business travel": "Perjalanan Bisnis",
    "personal travel": "Perjalanan Pribadi",
    "eco": "Ekonomi",
    "eco plus": "Ekonomi Plus",
    "cash": "Tunai",
    "card": "Kartu",
    "transfer": "Transfer",
    "student": "Pelajar/Mahasiswa",
    "worker": "Pekerja",
    "member": "Anggota",
}

warnings.filterwarnings("ignore", category=ConvergenceWarning)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


st.set_page_config(
    page_title="Komparasi JST - Prediksi Kepuasan Pelanggan",
    page_icon="JST",
    layout="wide",
)


def build_one_hot_encoder():
    """Menjaga kompatibilitas antara versi scikit-learn lama dan baru."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def normalize_column_name(column_name):
    return str(column_name).strip().lower().replace("_", " ")


def translate_column(column_name):
    normalized = normalize_column_name(column_name)
    return COLUMN_TRANSLATIONS.get(normalized, str(column_name))


def translate_value(value):
    if pd.isna(value):
        return value
    normalized = normalize_column_name(value)
    return VALUE_TRANSLATIONS.get(normalized, str(value))


def translate_dataframe_for_display(df):
    display_df = df.copy()
    display_df.columns = [translate_column(column) for column in display_df.columns]

    object_columns = display_df.select_dtypes(include=["object", "string", "category"]).columns
    for column in object_columns:
        display_df[column] = display_df[column].map(translate_value)

    return display_df


def find_target_column(columns):
    for column in columns:
        if normalize_column_name(column) == normalize_column_name(TARGET_COLUMN):
            return column
    return None


def find_id_columns(columns):
    id_columns = []
    for column in columns:
        normalized = normalize_column_name(column)
        tokens = normalized.replace("-", " ").split()
        if normalized in {"id", "customer id", "customerid"} or tokens[-1:] == ["id"]:
            id_columns.append(column)
    return id_columns


def transform_satisfaction_target(target_series):
    """Mengubah skor 1-5 menjadi 3 kelas jika target berbentuk angka."""
    numeric_target = pd.to_numeric(target_series, errors="coerce")
    non_missing_values = numeric_target.dropna()

    if not non_missing_values.empty and non_missing_values.between(1, 5).all():
        bins = [-np.inf, 2, 3, np.inf]
        labels = ["Tidak Puas", "Netral", "Puas"]
        return pd.cut(numeric_target, bins=bins, labels=labels).astype(str)

    return target_series.astype(str).str.strip().map(translate_value)


def prepare_dataset(df, selected_target_column=None):
    target_column = selected_target_column or find_target_column(df.columns)
    if target_column is None:
        raise ValueError(
            "Kolom target 'Satisfaction Score' tidak ditemukan. "
            "Pilih kolom target secara manual dari dropdown."
        )

    id_columns = [column for column in find_id_columns(df.columns) if column != target_column]
    cleaned_df = df.drop(columns=id_columns, errors="ignore").copy()

    y_raw = transform_satisfaction_target(cleaned_df[target_column])
    X = cleaned_df.drop(columns=[target_column])

    valid_rows = y_raw.notna() & (y_raw.astype(str).str.lower() != "nan") & (y_raw.astype(str).str.strip() != "")
    X = X.loc[valid_rows].reset_index(drop=True)
    y_raw = y_raw.loc[valid_rows].reset_index(drop=True)

    if X.empty:
        raise ValueError("Dataset tidak memiliki fitur training setelah kolom target dan ID dipisahkan.")

    if len(X) < 10:
        raise ValueError("Dataset terlalu sedikit untuk training. Gunakan minimal 10 baris data yang valid.")

    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Tidak Diketahui")),
            ("encoder", build_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    return {
        "X": X,
        "y": y,
        "y_raw": y_raw,
        "target_column": target_column,
        "id_columns": id_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
    }


def build_keras_model(input_dim, class_count):
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(class_count, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def report_to_row(model_name, y_true, y_pred, labels):
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(labels)),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "Metode": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1-Score": report["weighted avg"]["f1-score"],
    }


def render_dataset_information(df):
    st.subheader("Informasi Dataset")
    col1, col2 = st.columns(2)
    col1.metric("Jumlah Baris", f"{df.shape[0]:,}")
    col2.metric("Jumlah Kolom", f"{df.shape[1]:,}")

    info_df = pd.DataFrame(
        {
            "Kolom Asli": df.columns,
            "Nama Indonesia": [translate_column(column) for column in df.columns],
            "Tipe Data": [str(dtype) for dtype in df.dtypes],
            "Missing Value": df.isna().sum().values,
        }
    )
    st.dataframe(info_df, width="stretch")


def plot_keras_history(history, metric_name):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.history[metric_name], label=f"Training {metric_name}")
    validation_metric = f"val_{metric_name}"
    if validation_metric in history.history:
        ax.plot(history.history[validation_metric], label=f"Validation {metric_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name.capitalize())
    ax.legend()
    ax.grid(alpha=0.25)
    st.pyplot(fig)


def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    st.pyplot(fig)


def plot_numeric_correlation(df, target_column, id_columns):
    numeric_df = df.drop(columns=id_columns + [target_column], errors="ignore").select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        st.info("Heatmap korelasi membutuhkan minimal dua fitur numerik.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Heatmap Korelasi Fitur Numerik")
    st.pyplot(fig)


def make_evaluation_csv(comparison_df, keras_report_df, mlp_report_df):
    output = io.StringIO()
    output.write("Tabel Komparasi\n")
    comparison_df.to_csv(output, index=False)
    output.write("\nClassification Report Keras Sequential\n")
    keras_report_df.to_csv(output)
    output.write("\nClassification Report MLPClassifier\n")
    mlp_report_df.to_csv(output)
    return output.getvalue().encode("utf-8")


st.title("Komparasi Metode JST untuk Prediksi Kepuasan Pelanggan")
st.caption("Studi kasus: Customer Satisfaction 10K Kaggle")

with st.sidebar:
    st.header("Pengaturan Training")
    epochs = st.slider("Epoch Keras", min_value=10, max_value=150, value=50, step=10)
    batch_size = st.selectbox("Batch Size Keras", options=[16, 32, 64, 128], index=1)
    mlp_max_iter = st.slider("Max Iteration MLPClassifier", min_value=100, max_value=1000, value=500, step=100)
    test_size = st.info("Split data: 80% training dan 20% testing")

uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])

if uploaded_file is None:
    st.info("Silakan upload file CSV Customer Satisfaction 10K untuk memulai.")
    st.stop()

try:
    uploaded_bytes = uploaded_file.getvalue()
    dataset_hash = hashlib.md5(uploaded_bytes).hexdigest()
    df = pd.read_csv(io.BytesIO(uploaded_bytes))
except Exception as exc:
    st.error(f"File CSV gagal dibaca: {exc}")
    st.stop()

st.subheader("Preview Dataset")
st.dataframe(translate_dataframe_for_display(df.head(10)), width="stretch")

render_dataset_information(df)

detected_target_column = find_target_column(df.columns)
st.subheader("Target Prediksi")
if detected_target_column:
    st.success(f"Kolom target terdeteksi otomatis: {translate_column(detected_target_column)}")
    selected_target_column = detected_target_column
else:
    st.warning(
        "Kolom 'Satisfaction Score' tidak ditemukan. "
        "Pilih kolom target kepuasan pelanggan secara manual."
    )
    selected_target_column = st.selectbox(
        "Pilih kolom target",
        options=df.columns.tolist(),
        format_func=lambda column: f"{translate_column(column)} ({column})",
        help="Pilih kolom yang berisi nilai kepuasan, misalnya Satisfaction, Satisfaction Score, atau rating 1-5.",
    )

current_signature = f"{dataset_hash}:{selected_target_column}"
if st.session_state.get("trained_signature") != current_signature:
    st.session_state.pop("trained", None)

try:
    prepared = prepare_dataset(df, selected_target_column)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if len(np.unique(prepared["y"])) < 2:
    st.error("Target hanya memiliki satu kelas setelah preprocessing. Model klasifikasi membutuhkan minimal dua kelas.")
    st.stop()

st.subheader("Ringkasan Preprocessing")
preprocess_col1, preprocess_col2, preprocess_col3 = st.columns(3)
preprocess_col1.metric("Kolom ID Dihapus", len(prepared["id_columns"]))
preprocess_col2.metric("Fitur Numerik", len(prepared["numeric_columns"]))
preprocess_col3.metric("Fitur Kategorikal", len(prepared["categorical_columns"]))

if prepared["id_columns"]:
    st.write("Kolom ID yang tidak digunakan sebagai fitur training:", ", ".join(prepared["id_columns"]))

st.write("Distribusi target setelah transformasi:")
target_distribution_df = prepared["y_raw"].value_counts().rename_axis("Kelas").reset_index(name="Jumlah")
target_distribution_df["Kelas"] = target_distribution_df["Kelas"].map(translate_value)
st.dataframe(target_distribution_df, width="stretch")

class_counts = np.bincount(prepared["y"])
test_count = int(np.ceil(len(prepared["y"]) * 0.2))
train_count = len(prepared["y"]) - test_count
can_stratify = (
    class_counts.min() >= 2
    and test_count >= len(class_counts)
    and train_count >= len(class_counts)
)
stratify_y = prepared["y"] if can_stratify else None
X_train, X_test, y_train, y_test = train_test_split(
    prepared["X"],
    prepared["y"],
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=stratify_y,
)

X_train_processed = prepared["preprocessor"].fit_transform(X_train)
X_test_processed = prepared["preprocessor"].transform(X_test)

class_labels = prepared["label_encoder"].classes_.tolist()
display_class_labels = [translate_value(label) for label in class_labels]
class_count = len(class_labels)

st.subheader("Training dan Evaluasi Model")
if st.button("Mulai Training Model", type="primary"):
    with st.spinner("Melatih Keras Sequential dan MLPClassifier..."):
        y_train_categorical = to_categorical(y_train, num_classes=class_count)

        keras_model = build_keras_model(X_train_processed.shape[1], class_count)
        history = keras_model.fit(
            X_train_processed,
            y_train_categorical,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0,
        )

        keras_probabilities = keras_model.predict(X_test_processed, verbose=0)
        keras_predictions = np.argmax(keras_probabilities, axis=1)

        mlp_model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=mlp_max_iter,
            random_state=RANDOM_STATE,
        )
        mlp_model.fit(X_train_processed, y_train)
        mlp_predictions = mlp_model.predict(X_test_processed)

        comparison_df = pd.DataFrame(
            [
                report_to_row("TensorFlow/Keras Sequential", y_test, keras_predictions, display_class_labels),
                report_to_row("Scikit-Learn MLPClassifier", y_test, mlp_predictions, display_class_labels),
            ]
        )

        keras_report_df = pd.DataFrame(
            classification_report(
                y_test,
                keras_predictions,
                labels=np.arange(len(class_labels)),
                target_names=display_class_labels,
                output_dict=True,
                zero_division=0,
            )
        ).transpose()
        mlp_report_df = pd.DataFrame(
            classification_report(
                y_test,
                mlp_predictions,
                labels=np.arange(len(class_labels)),
                target_names=display_class_labels,
                output_dict=True,
                zero_division=0,
            )
        ).transpose()

        st.session_state["trained"] = {
            "keras_model": keras_model,
            "mlp_model": mlp_model,
            "history": history,
            "comparison_df": comparison_df,
            "keras_report_df": keras_report_df,
            "mlp_report_df": mlp_report_df,
            "keras_predictions": keras_predictions,
            "mlp_predictions": mlp_predictions,
            "y_test": y_test,
            "class_labels": class_labels,
            "display_class_labels": display_class_labels,
            "prepared": prepared,
        }
        st.session_state["trained_signature"] = current_signature

if "trained" not in st.session_state:
    st.warning("Klik tombol training untuk menampilkan evaluasi, grafik, dan form prediksi.")
    st.stop()

trained = st.session_state["trained"]

st.markdown("#### Tabel Komparasi Hasil")
st.dataframe(
    trained["comparison_df"].style.format(
        {
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-Score": "{:.4f}",
        }
    ),
    width="stretch",
)

report_col1, report_col2 = st.columns(2)
with report_col1:
    st.markdown("#### Classification Report Keras")
    st.dataframe(trained["keras_report_df"].style.format("{:.4f}"), width="stretch")
with report_col2:
    st.markdown("#### Classification Report MLPClassifier")
    st.dataframe(trained["mlp_report_df"].style.format("{:.4f}"), width="stretch")

csv_bytes = make_evaluation_csv(
    trained["comparison_df"],
    trained["keras_report_df"],
    trained["mlp_report_df"],
)
st.download_button(
    "Download Hasil Evaluasi CSV",
    data=csv_bytes,
    file_name="hasil_evaluasi_komparasi_jst.csv",
    mime="text/csv",
)

st.markdown("#### Grafik Training Keras")
keras_plot_col1, keras_plot_col2 = st.columns(2)
with keras_plot_col1:
    plot_keras_history(trained["history"], "accuracy")
with keras_plot_col2:
    plot_keras_history(trained["history"], "loss")

st.markdown("#### Confusion Matrix")
cm_col1, cm_col2 = st.columns(2)
with cm_col1:
    plot_confusion_matrix(
        trained["y_test"],
        trained["keras_predictions"],
        trained["display_class_labels"],
        "Keras Sequential",
    )
with cm_col2:
    plot_confusion_matrix(
        trained["y_test"],
        trained["mlp_predictions"],
        trained["display_class_labels"],
        "MLPClassifier",
    )

st.markdown("#### Korelasi Fitur Numerik")
plot_numeric_correlation(
    df,
    trained["prepared"]["target_column"],
    trained["prepared"]["id_columns"],
)

st.subheader("Prediksi Data Pelanggan Baru")
with st.form("new_customer_prediction_form"):
    form_values = {}
    input_columns = trained["prepared"]["X"].columns.tolist()
    for column in input_columns:
        if column in trained["prepared"]["numeric_columns"]:
            median_value = pd.to_numeric(trained["prepared"]["X"][column], errors="coerce").median()
            if pd.isna(median_value):
                median_value = 0.0
            form_values[column] = st.number_input(
                translate_column(column),
                value=float(median_value),
                key=f"num_{column}",
            )
        else:
            available_values = (
                trained["prepared"]["X"][column]
                .dropna()
                .astype(str)
                .sort_values()
                .unique()
                .tolist()
            )
            if not available_values:
                available_values = [""]
            form_values[column] = st.selectbox(
                translate_column(column),
                options=available_values,
                format_func=translate_value,
                key=f"cat_{column}",
            )

    selected_model = st.radio(
        "Pilih model untuk prediksi",
        ["TensorFlow/Keras Sequential", "Scikit-Learn MLPClassifier"],
        horizontal=True,
    )
    submitted = st.form_submit_button("Prediksi Kepuasan")

if submitted:
    new_customer_df = pd.DataFrame([form_values], columns=input_columns)
    new_customer_processed = trained["prepared"]["preprocessor"].transform(new_customer_df)

    if selected_model == "TensorFlow/Keras Sequential":
        probability = trained["keras_model"].predict(new_customer_processed, verbose=0)
        predicted_class_index = int(np.argmax(probability, axis=1)[0])
    else:
        predicted_class_index = int(trained["mlp_model"].predict(new_customer_processed)[0])

    predicted_label = translate_value(trained["prepared"]["label_encoder"].inverse_transform([predicted_class_index])[0])
    st.success(f"Hasil prediksi kepuasan pelanggan: {predicted_label}")

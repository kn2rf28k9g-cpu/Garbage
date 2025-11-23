import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Seiten-Konfiguration
st.set_page_config(
    page_title="♻️ Garbage Classifier",
    page_icon="♻️",
    layout="wide"
)

# CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E7D32;
        margin-bottom: 1rem;
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 8px;
        background: #f0f2f6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Klassen-Definitionen
CLASS_NAMES = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
CLASS_EMOJIS = ['📦', '🍾', '🥫', '📄', '🧴', '🗑️']
CLASS_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#95A5A6']

# Recycling-Informationen
RECYCLING_INFO = {
    'Cardboard': {
        'recyclable': True,
        'bin': 'Papiertonne (Blau)',
        'tips': '✅ Kartons flach zusammenfalten\n✅ Sauber und trocken halten\n❌ Keine beschichteten Kartons'
    },
    'Glass': {
        'recyclable': True,
        'bin': 'Glascontainer',
        'tips': '✅ Nach Farben trennen\n✅ Deckel entfernen\n❌ Kein Fensterglas oder Keramik'
    },
    'Metal': {
        'recyclable': True,
        'bin': 'Gelber Sack',
        'tips': '✅ Dosen ausspülen\n✅ Metalldeckel separat\n❌ Keine verschmutzten Metalle'
    },
    'Paper': {
        'recyclable': True,
        'bin': 'Papiertonne (Blau)',
        'tips': '✅ Sauber und trocken\n✅ Zeitungen und Zeitschriften\n❌ Kein verschmutztes Papier'
    },
    'Plastic': {
        'recyclable': True,
        'bin': 'Gelber Sack',
        'tips': '✅ Verpackungen ausspülen\n✅ Nach Nummer sortieren\n❌ Keine Plastiktüten'
    },
    'Trash': {
        'recyclable': False,
        'bin': 'Restmülltonne (Schwarz)',
        'tips': '⚠️ Nicht recycelbar\n⚠️ In Restmüll entsorgen\n💡 Versuche weniger Müll zu produzieren'
    }
}


@st.cache_resource
def load_model():
    """Lade das trainierte Modell"""
    try:
        model = tf.keras.models.load_model('models/best_model.keras')
        return model
    except Exception as e:
        st.error(f"❌ Fehler beim Laden des Modells: {e}")
        st.info("💡 Stelle sicher, dass 'models/best_model.keras' existiert")
        return None


def preprocess_image(image):
    """Bereite Bild für Vorhersage vor"""
    # Resize zu 299x299 (InceptionV3 Input-Größe)
    img = image.resize((299, 299))
    # Convert zu Array
    img_array = np.array(img)
    # Stelle sicher dass es 3 Kanäle hat (RGB)
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    # Normalisiere auf [0, 1]
    img_array = img_array.astype('float32') / 255.0
    # Füge Batch-Dimension hinzu
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def create_confidence_chart(predictions, class_names):
    """Erstelle interaktives Confidence-Chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=predictions[0] * 100,
            y=class_names,
            orientation='h',
            marker=dict(
                color=CLASS_COLORS,
                line=dict(color='white', width=2)
            ),
            text=[f'{p * 100:.1f}%' for p in predictions[0]],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title='Confidence Scores für alle Klassen',
        xaxis_title='Confidence (%)',
        yaxis_title='Kategorie',
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return fig


# Header
st.markdown('<div class="main-header">♻️ Garbage Classifier AI</div>', unsafe_allow_html=True)
st.markdown("### 🤖 Intelligente Müll-Klassifizierung mit KI")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x150/2E7D32/FFFFFF?text=AI+Recycling", use_container_width=True)
    st.markdown("## 📊 Über diese App")
    st.info("""
    Diese App nutzt ein **InceptionV3 Deep Learning Modell** 
    trainiert auf 10.000+ Bildern um Müll automatisch zu klassifizieren.

    **Kategorien:**
    - 📦 Cardboard (Karton)
    - 🍾 Glass (Glas)
    - 🥫 Metal (Metall)
    - 📄 Paper (Papier)
    - 🧴 Plastic (Plastik)
    - 🗑️ Trash (Restmüll)
    """)

    st.markdown("## 🎯 Model Performance")
    st.metric("Validation Accuracy", "88.1%", "+0.6%")
    st.metric("Top-2 Accuracy", "96.5%", "+2.1%")
    st.metric("Total Parameters", "24.4M", "")

# Hauptbereich
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📸 Bild hochladen")
    uploaded_file = st.file_uploader(
        "Wähle ein Bild von deinem Müll...",
        type=['jpg', 'jpeg', 'png'],
        help="Unterstützte Formate: JPG, JPEG, PNG"
    )

    # Demo-Bilder Option
    use_demo = st.checkbox("🎨 Oder nutze ein Demo-Bild")
    if use_demo:
        demo_option = st.selectbox(
            "Wähle Demo-Kategorie:",
            CLASS_NAMES
        )

with col2:
    st.markdown("### 🔍 Klassifizierung")

    if uploaded_file is not None or use_demo:
        # Lade Modell
        model = load_model()

        if model is not None:
            # Zeige Bild
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
            else:
                # Placeholder für Demo-Bilder
                st.info(f"Demo-Modus: {demo_option} ausgewählt")
                st.warning("💡 In echter App: Lade hier ein echtes {demo_option}-Bild")
                image = Image.new('RGB', (299, 299), color='lightgray')

            st.image(image, caption='Hochgeladenes Bild', use_column_width=True)

            # Klassifiziere Button
            if st.button('🚀 Jetzt klassifizieren!', type='primary'):
                with st.spinner('🤖 KI analysiert dein Bild...'):
                    # Preprocessing
                    processed_img = preprocess_image(image)

                    # Prediction
                    predictions = model.predict(processed_img, verbose=0)
                    predicted_class_idx = np.argmax(predictions[0])
                    predicted_class = CLASS_NAMES[predicted_class_idx]
                    confidence = predictions[0][predicted_class_idx] * 100

                    # Zeige Ergebnis
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h1>{CLASS_EMOJIS[predicted_class_idx]} {predicted_class}</h1>
                        <h2>{confidence:.1f}% Confidence</h2>
                    </div>
                    """, unsafe_allow_html=True)

                    # Recycling-Info
                    info = RECYCLING_INFO[predicted_class]

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("#### ♻️ Recycelbar?")
                        if info['recyclable']:
                            st.success("✅ Ja, recycelbar!")
                        else:
                            st.error("❌ Nein, Restmüll")

                    with col_b:
                        st.markdown("#### 🗑️ Richtige Tonne")
                        st.info(f"**{info['bin']}**")

                    st.markdown("#### 💡 Entsorgungstipps")
                    st.markdown(info['tips'])

                    # Confidence Chart
                    st.plotly_chart(
                        create_confidence_chart(predictions, CLASS_NAMES),
                        use_container_width=True
                    )

                    # Top-3 Predictions
                    st.markdown("#### 🏆 Top 3 Vorhersagen")
                    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
                    for i, idx in enumerate(top_3_idx, 1):
                        conf = predictions[0][idx] * 100
                        st.markdown(f"{i}. {CLASS_EMOJIS[idx]} **{CLASS_NAMES[idx]}** - {conf:.1f}%")
        else:
            st.error("❌ Modell konnte nicht geladen werden")

# Footer
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    st.markdown("### 📊 Statistiken")
    st.metric("Bilder klassifiziert", "0", "heute")

with col_f2:
    st.markdown("### ⏱️ Durchschnitt")
    st.metric("Inferenz-Zeit", "~0.5s", "pro Bild")

with col_f3:
    st.markdown("### 🌍 CO₂ Ersparnis")
    st.metric("Durch Recycling", "0 kg", "geschätzt")

st.markdown("""
---
<div style='text-align: center; color: gray;'>
    Made with ❤️ using TensorFlow & Streamlit | 
    Model: InceptionV3 | 
    Dataset: 10,517 Bilder
</div>
""", unsafe_allow_html=True)
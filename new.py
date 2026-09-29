import streamlit as st
import cv2 as cv
import numpy as np
import keras
from datetime import datetime

st.set_page_config(
    page_title="Grape Guardian",
    page_icon="🍇",
    layout="centered"
)

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'home'
if 'language' not in st.session_state:
    st.session_state['language'] = 'English'
if 'show_uploader' not in st.session_state:
    st.session_state['show_uploader'] = False

translations = {
    'English': {
        'title': "🍇 Grape Guardian",
        'subtitle': "Smart Disease Detection for Your Farm",
        'start_btn': "📸 Start Diagnosis",
        'upload_text': "Upload a photo of a grape leaf",
        'back': "← Back to Home",
        'learn_more': "📖 Disease Encyclopedia",
        'warning': "Please upload a clear image of a grape leaf only.",
        'analyzing': "Consulting the digital expert...",
        'reset_btn': "🔄 Check Another Leaf",
        'confidence': "Confidence",
        'symptoms_lbl': "🔍 Symptoms",
        'treatment_lbl': "💊 Treatment",
        'prevention_lbl': "🛡️ Prevention",
        'safe': "Healthy Leaf",
        'danger': "Disease Detected"
    },
    'Marathi': {
        'title': "🍇 द्राक्ष रक्षक",
        'subtitle': "तुमच्या बागेसाठी स्मार्ट रोग निदान",
        'start_btn': "📸 निदान सुरू करा",
        'upload_text': "द्राक्षाच्या पानाचा फोटो टाका",
        'back': "← मुख्य पानावर जा",
        'learn_more': "📖 रोगांची माहिती पुस्तिक",
        'warning': "कृपया फक्त द्राक्षाच्या पानाचा स्पष्ट फोटो टाका.",
        'analyzing': "तपासणी चालू आहे...",
        'reset_btn': "🔄 दुसरे पान तपासा",
        'confidence': "खात्री",
        'symptoms_lbl': "🔍 लक्षणे",
        'treatment_lbl': "💊 उपाययोजना (फवारणी)",
        'prevention_lbl': "🛡️ प्रतिबंध आणि काळजी",
        'safe': "निरोगी पान",
        'danger': "रोग आढळला"
    },
    'Hindi': {
        'title': "🍇 अंगूर रक्षक",
        'subtitle': "आपके खेत के लिए स्मार्ट रोग पहचान",
        'start_btn': "📸 निदान शुरू करें",
        'upload_text': "अंगूर के पत्ते की फोटो डालें",
        'back': "← मुख्य पृष्ठ पर जाएं",
        'learn_more': "📖 रोग ज्ञानकोश",
        'warning': "कृपया केवल अंगूर के पत्ते की स्पष्ट फोटो अपलोड करें।",
        'analyzing': "जांच चल रही है...",
        'reset_btn': "🔄 दूसरा पत्ता जांचें",
        'confidence': "विश्वास",
        'symptoms_lbl': "🔍 लक्षण",
        'treatment_lbl': "💊 उपचार",
        'prevention_lbl': "🛡️ बचाव",
        'safe': "स्वस्थ पत्ता",
        'danger': "रोग पाया गया"
    }
}

disease_db = {
    "Grape Black rot": {
        "en": {
            "name": "Black Rot",
            "symptoms": "Small yellowish spots on leaves. Berries shrivel and turn into hard black 'mummies'.",
            "cure": "Spray Mancozeb or Myclobutanil immediately.",
            "prev": "Remove mummified berries. Ensure good sunlight and airflow."
        },
        "mr": {
            "name": "काळा कुजवा (Black Rot)",
            "symptoms": "पानांवर पिवळसर ठिपके पडतात. द्राक्षमणी सुकून काळे होतात (ममीफिकेशन).",
            "cure": "तातडीने मॅनकोझेब (Mancozeb) किंवा मायक्लोबुटानिलची फवारणी करा.",
            "prev": "सुकलेले काळे मणी बागेतून काढून टाका. बागेत हवा खेळती ठेवा."
        },
        "hi": {
            "name": "काला सड़न (Black Rot)",
            "symptoms": "पत्तियों पर पीले धब्बे। फल सूखकर काले और सख्त हो जाते हैं।",
            "cure": "तुरंत मैनकोजेब या मायक्लोबुटानिल का छिड़काव करें।",
            "prev": "सूखे और संक्रमित फलों को हटा दें। धूप और हवा का प्रवाह सुनिश्चित करें।"
        }
    },
    "Grape Esca": {
        "en": {
            "name": "Esca (Measles)",
            "symptoms": "Leaves show 'tiger-stripes' (yellow/red patches). Berries have small dark spots.",
            "cure": "No chemical cure exists. Use wound sealants after pruning.",
            "prev": "Remove infected vines immediately to stop spread. Sanitize pruning tools."
        },
        "mr": {
            "name": "एस्का (हुमणी/रोगामुळे वाळणे)",
            "symptoms": "पानांवर वाघाच्या कातडीसारखे पट्टे दिसतात. मण्यांवर लहान काळे ठिपके येतात.",
            "cure": "यावर रासायनिक औषध नाही. छाटणीनंतर जखमेवर पेस्ट लावा.",
            "prev": "संसर्ग वाढू नये म्हणून बाधित झाड मुळासकट काढून टाका. हत्यारे निर्जंतुक करा."
        },
        "hi": {
            "name": "एस्का (Esca)",
            "symptoms": "पत्तियों पर 'टाइगर-स्ट्राइप' (पीले/लाल धब्बे) दिखाई देते हैं।",
            "cure": "कोई रासायनिक इलाज नहीं है। कटाई के बाद घाव पर लेप लगाएं।",
            "prev": "फैलाव रोकने के लिए संक्रमित बेलों को हटा दें। औजारों को साफ रखें।"
        }
    },
    "Grape Leaf blight": {
        "en": {
            "name": "Leaf Blight",
            "symptoms": "Irregular dark red/brown spots on leaves. Leaves may fall off early.",
            "cure": "Apply Copper Oxychloride or Bordeaux mixture.",
            "prev": "Avoid overhead irrigation. Prune dense canopies."
        },
        "mr": {
            "name": "पानावरील करपा (Leaf Blight)",
            "symptoms": "पानांवर अनियमित लाल/तपकिरी डाग पडतात. पाने वेळेआधी गळतात.",
            "cure": "कॉपर ऑक्सिक्लोराईड किंवा बोर्डो मिश्रणाची फवारणी करा.",
            "prev": "पानांवर पाणी साचू देऊ नका. गर्दी कमी करण्यासाठी छाटणी करा."
        },
        "hi": {
            "name": "पत्ती का झुलसा (Leaf Blight)",
            "symptoms": "पत्तियों पर अनियमित गहरे लाल/भूरे धब्बे। पत्ते जल्दी गिर सकते हैं।",
            "cure": "कॉपर ऑक्सीक्लोराइड या बोर्डो मिश्रण का प्रयोग करें।",
            "prev": "ऊपर से सिंचाई (Overhead irrigation) से बचें। हवा के लिए छंटाई करें।"
        }
    },
    "Grape healthy": {
        "en": {
            "name": "Healthy Leaf",
            "symptoms": "Leaves are vibrant green without spots or cuts.",
            "cure": "No treatment needed.",
            "prev": "Continue regular water and fertilizer schedule."
        },
        "mr": {
            "name": "निरोगी पान",
            "symptoms": "पाने हिरवीगार आणि तजेलदार आहेत. कोणताही डाग नाही.",
            "cure": "कोणत्याही औषधाची गरज नाही.",
            "prev": "नियमित पाणी आणि खत व्यवस्थापन चालू ठेवा.",
            "name": "निरोगी पान"
        },
        "hi": {
            "name": "स्वस्थ पत्ता",
            "symptoms": "पत्ते चमकदार हरे हैं और उन पर कोई धब्बे नहीं हैं।",
            "cure": "किसी उपचार की आवश्यकता नहीं है।",
            "prev": "नियमित पानी और खाद डालना जारी रखें।"
        }
    }
}

def get_text(key):
    return translations[st.session_state['language']][key]

def navigate_to(page):
    st.session_state['current_page'] = page

st.markdown("""
<style>
    /* Dark Background */
    .stApp {
        background: radial-gradient(at top left, #1e293b 0%, #0f172a 100%);
        color: #020800;
    }
    
    /* Glass Effect Containers */
    .glass-card, [data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Header Styling */
    .header-box {
        text-align: center;
        background: rgba(15, 23, 42, 0.8);
        padding: 25px;
        border-radius: 20px;
        margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .header-box h1 {
        background: linear-gradient(to right, #4ade80, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 30px;
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: white;
        border: none;
        font-weight: bold;
        padding: 0.75rem 1rem;
    }
    .stButton > button:hover {
        transform: scale(1.02);
    }
    
    /* Custom Text Classes */
    .highlight-text {
        font-size: 1.2rem;
        color: #f8fafc;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

def show_home_page():
    # Language Selector
    col_lang1, col_lang2 = st.columns([2.5, 1.5])
    with col_lang2:
        lang = st.selectbox("🌐 Language / भाषा", ['English', 'Marathi', 'Hindi'])
        st.session_state['language'] = lang

    # Header
    st.markdown(f"""
    <div class="header-box">
        <h1>{get_text('title')}</h1>
        <p style='color: #cbd5e1;'>{get_text('subtitle')}</p>
        <p style='font-size: 0.8rem; color: #94a3b8;'>📅 {datetime.now().strftime('%d %B %Y')}</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button(get_text('learn_more')):
        navigate_to('info')
        st.rerun()

    # --- MAIN LOGIC ---
    if not st.session_state['show_uploader']:
        st.write("")
        if st.button(get_text('start_btn'), type="primary", use_container_width=True):
            st.session_state['show_uploader'] = True
            st.rerun()
    else:
        st.markdown(f"### {get_text('upload_text')}")
        uploaded_file = st.file_uploader("", label_visibility="collapsed")
        
        if st.button(get_text('reset_btn')):
            st.session_state['show_uploader'] = False
            st.rerun()

        # Labels must match your model's output classes exactly
        label_name = ['Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy']
        model_path = r'D:\Projects\leaf-diseases-detect-main\Training\model\Leaf Deases(96,88)5.h5' 

        if uploaded_file is not None:
            try:
                # Display Image
                st.image(uploaded_file, use_container_width=True, caption="Uploaded Leaf")
                
                # Load Model & Predict
                model = keras.models.load_model(model_path)
                image_bytes = uploaded_file.read()
                img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
                normalized_image = np.expand_dims(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (150, 150)), axis=0)

                with st.spinner(get_text('analyzing')):
                    predictions = model.predict(normalized_image)

                pred = predictions[0]
                idx = int(np.argmax(pred))
                confidence = float(pred[idx]) * 100.0
                raw_name = label_name[idx]

                # --- DISPLAY RESULTS ---
                if confidence >= 95:
                    lang_code = {'English':'en', 'Marathi':'mr', 'Hindi':'hi'}[st.session_state['language']]
                    
                    # Safety check for dictionary key
                    if raw_name in disease_db:
                        d_info = disease_db[raw_name][lang_code]
                        is_healthy = "healthy" in raw_name.lower()
                        
                        # Status Banner
                        if is_healthy:
                            st.success(f"✅ {d_info['name']} ({confidence:.1f}%)")
                        else:
                            st.error(f"⚠️ {d_info['name']} ({confidence:.1f}%)")
                        
                        # Detailed Info in Native Streamlit Cards (Guaranteed visibility)
                        with st.container():
                            st.markdown("---")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.info(f"**{get_text('symptoms_lbl')}**\n\n{d_info['symptoms']}")
                            with col2:
                                st.warning(f"**{get_text('treatment_lbl')}**\n\n{d_info['cure']}")
                            
                            st.success(f"**{get_text('prevention_lbl')}**\n\n{d_info['prev']}")
                    else:
                        st.error(f"Disease '{raw_name}' details not found in database.")
                else:
                    st.warning(get_text('warning'))

            except Exception as e:
                st.error(f"Error: {e}")

def show_info_page():
    st.markdown(f"<div class='header-box'><h1>{get_text('learn_more')}</h1></div>", unsafe_allow_html=True)

    if st.button(get_text('back')):
        navigate_to('home')
        st.rerun()

    lang_code = {'English':'en', 'Marathi':'mr', 'Hindi':'hi'}[st.session_state['language']]

    for key, data in disease_db.items():
        info = data[lang_code]
        with st.expander(f"🍇 {info['name']}", expanded=False):
            st.markdown(f"""
            **{get_text('symptoms_lbl')}:** {info['symptoms']}
            
            **{get_text('treatment_lbl')}:** {info['cure']}
            
            **{get_text('prevention_lbl')}:** {info['prev']}
            """)

if st.session_state['current_page'] == 'home':
    show_home_page()
else:
    show_info_page()
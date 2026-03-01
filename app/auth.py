import streamlit as st
from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_PUBLISHABLE_KEY")
)

PLANES = {
    "basico":      {"nombre": "Básico",      "meses": 3,  "precio": "$9/mes"},
    "pro":         {"nombre": "Pro",          "meses": 12, "precio": "$19/mes"},
    "empresarial": {"nombre": "Empresarial",  "meses": 999,"precio": "$39/mes"},
}

def get_plan_usuario():
    """Obtiene el plan del usuario logueado desde Supabase."""
    try:
        user = supabase.auth.get_user()
        if not user: return None
        perfil = supabase.table("profiles").select("*").eq("id", user.user.id).single().execute()
        return perfil.data
    except:
        return None

def login_ui():
    """Muestra el formulario de login/registro."""
    st.markdown("""
    <style>
    .auth-box {
        background: white;
        border-radius: 16px;
        padding: 40px;
        max-width: 420px;
        margin: 60px auto;
        border: 1px solid #e8e3d9;
        box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    }
    .auth-title {
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.8rem;
        color: #1c1f26;
        margin-bottom: 6px;
        text-align: center;
    }
    .auth-sub {
        font-size: 13px;
        color: #8a909a;
        text-align: center;
        margin-bottom: 28px;
    }
    .plan-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1px;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="auth-box">
            <div class="auth-title">SalesIQ Pro</div>
            <div class="auth-sub">Inteligencia de ventas con ML</div>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_registro = st.tabs(["Iniciar sesión", "Crear cuenta"])

        with tab_login:
            email = st.text_input("Email", key="login_email", placeholder="tu@email.com")
            password = st.text_input("Contraseña", type="password", key="login_pass", placeholder="••••••••")
            if st.button("Entrar →", key="btn_login"):
                try:
                    res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    st.session_state["user"] = res.user
                    st.success("¡Bienvenido!")
                    st.rerun()
                except Exception as e:
                    st.error("Email o contraseña incorrectos")

        with tab_registro:
            st.markdown("""
            <div style='background:#f0f7ff;border-radius:10px;padding:12px 16px;
                        font-size:12px;color:#1e4a8a;margin-bottom:16px;
                        border-left:3px solid #3d52a0;'>
                <strong>¿Ya pagaste?</strong> Crea tu cuenta con el mismo email
                que usaste al contactarnos por WhatsApp.
            </div>
            """, unsafe_allow_html=True)
            nombre = st.text_input("Nombre", key="reg_nombre", placeholder="Tu nombre")
            email_r = st.text_input("Email", key="reg_email", placeholder="tu@email.com")
            pass_r = st.text_input("Contraseña", type="password", key="reg_pass", placeholder="Mínimo 6 caracteres")
            if st.button("Crear cuenta →", key="btn_registro"):
                try:
                    res = supabase.auth.sign_up({"email": email_r, "password": pass_r})
                    st.success("¡Cuenta creada! Revisa tu email para confirmar.")
                    st.info("Después de confirmar tu email, escríbenos por WhatsApp para activar tu plan.")
                except Exception as e:
                    st.error(f"Error: {e}")

def logout():
    supabase.auth.sign_out()
    st.session_state.clear()
    st.rerun()

def require_auth():
    """
    Llama esto al inicio de app.py.
    Si el usuario no está logueado, muestra el login.
    Si está logueado, retorna su perfil con el plan.
    """
    if "user" not in st.session_state:
        login_ui()
        st.stop()
    perfil = get_plan_usuario()
    if not perfil:
        login_ui()
        st.stop()
    if not perfil.get("activo"):
        st.warning("⏳ Tu cuenta está pendiente de activación. Escríbenos por WhatsApp: +51929201444")
        st.stop()
    return perfil

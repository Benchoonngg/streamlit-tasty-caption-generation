import streamlit as st
import os
import requests
import json
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from dotenv import load_dotenv 
import hashlib
import time
import hmac
import extra_streamlit_components as stx
import datetime

load_dotenv() 

# Initialize session state variables
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'password_correct' not in st.session_state:
    st.session_state.password_correct = False

def get_manager():
    return stx.CookieManager()

cookie_manager = get_manager()

# --- Authentication Functions ---
def check_credentials(username, password):
    """Check if username/password combination exists in authorized users"""
    try:
        # Check if username exists and password matches
        if username in st.secrets.authorized_users:
            return st.secrets.authorized_users[username] == password
        return False
    except Exception as e:
        st.error(f"Error in authentication")
        return False

def get_session_token(username: str, password: str) -> str:
    """Create a session token from username and timestamp"""
    timestamp = str(int(time.time()))
    token = f"{username}{password}{timestamp}"
    return hashlib.sha256(token.encode()).hexdigest()

def is_valid_token(token: str) -> bool:
    """Validate the session token from query parameters"""
    if not token:
        return False
    try:
        # Get stored token from session state
        stored_token = st.session_state.get('session_token')
        return stored_token == token
    except Exception:
        return False

def check_password():
    """Returns `True` if the user had the correct password."""
    
    # Check if authentication cookie exists
    cookie_name = "tasty_caption_auth"
    cookie_value = cookie_manager.get(cookie_name)
    
    # Only accept non-empty cookie values
    if cookie_value and cookie_value.strip():
        stored_username = cookie_value
        st.session_state.authenticated = True
        st.session_state.username = stored_username
        return True

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            username = st.text_input("Username", key="username")
            password = st.text_input("Password", type="password", key="password")
            submitted = st.form_submit_button("Log in", on_click=password_entered)
            return submitted

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets["authorized_users"]:
            if hmac.compare_digest(
                st.session_state["password"],
                st.secrets.authorized_users[st.session_state["username"]],
            ):
                st.session_state["password_correct"] = True
                st.session_state["authenticated"] = True
                # Don't store password in session state
                del st.session_state["password"]
                # Store username in session state
                st.session_state["username"] = st.session_state["username"]
                # Set authentication cookie
                cookie_manager.set(
                    "tasty_caption_auth", 
                    st.session_state["username"], 
                    expires_at=datetime.datetime.now() + datetime.timedelta(days=30)
                )
            else:
                st.session_state["password_correct"] = False
                st.session_state["authenticated"] = False
        else:
            st.session_state["password_correct"] = False
            st.session_state["authenticated"] = False

    # Return True if the user is already authenticated
    if st.session_state.authenticated:
        return True

    # Show inputs for username + password
    if login_form():
        if not st.session_state.password_correct:
            # Clear any existing cookies on failed login
            cookie_manager.delete("tasty_caption_auth")
            cookie_manager.set("tasty_caption_auth", "", expires_at=datetime.datetime.now())
            st.error("😕 User not known or password incorrect")
        else:
            return True
    return False

def initialize_api_credentials():
    """Initialize API credentials after successful login"""
    # Define the required scope
    scope = "https://www.googleapis.com/auth/cloud-platform"
    service_account_info = st.secrets["credentials"]

    # Load the service account credentials
    vertex_credentials = service_account.Credentials.from_service_account_info(
        service_account_info, scopes=[scope]
    )

    # Refresh the credentials to get an access token
    vertex_credentials.refresh(Request())
    return vertex_credentials.token

# --- Load External CSS ---
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize all session state variables with default values"""
    defaults = {
        "authenticated": False,
        "username": None,
        "generated_captions": [],
        "num_captions": 1,
        "max_length": 1024,
        "temperature": 0.90,
        "top_k": 50,
        "top_p": 0.90
    }
    
    # Only set defaults if they don't exist in session state
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def add_logout_button():
    """Add a logout button to the bottom of the sidebar"""
    st.sidebar.markdown('<div style="height: 40vh;"></div>', unsafe_allow_html=True)
    if st.sidebar.button("Logout"):
        # Clear cookie
        cookie_manager.delete("tasty_caption_auth")
        cookie_manager.set("tasty_caption_auth", "", expires_at=datetime.datetime.now())
        
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Initialize fresh session state
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.password_correct = False
        
        # Force cookie expiration in browser
        cookie_manager.set(
            "tasty_caption_auth",
            "",
            expires_at=datetime.datetime.now() - datetime.timedelta(days=1)
        )
        
        # Force a rerun to update the UI
        st.rerun()

# Main Streamlit App
def main():
    if not check_password():
        st.stop()

    # Initialize session state
    initialize_session_state()
    
    # Load CSS file
    load_css("styles.css")
    
    # Initialize API credentials after successful login
    access_token = initialize_api_credentials()
    
    # Streamlit App Title
    st.markdown(
        "<h1 style='text-align: center;'>🫦 Tasty Caption Generation 💦</h1>",
        unsafe_allow_html=True
    )
    
    # Welcome message with username
    st.markdown(f"Welcome, {st.session_state.username}!")
    
    # Main content: Input fields and caption generation
    instruction = st.text_input("Enter Instruction:", placeholder="Generate a *Category* Caption")
    input_text = st.text_area("Enter Context:", placeholder="Describe the Caption")
    
    if st.button("Generate Captions"):
        is_valid, error_message = validate_inputs(instruction, input_text)
        if not is_valid:
            st.error(error_message)
        else:
            with st.spinner("Generating captions..."):
                # Clear previous captions
                st.session_state.generated_captions = []

                # Generate captions logic
                for i in range(st.session_state["num_captions"]):
                    response = generate_caption_from_api(
                        instruction,
                        input_text,
                        st.session_state["max_length"],
                        st.session_state["temperature"],
                        st.session_state["top_k"],
                        st.session_state["top_p"],
                        access_token
                    )
                    
                    if response:
                        # Store caption in session state
                        st.session_state.generated_captions.append(response)

    # Display stored captions
    for i, caption in enumerate(st.session_state.generated_captions):
        st.write(f"**Caption {i + 1}:** {caption}")

    # Generation Settings in Sidebar
    with st.sidebar:
        st.header("Generation Settings")
        
        # Add template buttons first
        st.markdown("### Templates")
        st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        # Default Template
        if col1.button("Default"):
            st.session_state["num_captions"] = 1
            st.session_state["max_length"] = 1024
            st.session_state["temperature"] = 0.90
            st.session_state["top_k"] = 50
            st.session_state["top_p"] = 0.90
            st.rerun()
        
        # Template 2
        if col2.button("Template 2"):
            st.session_state["num_captions"] = 1
            st.session_state["max_length"] = 1024
            st.session_state["temperature"] = 1.20
            st.session_state["top_k"] = 80
            st.session_state["top_p"] = 0.40
            st.rerun()
        
        # Template 3
        if col3.button("Template 3"):
            st.session_state["num_captions"] = 1
            st.session_state["max_length"] = 1024
            st.session_state["temperature"] = 1.30
            st.session_state["top_k"] = 90
            st.session_state["top_p"] = 0.50
            st.rerun()
        
        st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)
        
        # Sliders and inputs for settings with tooltips
        st.slider(
            "Number of Captions", 
            min_value=1, 
            max_value=100, 
            value=st.session_state.num_captions,
            key="num_captions",
            help="Controls how many different captions to generate. Higher values will generate more variations but take longer."
        )
        st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
        
        st.select_slider(
            "Max Tokens", 
            options=[256, 512, 1024], 
            value=st.session_state.max_length,
            key="max_length",
            help="Maximum length of the generated caption in tokens. Higher values allow for longer captions but may increase generation time."
        )
        st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
        
        st.slider(
            "Temperature", 
            min_value=0.0, 
            max_value=1.5, 
            value=st.session_state.temperature,
            step=0.10, 
            key="temperature",
            help="Controls randomness in the generation. Higher values (e.g., 1.0) make output more random, lower values (e.g., 0.2) make it more focused and deterministic."
        )
        st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
        
        st.slider(
            "Top-K", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.top_k,
            step=10, 
            key="top_k",
            help="Limits the cumulative probability of tokens considered for generation. Only the top K most likely tokens are considered. Lower values increase focus but may reduce creativity."
        )
        st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
        
        st.slider(
            "Top-P", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state.top_p,
            step=0.10, 
            key="top_p",
            help="Also known as nucleus sampling. Controls diversity by considering tokens whose cumulative probability exceeds P. Lower values (0.1) are more focused, higher values (0.9) are more diverse."
        )

    # Add logout button at the bottom of sidebar
    add_logout_button()

def generate_caption_from_api(
    instruction: str,
    input_text: str,
    max_length: int,
    temperature: float,
    top_k: int,
    top_p: float,
    access_token: str
) -> str:
    """
    Generate caption using API call to Vertex AI.
    
    Args:
        instruction: The instruction for caption generation
        input_text: The context for caption generation
        max_length: Maximum length of generated text
        temperature: Temperature for text generation
        top_k: Top-k parameter for sampling
        top_p: Top-p parameter for sampling
        access_token: Authentication token for API access
    
    Returns:
        str: Generated caption text
    
    Raises:
        ValueError: If API request fails or returns error
    """
    # Alpaca prompt template
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    # Define the endpoint URL using the endpoints section
    url = f"https://{st.secrets.endpoints.ENDPOINT_DNS}/v1beta1/{st.secrets.endpoints.ENDPOINT_RESOURCE_NAME}/chat/completions"

    payload = {
        "messages": [{"role": "user", "content": alpaca_prompt.format(instruction, input_text, "")}],
        "max_tokens": max_length,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    # Send the POST request to the API
    response = requests.post(
        url, headers={"Authorization": f"Bearer {access_token}"}, json=payload, stream=True
    )
    
    if not response.ok:
        raise ValueError(response.text)

    result = []  # List to accumulate the chunks
    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False):
        if chunk:
            chunk = chunk.decode("utf-8").removeprefix("data:").strip()
            if chunk == "[DONE]":
                break
            data = json.loads(chunk)
            if type(data) is not dict or "error" in data:
                raise ValueError(data)
            delta = data["choices"][0]["delta"].get("content")
            if delta:
                result.append(delta)  # Accumulate the chunks
    full_result = ''.join(result)
    return full_result

def validate_inputs(instruction: str, input_text: str) -> tuple[bool, str]:
    """Validate user inputs before generation"""
    if not instruction.strip():
        return False, "Instruction cannot be empty"
    if not input_text.strip():
        return False, "Context cannot be empty"
    if len(input_text) > 1000:  # Example limit
        return False, "Context is too long (max 1000 characters)"
    return True, ""

# Start the Streamlit app
if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import os
import requests
import json
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from dotenv import load_dotenv 
import streamlit.components.v1 as components
import pandas as pd

load_dotenv() 

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

def login_page():
    """Display login page and handle authentication"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.markdown("<h1 style='text-align: center;'>Login</h1>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password", autocomplete="off")
            
            if st.button("Login"):
                if check_credentials(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        return False
    return True

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
        "current_captions": [],
        "caption_history": [],
        "num_captions": 1,
        "max_length": 1024,
        "temperature": 0.90,
        "top_k": 50,
        "top_p": 0.90,
        "pending_settings": None,
        "settings_updated": False,
        "is_generating": False,
        "instruction_categories": {
            "Tip me": "Generate a Tip Me Caption",
            "Winner": "Generate a Winner Caption",
            "Holiday": "Generate a Holiday Caption",
            "Bundle": "Generate a Bundle Caption",
            "Descriptive": "Generate a Descriptive Caption",
            "Spin the Wheel": "Generate a Spin the Wheel Caption",
            "Girlfriend Non-Explicit": "Generate a Girlfriend Non-Explicit Caption",
            "Girlfriend Explicits": "Generate a Girlfriend Explicits Caption",
            "List": "Generate a List Caption",
            "Short": "Generate a Short Caption",
            "Sub Promo": "Generate a Sub Promo Caption",
            "VIP": "Generate a VIP Caption"
        }
    }
    
    # Only set defaults if they don't exist in session state
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def add_logout_button():
    """Add a logout button to the bottom of the sidebar"""
    # Create empty space to push the button to the bottom
    st.sidebar.markdown('<div style="height: 0vh;"></div>', unsafe_allow_html=True)
    if st.sidebar.button("Logout"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

def show_generation_page(access_token):
    """Display the main caption generation page"""
    try:
        # Title and welcome message
        st.markdown(
            f"""
            <div style="text-align: center; padding: 2rem 0; margin-bottom: 2rem; background-color: #0E1117;">
                <h1 style="margin: 0; font-size: 2.5rem;">Tasty Caption Generation 🫦</h1>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">Welcome, {st.session_state.username}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        with st.spinner("Loading..."):
            # Show success message if settings were updated
            if st.session_state.get('settings_updated', False):
                st.success("✨ Settings updated successfully!")
                st.session_state.settings_updated = False
            
            # Dropdown with better styling
            selected_category = st.selectbox(
                "Select Caption Category:",
                options=list(st.session_state.instruction_categories.keys()),
                disabled=st.session_state.is_generating,
                key="category_selector"
            )
            
            # Get the corresponding instruction
            instruction = st.session_state.instruction_categories[selected_category]
            banned_words_list = access_banned_words_list()
            
            # Context input with better styling
            input_text = st.text_area(
                "Enter Context:", 
                placeholder="Describe your caption here...",
                disabled=st.session_state.is_generating,
                height=150
            )
            
            # Create a placeholder for captions
            caption_placeholder = st.empty()
            
            # Display existing captions if they exist
            if 'current_captions' in st.session_state and st.session_state.current_captions:
                caption_text = "<div style='background-color: #1E1E1E; padding: 20px; border-radius: 10px; margin: 20px 0;'>"
                for idx, caption in enumerate(st.session_state.current_captions):
                    caption_text += f"<p><strong>Caption {idx + 1}:</strong> {caption}</p>"
                caption_text += "</div>"
                caption_placeholder.markdown(caption_text, unsafe_allow_html=True)
            
            # Generate button with loading state
            if st.button("✨ Generate Captions", use_container_width=True, type="primary"):
                is_valid, error_message = validate_inputs(instruction, input_text)
                if not is_valid:
                    st.error(f"⚠️ {error_message}")
                else:
                    st.session_state.current_captions = []
                    for i in range(st.session_state["num_captions"]):
                        with st.spinner(f"🎯 Generating caption {i + 1}..."):
                            try:
                                valid_caption = False
                                retries = 0
                                max_retries = 3
                                
                                while not valid_caption and retries < max_retries:
                                    response = generate_caption_from_api(
                                        instruction,
                                        input_text,
                                        st.session_state["max_length"],
                                        st.session_state["temperature"],
                                        st.session_state["top_k"],
                                        st.session_state["top_p"],
                                        access_token,
                                    )
                                    
                                    if response:
                                        words_in_response = response.lower().split()
                                        if not any(banned_word in words_in_response for banned_word in banned_words_list):
                                            valid_caption = True
                                        else:
                                            retries += 1
                                            if retries < max_retries:
                                                st.warning(f"⚠️ Attempt {retries}: Generated caption contains banned words. Retrying...")
                                            else:
                                                st.error("❌ Max retries reached. Please try again with different parameters.")
                                                break
                                
                                if valid_caption:
                                    st.session_state.current_captions.append(response)
                                    caption_text = "<div style='background-color: #1E1E1E; padding: 20px; border-radius: 10px; margin: 20px 0;'>"
                                    for idx, caption in enumerate(st.session_state.current_captions):
                                        caption_text += f"<p><strong>Caption {idx + 1}:</strong> {caption}</p>"
                                    caption_text += "</div>"
                                    caption_placeholder.markdown(caption_text, unsafe_allow_html=True)
                            
                            except Exception as e:
                                st.error(f"❌ Error generating caption: {str(e)}")
                                break
                    
                    if st.session_state.current_captions:
                        st.success("✨ Caption generation completed!")
                        
                        # Add to history
                        history_entry = {
                            "instruction": instruction,
                            "context": input_text,
                            "captions": st.session_state.current_captions.copy(),
                            "settings": {
                                "temperature": st.session_state.temperature,
                                "top_k": st.session_state.top_k,
                                "top_p": st.session_state.top_p
                            }
                        }
                        st.session_state.caption_history.insert(0, history_entry)
    
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.error("Please try refreshing the page or contact support if the issue persists.")

def show_history_page():
    """Display the history page"""
    st.markdown(
        "<h1 style='text-align: center;'>Generation History 📜</h1>",
        unsafe_allow_html=True
    )
    
    if not st.session_state.caption_history:
        st.info("No generation history yet")
    else:
        total_entries = len(st.session_state.caption_history)
        for idx, entry in enumerate(st.session_state.caption_history):
            display_num = total_entries - idx
            
            with st.expander(f"Generation {display_num}", expanded=(idx == 0)):
                st.write("**Instruction:**")
                st.write(entry["instruction"])
                st.write("**Context:**")
                st.write(entry["context"])
                st.write("**Settings:**")
                st.write(f"- Temperature: {entry['settings']['temperature']}")
                st.write(f"- Top-k: {entry['settings']['top_k']}")
                st.write(f"- Top-p: {entry['settings']['top_p']}")
                st.write("**Generated Captions:**")
                for i, caption in enumerate(entry["captions"]):
                    st.write(f"*Caption {i + 1}:* {caption}")
                
                # Button to load parameters used
                if st.button("Load Parameters Used", key=f"use_settings_{display_num}"):
                    # Store the settings we want to apply
                    st.session_state.pending_settings = entry["settings"]
                    st.session_state.show_history = False
                    st.session_state.settings_updated = True
                    st.rerun()
                
                # Button to export captions to Excel
                if st.button("Export to Excel", key=f"export_to_excel_{display_num}"):
                    # Create a DataFrame from the captions
                    df = pd.DataFrame({"Captions": entry["captions"]})
                    # Save the DataFrame to an Excel file
                    excel_file = "generated_captions.xlsx"
                    df.to_excel(excel_file, index=False)
                    
                    # Download the Excel file
                    with open(excel_file, "rb") as file:
                        st.download_button(
                            label="Download Captions",
                            data=file,
                            file_name=excel_file,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"download_excel_{display_num}"
                        )

def main():
    # Initialize session state
    initialize_session_state()
    
    # Apply any pending settings before creating widgets
    if st.session_state.pending_settings is not None:
        st.session_state.temperature = st.session_state.pending_settings["temperature"]
        st.session_state.top_k = st.session_state.pending_settings["top_k"]
        st.session_state.top_p = st.session_state.pending_settings["top_p"]
        st.session_state.pending_settings = None  # Clear pending settings
    
    if 'show_history' not in st.session_state:
        st.session_state.show_history = False
    
    # First check login
    if not login_page():
        st.stop()
    
    # Load CSS file
    load_css("styles.css")
    
    # Initialize API credentials
    access_token = initialize_api_credentials()

    # Left sidebar for Generation Settings
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

        # Add some space before the toggle button
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Dynamic button text and styling based on current page
        button_text = "🫦 Generate Captions" if st.session_state.show_history else "📜 View Generation History"
        active_style = "background-color: white; color: black;" if st.session_state.show_history else ""
        
        st.markdown(
            f"""
            <style>
            div.history-button button {{
                width: 100%;
                {active_style}
            }}
            div:not(.history-button) button {{
                background-color: inherit;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Wrap the toggle button in a div with the specific class
        st.markdown('<div class="history-button">', unsafe_allow_html=True)
        if st.button(button_text, 
                    use_container_width=True,
                    key="toggle_page_button",
                    disabled=st.session_state.is_generating):
            st.session_state.show_history = not st.session_state.show_history
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add Logout button at the bottom
        st.markdown('<div style="position: fixed; bottom: 20px; width: 300px;">', unsafe_allow_html=True)
        if st.button("Logout", use_container_width=True):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Main content area - toggle between generation and history
    if st.session_state.show_history:
        show_history_page()
    else:
        show_generation_page(access_token)

# Add this to handle the component value changes
if 'component_value' in st.session_state:
    if isinstance(st.session_state.component_value, dict):
        # Handle settings update
        st.session_state.temperature = st.session_state.component_value['temperature']
        st.session_state.top_k = st.session_state.component_value['top_k']
        st.session_state.top_p = st.session_state.component_value['top_p']
        st.rerun()
    else:
        # Handle modal close
        st.session_state.show_history = st.session_state.component_value

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

def access_banned_words_list():
    df = pd.read_csv(r'assets/banned_words.csv', header=None, names=['word'])
    df['word'] = df['word'].str.lower().str.strip()
    banned_word_list = df['word'].tolist()
    return banned_word_list

# Start the Streamlit app
if __name__ == "__main__":
    # Initialize session state for modal controls
    if 'closeModal' not in st.session_state:
        st.session_state.closeModal = False
    if 'useSettings' not in st.session_state:
        st.session_state.useSettings = None
    main()

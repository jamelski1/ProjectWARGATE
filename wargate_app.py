"""
Project WARGATE - Streamlit UI

A web-based interface for the multi-agent joint staff planning system.

How to run:
    streamlit run wargate_app.py

Requirements:
    pip install streamlit fpdf2

Environment:
    export OPENAI_API_KEY="your-api-key"

Image Files:
    Place the following images in the ./assets/ folder (or root directory):
    - Cyber National Mission Force.png
    - Seal_of_the_United_States_Cyber_Command.svg.png
    - Joint_Chiefs_of_Staff_seal_(2).svg - Copy.png
    - DoW.png
    - DoAF.png
    - DoN.avif
    - Emblem_of_the_U.S._Department_of_the_Army.svg.png

Author: Project WARGATE Team
"""

import os
import re
import streamlit as st
from io import BytesIO
from datetime import datetime
from pathlib import Path

# PDF generation
from fpdf import FPDF

# Import the structured backend
from wargate_backend import run_joint_staff_planning_structured, PlanningResult


# =============================================================================
# ASSET PATHS
# =============================================================================

def get_asset_path(filename: str) -> str | None:
    """
    Find an asset file in either the assets folder or root directory.
    Returns the path if found, None otherwise.
    """
    # Check assets folder first
    assets_path = Path("assets") / filename
    if assets_path.exists():
        return str(assets_path)

    # Check root directory
    root_path = Path(filename)
    if root_path.exists():
        return str(root_path)

    return None


# Logo filenames
LOGO_FILES = [
    "Cyber National Mission Force.png",
    "Seal_of_the_United_States_Cyber_Command.svg.png",
    "Joint_Chiefs_of_Staff_seal_(2).svg - Copy.png",
    "DoW.png",
    "DoAF.png",
    "DoN.avif",
    "Emblem_of_the_U.S._Department_of_the_Army.svg.png",
]

SIDEBAR_LOGO = "Joint_Chiefs_of_Staff_seal_(2).svg - Copy.png"


# =============================================================================
# PDF GENERATION
# =============================================================================

class WARGATEReportPDF(FPDF):
    """Custom PDF class with WARGATE branding and formatting."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        """Add header to each page."""
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, "PROJECT WARGATE - Joint Staff Planning System", align="C")
        self.ln(5)
        self.set_draw_color(200, 200, 200)
        self.line(10, 18, 200, 18)
        self.ln(10)

    def footer(self):
        """Add footer with page numbers."""
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")

    def add_title(self, title: str):
        """Add a main title to the document."""
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(106, 13, 173)  # Joint purple
        self.cell(0, 15, title, ln=True, align="C")
        self.ln(5)

    def add_section_header(self, header: str):
        """Add a section header."""
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(106, 13, 173)  # Joint purple
        self.cell(0, 10, header, ln=True)
        self.set_draw_color(106, 13, 173)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def add_subsection_header(self, header: str):
        """Add a subsection header."""
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(51, 51, 51)
        self.cell(0, 8, header, ln=True)
        self.ln(2)

    def add_body_text(self, text: str):
        """Add body text with proper formatting."""
        self.set_font("Helvetica", "", 10)
        self.set_text_color(0, 0, 0)

        # Clean and process the text first
        text = self._clean_text(text)

        # Split into paragraphs and process
        paragraphs = text.split("\n\n")

        for para in paragraphs:
            if not para.strip():
                continue

            # Check for markdown headers
            if para.startswith("### "):
                self.add_subsection_header(para[4:].strip())
            elif para.startswith("## "):
                self.add_section_header(para[3:].strip())
            elif para.startswith("# "):
                self.add_title(para[2:].strip())
            elif para.startswith("---"):
                self.ln(3)
                self.set_draw_color(200, 200, 200)
                self.line(10, self.get_y(), 200, self.get_y())
                self.ln(5)
            else:
                # Regular paragraph
                lines = para.split("\n")
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Handle bullet points - use calculated width, not 0
                    # Page width is 210mm (A4), with 10mm margins = 190mm usable
                    if line.startswith(("- ", "* ", "• ")):
                        bullet_text = "  * " + line[2:]
                        self.multi_cell(190, 5, bullet_text)
                    elif re.match(r"^\d+\.", line):
                        self.multi_cell(190, 5, "  " + line)
                    else:
                        self.multi_cell(190, 5, line)

                self.ln(3)

    def _clean_text(self, text: str) -> str:
        """Clean text for PDF output - remove problematic characters."""
        # Remove markdown bold/italic markers
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)

        # Replace special characters that might cause issues
        replacements = {
            "\u2019": "'",  # Right single quote
            "\u2018": "'",  # Left single quote
            "\u201c": '"',  # Left double quote
            "\u201d": '"',  # Right double quote
            "\u2014": "-",  # Em dash
            "\u2013": "-",  # En dash
            "\u2022": "-",  # Bullet
            "\u2026": "...",  # Ellipsis
            "\u00a0": " ",  # Non-breaking space
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        # Encode to latin-1 compatible characters
        text = text.encode("latin-1", errors="replace").decode("latin-1")

        return text


def generate_pdf_from_text(title: str, content: str, include_scenario: str = None) -> BytesIO:
    """
    Generate a PDF document from title and content.

    Args:
        title: The main title for the document
        content: The body content (markdown-like text)
        include_scenario: Optional scenario text to include at the start

    Returns:
        BytesIO buffer containing the PDF data
    """
    pdf = WARGATEReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # Add main title
    pdf.add_title(title)
    pdf.ln(5)

    # Add scenario if provided
    if include_scenario:
        pdf.add_section_header("Scenario")
        pdf.add_body_text(include_scenario)
        pdf.ln(5)

    # Add main content
    pdf.add_body_text(content)

    # Generate to BytesIO
    buffer = BytesIO()
    pdf_output = pdf.output()
    buffer.write(pdf_output)
    buffer.seek(0)

    return buffer


def generate_full_report_pdf(result: PlanningResult, scenario: str) -> BytesIO:
    """Generate a comprehensive PDF with all planning phases."""
    pdf = WARGATEReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # Title page
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(106, 13, 173)  # Joint purple
    pdf.ln(40)
    pdf.cell(0, 20, "PROJECT WARGATE", ln=True, align="C")
    pdf.set_font("Helvetica", "", 16)
    pdf.cell(0, 10, "Joint Staff Planning Product", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.cell(0, 10, "Classification: UNCLASSIFIED // FOR EXERCISE PURPOSES ONLY", ln=True, align="C")

    # Scenario
    pdf.add_page()
    pdf.add_section_header("Strategic Scenario")
    pdf.add_body_text(scenario)

    # Intelligence Estimate
    pdf.add_page()
    pdf.add_section_header("J2 - Intelligence Estimate")
    pdf.add_body_text(result["intel_estimate"])

    # COA Development
    pdf.add_page()
    pdf.add_section_header("J5/J3 - COA Development")
    pdf.add_body_text(result["coa_development"])

    # Staff Estimates
    pdf.add_page()
    pdf.add_section_header("Functional Staff Estimates (J1/J4/J6/Cyber/Fires/ENG/Protection)")
    pdf.add_body_text(result["staff_estimates"])

    # Legal/Ethics
    pdf.add_page()
    pdf.add_section_header("SJA - Legal & Ethics Review")
    pdf.add_body_text(result["legal_ethics"])

    # Commander's Brief
    pdf.add_page()
    pdf.add_section_header("Commander's Brief")
    pdf.add_body_text(result["commander_brief"])

    # Generate to BytesIO
    buffer = BytesIO()
    pdf_output = pdf.output()
    buffer.write(pdf_output)
    buffer.seek(0)

    return buffer


# =============================================================================
# STREAMLIT UI
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "planning_result" not in st.session_state:
        st.session_state.planning_result = None
    if "scenario_text" not in st.session_state:
        st.session_state.scenario_text = ""
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "current_step" not in st.session_state:
        st.session_state.current_step = ""
    if "progress" not in st.session_state:
        st.session_state.progress = 0.0


def render_logo_strip():
    """Render the horizontal logo strip at the top of the page."""
    # Create columns for logos
    cols = st.columns(len(LOGO_FILES))

    for i, (col, logo_file) in enumerate(zip(cols, LOGO_FILES)):
        with col:
            logo_path = get_asset_path(logo_file)
            if logo_path:
                st.image(logo_path, use_container_width=True)
            else:
                # Placeholder if image not found
                st.caption(f"[{logo_file.split('.')[0][:10]}...]")

    # Horizontal rule below logos
    st.markdown("---")


def render_sidebar():
    """Render the sidebar with controls."""
    with st.sidebar:
        # Sidebar logo - Joint Chiefs of Staff seal
        sidebar_logo_path = get_asset_path(SIDEBAR_LOGO)
        if sidebar_logo_path:
            st.image(sidebar_logo_path, use_container_width=True)
        else:
            st.markdown("### Joint Chiefs of Staff")

        st.title("WARGATE Controls")

        st.markdown("---")

        # Model configuration
        st.subheader("Model Settings")

        model_name = st.text_input(
            "Model Name",
            value="gpt-4.1",
            help="OpenAI model to use (e.g., gpt-4.1, gpt-4o, gpt-4-turbo)"
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values = more creative, lower = more focused"
        )

        persona_seed = st.number_input(
            "Persona Seed (optional)",
            min_value=0,
            max_value=99999,
            value=0,
            help="Set a seed for reproducible staff personas (0 = random)"
        )

        st.markdown("---")

        # Scenario input
        st.subheader("Scenario Input")

        input_method = st.radio(
            "Input Method",
            ["Text Input", "File Upload"],
            horizontal=True
        )

        if input_method == "Text Input":
            scenario = st.text_area(
                "Enter Scenario",
                height=200,
                placeholder="Paste your operational scenario here...\n\nInclude:\n- Background/context\n- Current situation\n- Key indicators\n- Planning task",
                value=st.session_state.scenario_text
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload Scenario File",
                type=["txt", "md"],
                help="Upload a .txt or .md file containing the scenario"
            )
            if uploaded_file:
                scenario = uploaded_file.read().decode("utf-8")
                st.text_area("Preview", scenario, height=150, disabled=True)
            else:
                scenario = ""

        st.markdown("---")

        # Run button
        col1, col2 = st.columns(2)

        with col1:
            run_clicked = st.button(
                "Run Planning",
                type="primary",
                disabled=st.session_state.is_running,
                use_container_width=True
            )

        with col2:
            if st.button("Clear", use_container_width=True):
                st.session_state.planning_result = None
                st.session_state.scenario_text = ""
                st.rerun()

        # API key status
        st.markdown("---")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            st.success("API Key configured")
        else:
            st.error("OPENAI_API_KEY not set")
            st.caption("Set via: `export OPENAI_API_KEY='...'`")

        return {
            "model_name": model_name,
            "temperature": temperature,
            "persona_seed": persona_seed if persona_seed > 0 else None,
            "scenario": scenario,
            "run_clicked": run_clicked
        }


def render_welcome():
    """Render the welcome/landing page."""
    st.markdown("""
    ## Welcome to Project WARGATE

    **WARGATE** (War Gaming and Analysis for Responsive Tactical Engagement) is a
    multi-agent AI system that simulates joint military staff planning processes.

    ### How It Works

    The system models a complete **Joint Staff** with specialized AI agents:

    | Role | Function |
    |------|----------|
    | **J2** | Intelligence analysis and threat assessment |
    | **J3** | Operations planning and execution details |
    | **J5** | Strategic planning and COA development |
    | **J1, J4, J6** | Personnel, Logistics, Communications |
    | **Cyber/EW** | Cyber and electronic warfare integration |
    | **Fires** | Joint fires and targeting coordination |
    | **SJA** | Legal and ethics review |
    | **Commander** | Final synthesis and decision |

    ### Getting Started

    1. **Enter a Scenario** in the sidebar (or upload a file)
    2. **Configure** model settings (optional)
    3. **Click "Run Planning"** to start the multi-agent process
    4. **Review** the structured outputs by phase
    5. **Export** any section to PDF

    ---

    #### Example Scenario Format

    ```
    Background: [Strategic context and history]

    Situation: [Current events and indicators]
    - Indicator 1
    - Indicator 2
    - Indicator 3

    Task: [What the planning team needs to accomplish]
    ```

    ---

    *Paste your scenario in the sidebar and click **Run Planning** to begin.*
    """)


def render_results(result: PlanningResult, scenario: str):
    """Render the planning results in tabs."""

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Intel Estimate", f"{len(result['intel_estimate']):,} chars")
    with col2:
        st.metric("COA Development", f"{len(result['coa_development']):,} chars")
    with col3:
        st.metric("Staff Estimates", f"{len(result['staff_estimates']):,} chars")
    with col4:
        st.metric("Full Report", f"{len(result['full_report']):,} chars")

    st.markdown("---")

    # Tabs for each phase - JPP-aligned names
    tabs = st.tabs([
        "J2 – Intelligence Estimate",
        "J5/J3 – COA Development",
        "Functional Staff Estimates (J1/J4/J6/Cyber/Fires/ENG/Protection)",
        "SJA – Legal & Ethics Review",
        "Commander's Brief",
        "Full Joint Planning Report"
    ])

    # J2 Intel Tab
    with tabs[0]:
        st.subheader("J2 – Intelligence Estimate")
        st.markdown("""
        The J2 (Intelligence) provides the threat assessment, enemy courses of action,
        and intelligence gaps that inform the planning process.
        """)

        with st.expander("View Full Intelligence Estimate", expanded=True):
            st.markdown(result["intel_estimate"])

        pdf_buffer = generate_pdf_from_text(
            "J2 - Intelligence Estimate",
            result["intel_estimate"],
            include_scenario=scenario
        )
        st.download_button(
            label="Download J2 Intel PDF",
            data=pdf_buffer,
            file_name="j2_intel_estimate.pdf",
            mime="application/pdf"
        )

    # COA Development Tab
    with tabs[1]:
        st.subheader("J5/J3 – COA Development")
        st.markdown("""
        The J5 (Plans) develops strategic approaches, while J3 (Operations)
        refines them into executable courses of action with detailed phasing.
        """)

        with st.expander("View COA Development", expanded=True):
            st.markdown(result["coa_development"])

        pdf_buffer = generate_pdf_from_text(
            "J5/J3 - COA Development",
            result["coa_development"],
            include_scenario=scenario
        )
        st.download_button(
            label="Download COA PDF",
            data=pdf_buffer,
            file_name="coa_development.pdf",
            mime="application/pdf"
        )

    # Staff Estimates Tab
    with tabs[2]:
        st.subheader("Functional Staff Estimates (J1/J4/J6/Cyber/Fires/ENG/Protection)")
        st.markdown("""
        Each functional staff section provides their assessment of feasibility,
        risks, and requirements for the proposed courses of action.
        """)

        with st.expander("View Staff Estimates", expanded=True):
            st.markdown(result["staff_estimates"])

        pdf_buffer = generate_pdf_from_text(
            "Functional Staff Estimates",
            result["staff_estimates"],
            include_scenario=scenario
        )
        st.download_button(
            label="Download Staff Estimates PDF",
            data=pdf_buffer,
            file_name="staff_estimates.pdf",
            mime="application/pdf"
        )

    # Legal & Ethics Tab
    with tabs[3]:
        st.subheader("SJA – Legal & Ethics Review")
        st.markdown("""
        The Staff Judge Advocate reviews all courses of action for compliance
        with law of armed conflict, rules of engagement, and ethical considerations.
        """)

        with st.expander("View Legal/Ethics Review", expanded=True):
            st.markdown(result["legal_ethics"])

        pdf_buffer = generate_pdf_from_text(
            "SJA - Legal and Ethics Review",
            result["legal_ethics"],
            include_scenario=scenario
        )
        st.download_button(
            label="Download Legal Review PDF",
            data=pdf_buffer,
            file_name="legal_ethics_review.pdf",
            mime="application/pdf"
        )

    # Commander's Brief Tab
    with tabs[4]:
        st.subheader("Commander's Brief")
        st.markdown("""
        The synthesis of all staff inputs into the commander's decision products:
        COA comparison, recommended COA, and Commander's Intent.
        """)

        with st.expander("View Commander's Brief", expanded=True):
            st.markdown(result["commander_brief"])

        pdf_buffer = generate_pdf_from_text(
            "Commander's Brief",
            result["commander_brief"],
            include_scenario=scenario
        )
        st.download_button(
            label="Download Commander's Brief PDF",
            data=pdf_buffer,
            file_name="commanders_brief.pdf",
            mime="application/pdf"
        )

    # Full Report Tab
    with tabs[5]:
        st.subheader("Full Joint Planning Report")
        st.markdown("""
        The complete, concatenated planning product containing all phases
        and supporting analysis.
        """)

        with st.expander("View Full Report", expanded=False):
            st.text(result["full_report"])

        # Full report PDF
        pdf_buffer = generate_full_report_pdf(result, scenario)
        st.download_button(
            label="Download Complete Report PDF",
            data=pdf_buffer,
            file_name="wargate_full_report.pdf",
            mime="application/pdf",
            type="primary"
        )


def run_planning_with_progress(scenario: str, model_name: str, temperature: float, persona_seed: int | None):
    """Run the planning process with progress updates."""

    progress_bar = st.progress(0, text="Initializing...")
    status_text = st.empty()

    def update_progress(step_name: str, step_num: int, total_steps: int):
        progress = step_num / total_steps
        progress_bar.progress(progress, text=f"Step {step_num}/{total_steps}: {step_name}")
        status_text.info(f"Processing: {step_name}...")

    try:
        result = run_joint_staff_planning_structured(
            scenario_text=scenario,
            model_name=model_name,
            temperature=temperature,
            verbose=False,
            persona_seed=persona_seed,
            progress_callback=update_progress
        )

        progress_bar.progress(1.0, text="Complete!")
        status_text.success("Planning complete!")

        return result

    except Exception as e:
        progress_bar.empty()
        status_text.error(f"Error: {str(e)}")
        st.exception(e)
        return None


def main():
    """Main Streamlit application."""

    # Page configuration
    st.set_page_config(
        page_title="Project WARGATE - Joint Staff Planning",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for Joint Staff theme
    st.markdown("""
    <style>
    /* Main theme overrides */
    .stApp {
        background-color: #FFFFFF;
    }

    /* Tab styling - Joint purple theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #101018;
        padding: 10px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 16px;
        background-color: #1a1a2e;
        border-radius: 5px;
        color: #FFFFFF;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6A0DAD;
        color: #FFFFFF;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #4a0a7a;
    }

    /* Expander styling */
    .stExpander {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }

    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: #6A0DAD;
        font-weight: bold;
    }

    /* Button styling */
    .stButton > button[kind="primary"] {
        background-color: #6A0DAD;
        border-color: #6A0DAD;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #4a0a7a;
        border-color: #4a0a7a;
    }

    /* Download button styling */
    .stDownloadButton > button {
        background-color: #101018;
        color: #FFFFFF;
        border: 1px solid #6A0DAD;
    }
    .stDownloadButton > button:hover {
        background-color: #6A0DAD;
        border-color: #6A0DAD;
    }

    /* Header styling */
    h1, h2, h3 {
        color: #101018;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #101018;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #FFFFFF;
    }

    /* Progress bar */
    .stProgress > div > div {
        background-color: #6A0DAD;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    init_session_state()

    # Render logo strip at top
    render_logo_strip()

    # Header
    st.title("Project WARGATE")
    st.caption("Joint Staff Planning Interface | Multi-Agent AI Planning System")

    # Render sidebar and get inputs
    inputs = render_sidebar()

    # Handle run button
    if inputs["run_clicked"]:
        if not inputs["scenario"]:
            st.warning("Please enter a scenario before running the planning process.")
        elif not os.environ.get("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY environment variable is not set.")
        else:
            st.session_state.scenario_text = inputs["scenario"]
            st.session_state.is_running = True

            with st.status("Project WARGATE is generating staff products...", expanded=True, state="running") as status:
                st.write("Initializing joint staff agents...")
                result = run_planning_with_progress(
                    scenario=inputs["scenario"],
                    model_name=inputs["model_name"],
                    temperature=inputs["temperature"],
                    persona_seed=inputs["persona_seed"]
                )
                if result:
                    status.update(label="Planning complete!", state="complete", expanded=False)
                else:
                    status.update(label="Planning failed", state="error", expanded=True)

            st.session_state.is_running = False

            if result:
                st.session_state.planning_result = result
                st.rerun()

    # Main content area
    if st.session_state.planning_result:
        render_results(st.session_state.planning_result, st.session_state.scenario_text)
    else:
        render_welcome()


if __name__ == "__main__":
    main()

"""
Project WARGATE - Streamlit UI v2.0

A professional government-style interface for the multi-agent joint staff planning system.
Implements the full 7-step Joint Planning Process (JPP) with interactive dialogue,
PDF slide generation, and sequential phase execution.

How to run:
    streamlit run wargate_app.py

Requirements:
    pip install streamlit fpdf2

Environment:
    export OPENAI_API_KEY="your-api-key"

Image Files (place in ./assets/ folder):
    - WARGATE_logo.png (main logo)
    - Cyber National Mission Force.png
    - Seal_of_the_United_States_Cyber_Command.svg.png
    - Joint_Chiefs_of_Staff_seal_(2).svg - Copy.png
    - DoW.png, DoAF.png, DoN.avif
    - Emblem_of_the_U.S._Department_of_the_Army.svg.png

Author: Project WARGATE Team
"""

from __future__ import annotations

import os
import re
import time
import streamlit as st
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from dataclasses import dataclass
from enum import Enum

# PDF generation
from fpdf import FPDF

# Import the structured backend
from wargate_backend import run_joint_staff_planning_structured, PlanningResult


# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

# Logo filenames
WARGATE_LOGO = "WARGATE_logo.png"
SIDEBAR_LOGO = "Joint_Chiefs_of_Staff_seal_(2).svg - Copy.png"

LOGO_FILES = [
    "Cyber National Mission Force.png",
    "Seal_of_the_United_States_Cyber_Command.svg.png",
    "Joint_Chiefs_of_Staff_seal_(2).svg - Copy.png",
    "DoW.png",
    "DoAF.png",
    "DoN.avif",
    "Emblem_of_the_U.S._Department_of_the_Army.svg.png",
]

# Military branch colors for dialogue bubbles
BRANCH_COLORS = {
    "US Army": "#4B5320",       # Army Green
    "US Navy": "#000080",       # Navy Blue
    "US Air Force": "#00308F",  # Air Force Blue
    "US Marine Corps": "#8B0000", # Marine Scarlet
    "US Space Force": "#1C1C1C", # Space Force Black
    "US Coast Guard": "#FF6600", # Coast Guard Orange
    "Joint": "#6A0DAD",         # Joint Purple
}

# Staff role display names
STAFF_ROLE_DISPLAY = {
    "commander": ("Commander", "CMDR"),
    "j1_personnel": ("J1 - Personnel", "J1"),
    "j2_intelligence": ("J2 - Intelligence", "J2"),
    "j3_operations": ("J3 - Operations", "J3"),
    "j4_logistics": ("J4 - Logistics", "J4"),
    "j5_plans": ("J5 - Plans", "J5"),
    "j6_communications": ("J6 - Communications", "J6"),
    "j7_training": ("J7 - Training", "J7"),
    "j8_resources": ("J8 - Resources", "J8"),
    "cyber_ew_oic": ("Cyber/EW", "CYBER"),
    "fires_oic": ("Fires", "FIRES"),
    "engineer_oic": ("Engineer", "ENG"),
    "protection_oic": ("Protection", "PROT"),
    "sja_legal": ("SJA - Legal", "SJA"),
    "pao_io": ("PAO/IO", "PAO"),
}


# =============================================================================
# CUSTOM CSS - GOVERNMENT STYLE TEMPLATE
# =============================================================================

GOVERNMENT_CSS = """
<style>
/* =================================================================
   WARGATE GOVERNMENT-STYLE CSS TEMPLATE
   Professional U.S. Government Interface Design
   ================================================================= */

/* CSS Variables for Light/Dark Mode */
:root {
    --primary-purple: #6A0DAD;
    --primary-purple-dark: #4a0a7a;
    --primary-purple-light: #8B5CF6;
    --bg-primary: #FFFFFF;
    --bg-secondary: #F5F5F5;
    --bg-tertiary: #E8E8E8;
    --text-primary: #1a1a1a;
    --text-secondary: #4a4a4a;
    --text-muted: #6b6b6b;
    --border-color: #d0d0d0;
    --shadow-color: rgba(0, 0, 0, 0.1);
    --sidebar-bg: #101018;
    --sidebar-text: #FFFFFF;
    --accent-gold: #C5A572;
}

/* Dark mode detection and overrides */
@media (prefers-color-scheme: dark) {
    :root {
        --bg-primary: #1a1a2e;
        --bg-secondary: #16213e;
        --bg-tertiary: #0f3460;
        --text-primary: #FFFFFF;
        --text-secondary: #E0E0E0;
        --text-muted: #B0B0B0;
        --border-color: #3a3a5a;
        --shadow-color: rgba(0, 0, 0, 0.3);
    }
}

/* =================================================================
   GLOBAL RESETS & BASE STYLES
   ================================================================= */

/* Remove Streamlit branding and default styles */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
    background-color: var(--bg-primary);
    font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
}

/* Force text color for visibility */
.stApp, .stApp p, .stApp span, .stApp div, .stApp label {
    color: var(--text-primary) !important;
}

/* Main content area */
.main .block-container {
    padding: 1rem 2rem 2rem 2rem;
    max-width: 100%;
}

/* =================================================================
   TYPOGRAPHY
   ================================================================= */

h1, h2, h3, h4, h5, h6 {
    font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
    font-weight: 600;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
}

h1 {
    font-size: 2.25rem;
    border-bottom: 3px solid var(--primary-purple);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

h2 {
    font-size: 1.75rem;
    border-bottom: 2px solid var(--primary-purple);
    padding-bottom: 0.4rem;
    margin-bottom: 0.8rem;
}

h3 {
    font-size: 1.35rem;
    color: var(--primary-purple) !important;
}

/* =================================================================
   HEADER BAR & LOGO AREA
   ================================================================= */

.wargate-header {
    background: linear-gradient(135deg, #101018 0%, #1a1a2e 50%, #101018 100%);
    padding: 1rem 2rem;
    margin: -1rem -2rem 1.5rem -2rem;
    border-bottom: 3px solid var(--primary-purple);
    box-shadow: 0 4px 12px var(--shadow-color);
}

.wargate-header-content {
    display: flex;
    align-items: center;
    gap: 1.5rem;
}

.wargate-logo {
    height: 80px;
    width: auto;
}

.wargate-title {
    color: #FFFFFF !important;
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}

.wargate-subtitle {
    color: var(--accent-gold) !important;
    font-size: 1rem;
    font-weight: 400;
    margin: 0;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* =================================================================
   SIDEBAR STYLING - High Contrast Accessible Theme
   ================================================================= */

/* Base sidebar container */
section[data-testid="stSidebar"] {
    background-color: #050509 !important;
    border-right: 2px solid var(--primary-purple);
}

/* Force ALL sidebar text to be light/readable */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] * {
    color: #F5F5F5 !important;
}

/* Sidebar headings - pure white, bold */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #FFFFFF !important;
    font-weight: 700;
    border-bottom: 2px solid var(--primary-purple);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* All input labels - light gray for readability */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stTextArea label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stFileUploader label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    color: #F5F5F5 !important;
    font-weight: 500;
}

/* Radio button options and checkbox text */
section[data-testid="stSidebar"] .stRadio > div,
section[data-testid="stSidebar"] .stRadio label span,
section[data-testid="stSidebar"] .stCheckbox label span,
section[data-testid="stSidebar"] [data-baseweb="radio"] label {
    color: #F5F5F5 !important;
}

/* Help text / small description text */
section[data-testid="stSidebar"] .stTooltipIcon,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] .caption,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] .stMarkdown p {
    color: #CCCCCC !important;
}

/* Input fields - dark background with light text */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stTextArea textarea,
section[data-testid="stSidebar"] .stNumberInput input {
    background-color: #1a1a2e !important;
    color: #F5F5F5 !important;
    border: 1px solid #3a3a5a !important;
}

/* Placeholder text - visible but subtle */
section[data-testid="stSidebar"] input::placeholder,
section[data-testid="stSidebar"] textarea::placeholder {
    color: #AAAAAA !important;
    opacity: 1 !important;
}

/* Selectbox dropdown */
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background-color: #1a1a2e !important;
    color: #F5F5F5 !important;
    border: 1px solid #3a3a5a !important;
}

/* Slider styling */
section[data-testid="stSidebar"] .stSlider > div > div > div {
    background-color: var(--primary-purple) !important;
}

section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
    color: #CCCCCC !important;
}

/* Buttons in sidebar - high contrast */
section[data-testid="stSidebar"] .stButton > button {
    background-color: var(--primary-purple) !important;
    color: #FFFFFF !important;
    border: none !important;
    font-weight: 600;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #8B5CF6 !important;
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"] .stButton > button:disabled {
    background-color: #333333 !important;
    color: #AAAAAA !important;
    cursor: not-allowed;
}

/* Secondary/outline buttons */
section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    background-color: #1a1a2e !important;
    color: #F5F5F5 !important;
    border: 2px solid var(--primary-purple) !important;
}

section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
    background-color: var(--primary-purple) !important;
    color: #FFFFFF !important;
}

/* Success/error messages in sidebar */
section[data-testid="stSidebar"] .stSuccess,
section[data-testid="stSidebar"] [data-testid="stNotification"] {
    background-color: #1a3a1a !important;
    color: #90EE90 !important;
}

section[data-testid="stSidebar"] .stError {
    background-color: #3a1a1a !important;
    color: #FF6B6B !important;
}

/* File uploader in sidebar */
section[data-testid="stSidebar"] .stFileUploader > div {
    background-color: #1a1a2e !important;
    border: 1px dashed #3a3a5a !important;
}

section[data-testid="stSidebar"] .stFileUploader > div:hover {
    border-color: var(--primary-purple) !important;
}

/* Horizontal rules in sidebar - purple accent */
section[data-testid="stSidebar"] hr {
    border-color: var(--primary-purple) !important;
    opacity: 0.5;
}

/* =================================================================
   TAB STYLING - Sharp Government Look
   ================================================================= */

.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background-color: var(--bg-tertiary);
    padding: 0;
    border-radius: 0;
    border-bottom: 2px solid var(--primary-purple);
}

.stTabs [data-baseweb="tab"] {
    padding: 12px 24px;
    background-color: var(--bg-secondary);
    border-radius: 0;
    border: 1px solid var(--border-color);
    border-bottom: none;
    color: var(--text-primary) !important;
    font-weight: 500;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-right: -1px;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: var(--primary-purple-light);
    color: #FFFFFF !important;
}

.stTabs [aria-selected="true"] {
    background-color: var(--primary-purple) !important;
    color: #FFFFFF !important;
    border-color: var(--primary-purple);
    font-weight: 600;
}

/* =================================================================
   BUTTON STYLING
   ================================================================= */

.stButton > button {
    background-color: var(--primary-purple);
    color: #FFFFFF !important;
    border: none;
    border-radius: 0;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    box-shadow: 0 2px 4px var(--shadow-color);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background-color: var(--primary-purple-dark);
    box-shadow: 0 4px 8px var(--shadow-color);
    transform: translateY(-1px);
}

.stButton > button[kind="secondary"] {
    background-color: var(--bg-secondary);
    color: var(--text-primary) !important;
    border: 2px solid var(--primary-purple);
}

.stButton > button[kind="secondary"]:hover {
    background-color: var(--primary-purple);
    color: #FFFFFF !important;
}

/* Download buttons */
.stDownloadButton > button {
    background-color: #1a1a2e;
    color: #FFFFFF !important;
    border: 2px solid var(--primary-purple);
    border-radius: 0;
}

.stDownloadButton > button:hover {
    background-color: var(--primary-purple);
}

/* =================================================================
   EXPANDER STYLING
   ================================================================= */

.stExpander {
    background-color: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 0;
    box-shadow: 0 2px 4px var(--shadow-color);
}

.stExpander > div:first-child {
    background-color: var(--bg-tertiary);
    border-bottom: 1px solid var(--border-color);
}

.stExpander [data-testid="stExpanderToggleIcon"] {
    color: var(--primary-purple) !important;
}

/* =================================================================
   DIALOGUE BUBBLE STYLING
   ================================================================= */

.dialogue-container {
    margin: 1rem 0;
    padding: 0;
}

.dialogue-bubble {
    background-color: var(--bg-secondary);
    border-radius: 0;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
    border-left: 4px solid var(--primary-purple);
    box-shadow: 0 2px 6px var(--shadow-color);
    position: relative;
}

.dialogue-bubble.army { border-left-color: #4B5320; }
.dialogue-bubble.navy { border-left-color: #000080; }
.dialogue-bubble.air-force { border-left-color: #00308F; }
.dialogue-bubble.marine-corps { border-left-color: #8B0000; }
.dialogue-bubble.space-force { border-left-color: #1C1C1C; }
.dialogue-bubble.coast-guard { border-left-color: #FF6600; }
.dialogue-bubble.joint { border-left-color: #6A0DAD; }

.dialogue-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-color);
}

.dialogue-badge {
    background-color: var(--primary-purple);
    color: #FFFFFF;
    padding: 0.25rem 0.5rem;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.dialogue-badge.army { background-color: #4B5320; }
.dialogue-badge.navy { background-color: #000080; }
.dialogue-badge.air-force { background-color: #00308F; }
.dialogue-badge.marine-corps { background-color: #8B0000; }
.dialogue-badge.space-force { background-color: #1C1C1C; }
.dialogue-badge.coast-guard { background-color: #FF6600; }
.dialogue-badge.joint { background-color: #6A0DAD; }

.dialogue-rank {
    font-weight: 600;
    color: var(--text-primary);
    font-size: 0.95rem;
}

.dialogue-role {
    color: var(--text-muted);
    font-size: 0.85rem;
    font-style: italic;
}

.dialogue-content {
    color: var(--text-primary);
    line-height: 1.6;
    font-size: 0.95rem;
}

.dialogue-content p {
    margin: 0.5rem 0;
}

/* Commander bubble - special styling */
.dialogue-bubble.commander {
    background-color: #f8f4ff;
    border-left-color: var(--accent-gold);
    border-left-width: 6px;
}

.dialogue-badge.commander {
    background-color: var(--accent-gold);
    color: #1a1a1a;
}

/* =================================================================
   PHASE/STEP SECTION STYLING
   ================================================================= */

.phase-container {
    background-color: var(--bg-secondary);
    border: 1px solid var(--border-color);
    margin: 1.5rem 0;
    box-shadow: 0 4px 12px var(--shadow-color);
}

.phase-header {
    background: linear-gradient(135deg, var(--primary-purple) 0%, var(--primary-purple-dark) 100%);
    color: #FFFFFF !important;
    padding: 1rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}

.phase-number {
    background-color: rgba(255, 255, 255, 0.2);
    padding: 0.5rem 1rem;
    font-size: 1.5rem;
    font-weight: 700;
}

.phase-title {
    font-size: 1.25rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.phase-content {
    padding: 1.5rem;
}

.substep-container {
    border-left: 3px solid var(--border-color);
    margin-left: 1rem;
    padding-left: 1.5rem;
    margin-bottom: 1rem;
}

.substep-header {
    color: var(--primary-purple);
    font-weight: 600;
    font-size: 1rem;
    margin-bottom: 0.5rem;
}

/* =================================================================
   METRICS & INFO BOXES
   ================================================================= */

[data-testid="stMetricValue"] {
    color: var(--primary-purple) !important;
    font-weight: 700;
    font-size: 1.5rem;
}

[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
}

.info-box {
    background-color: var(--bg-tertiary);
    border-left: 4px solid var(--primary-purple);
    padding: 1rem;
    margin: 1rem 0;
}

.warning-box {
    background-color: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 1rem;
    margin: 1rem 0;
    color: #1a1a1a !important;
}

.success-box {
    background-color: #d4edda;
    border-left: 4px solid #28a745;
    padding: 1rem;
    margin: 1rem 0;
    color: #1a1a1a !important;
}

/* =================================================================
   SCROLLBAR STYLING
   ================================================================= */

::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: var(--bg-tertiary);
}

::-webkit-scrollbar-thumb {
    background: var(--primary-purple);
    border-radius: 0;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--primary-purple-dark);
}

/* =================================================================
   PROGRESS INDICATORS
   ================================================================= */

.stProgress > div > div {
    background-color: var(--primary-purple);
    border-radius: 0;
}

.stSpinner > div {
    border-top-color: var(--primary-purple) !important;
}

/* =================================================================
   RESPONSIVE DESIGN
   ================================================================= */

@media screen and (max-width: 1200px) {
    .wargate-title {
        font-size: 2rem;
    }

    .wargate-logo {
        height: 60px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 16px;
        font-size: 0.8rem;
    }
}

@media screen and (max-width: 768px) {
    .wargate-header-content {
        flex-direction: column;
        text-align: center;
    }

    .dialogue-header {
        flex-direction: column;
        align-items: flex-start;
    }
}

/* =================================================================
   PDF PREVIEW AREA
   ================================================================= */

.pdf-preview {
    background-color: #FFFFFF;
    color: #1a1a1a !important;
    padding: 2rem;
    border: 1px solid var(--border-color);
    box-shadow: 0 4px 12px var(--shadow-color);
    max-height: 600px;
    overflow-y: auto;
}

.pdf-preview * {
    color: #1a1a1a !important;
}

/* Force dark text in markdown content areas */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {
    color: var(--text-primary) !important;
}

/* Ensure text areas have proper contrast */
.stTextArea textarea {
    color: var(--text-primary) !important;
    background-color: var(--bg-primary) !important;
}

.stTextInput input {
    color: var(--text-primary) !important;
    background-color: var(--bg-primary) !important;
}

</style>
"""


# =============================================================================
# ASSET MANAGEMENT
# =============================================================================

def get_asset_path(filename: str) -> str | None:
    """Find an asset file in either the assets folder or root directory."""
    assets_path = Path("assets") / filename
    if assets_path.exists():
        return str(assets_path)

    root_path = Path(filename)
    if root_path.exists():
        return str(root_path)

    return None


# =============================================================================
# AGENT PERSONA DATA STRUCTURE
# =============================================================================

@dataclass
class AgentPersona:
    """Represents a staff agent's persona for dialogue display."""
    role: str
    role_short: str
    rank: str
    name: str
    branch: str

    @property
    def full_designation(self) -> str:
        return f"{self.rank} {self.name}"

    @property
    def branch_class(self) -> str:
        """Get CSS class for branch color."""
        branch_map = {
            "US Army": "army",
            "US Navy": "navy",
            "US Air Force": "air-force",
            "US Marine Corps": "marine-corps",
            "US Space Force": "space-force",
            "US Coast Guard": "coast-guard",
        }
        return branch_map.get(self.branch, "joint")


# Default personas (will be replaced by actual agent personas when available)
DEFAULT_PERSONAS = {
    "commander": AgentPersona("Commander", "CMDR", "LTG", "Williams", "US Army"),
    "j2_intelligence": AgentPersona("J2 - Intelligence", "J2", "COL", "Chen", "US Air Force"),
    "j3_operations": AgentPersona("J3 - Operations", "J3", "COL", "Martinez", "US Marine Corps"),
    "j5_plans": AgentPersona("J5 - Plans", "J5", "COL", "Johnson", "US Army"),
    "j1_personnel": AgentPersona("J1 - Personnel", "J1", "COL", "Davis", "US Navy"),
    "j4_logistics": AgentPersona("J4 - Logistics", "J4", "COL", "Thompson", "US Army"),
    "j6_communications": AgentPersona("J6 - Communications", "J6", "COL", "Park", "US Space Force"),
    "cyber_ew_oic": AgentPersona("Cyber/EW", "CYBER", "COL", "Nakamura", "US Air Force"),
    "fires_oic": AgentPersona("Fires", "FIRES", "COL", "O'Brien", "US Army"),
    "engineer_oic": AgentPersona("Engineer", "ENG", "COL", "Patel", "US Army"),
    "protection_oic": AgentPersona("Protection", "PROT", "COL", "Rodriguez", "US Marine Corps"),
    "sja_legal": AgentPersona("SJA - Legal", "SJA", "COL", "Washington", "US Navy"),
    "pao_io": AgentPersona("PAO/IO", "PAO", "COL", "Lee", "US Air Force"),
}


# =============================================================================
# DIALOGUE BUBBLE COMPONENT
# =============================================================================

def render_dialogue_bubble(
    persona: AgentPersona,
    content: str,
    is_commander: bool = False
) -> None:
    """
    Render an agent dialogue bubble with rank, service, and role information.

    Args:
        persona: The agent's persona information
        content: The dialogue content (markdown supported)
        is_commander: Whether this is a commander bubble (special styling)
    """
    branch_class = persona.branch_class
    commander_class = "commander" if is_commander else ""

    html = f"""
    <div class="dialogue-bubble {branch_class} {commander_class}">
        <div class="dialogue-header">
            <span class="dialogue-badge {branch_class}">{persona.branch}</span>
            <span class="dialogue-rank">{persona.full_designation}</span>
            <span class="dialogue-role">{persona.role}</span>
        </div>
        <div class="dialogue-content">
            {content}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_dialogue_sequence(dialogues: list[tuple[str, str]]) -> None:
    """
    Render a sequence of dialogue bubbles.

    Args:
        dialogues: List of (role_key, content) tuples
    """
    st.markdown('<div class="dialogue-container">', unsafe_allow_html=True)

    for role_key, content in dialogues:
        persona = DEFAULT_PERSONAS.get(role_key, DEFAULT_PERSONAS["commander"])
        is_commander = role_key == "commander"
        render_dialogue_bubble(persona, content, is_commander)

    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# PDF SLIDE DECK GENERATION
# =============================================================================

class WARGATESlidePDF(FPDF):
    """Custom PDF class for WARGATE slide decks with government styling."""

    def __init__(self, title: str = "WARGATE Planning Product"):
        super().__init__(orientation='L')  # Landscape for slides
        self.slide_title = title
        self.slide_num = 0
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        """Slide header with classification banner."""
        # Classification banner
        self.set_fill_color(106, 13, 173)  # Purple
        self.rect(0, 0, 297, 8, 'F')
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(255, 255, 255)
        self.set_xy(0, 1)
        self.cell(297, 6, "UNCLASSIFIED // FOR EXERCISE PURPOSES ONLY", align="C")

        # Logo area and title
        self.set_xy(10, 12)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(106, 13, 173)
        self.cell(0, 8, "PROJECT WARGATE", ln=True)

        # Purple line
        self.set_draw_color(106, 13, 173)
        self.set_line_width(0.5)
        self.line(10, 22, 287, 22)
        self.ln(8)

    def footer(self):
        """Slide footer with page number."""
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Slide {self.page_no()}", align="R")

    def add_slide(self, title: str, content: str = "", bullets: list[str] = None):
        """Add a new slide with title and content."""
        self.add_page()
        self.slide_num += 1

        # Slide title
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(26, 26, 46)
        self.cell(0, 12, title, ln=True)
        self.ln(5)

        # Content
        self.set_font("Helvetica", "", 12)
        self.set_text_color(0, 0, 0)

        if content:
            # Clean content for PDF
            clean_content = self._clean_text(content)
            self.multi_cell(0, 7, clean_content)
            self.ln(5)

        if bullets:
            for bullet in bullets:
                clean_bullet = self._clean_text(bullet)
                self.set_x(20)
                self.multi_cell(257, 7, f"* {clean_bullet}")
                self.ln(2)

    def add_title_slide(self, main_title: str, subtitle: str = ""):
        """Add a title slide."""
        self.add_page()

        # Center content vertically
        self.ln(50)

        # Main title
        self.set_font("Helvetica", "B", 32)
        self.set_text_color(106, 13, 173)
        self.cell(0, 20, main_title, align="C", ln=True)

        # Subtitle
        if subtitle:
            self.set_font("Helvetica", "", 18)
            self.set_text_color(100, 100, 100)
            self.cell(0, 12, subtitle, align="C", ln=True)

        # Date
        self.ln(20)
        self.set_font("Helvetica", "I", 12)
        self.cell(0, 8, datetime.now().strftime("%d %B %Y"), align="C")

    def _clean_text(self, text: str) -> str:
        """Clean text for PDF output."""
        # Remove markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        text = re.sub(r'#+\s*', '', text)

        # Replace special characters
        replacements = {
            '\u2019': "'", '\u2018': "'",
            '\u201c': '"', '\u201d': '"',
            '\u2014': '-', '\u2013': '-',
            '\u2022': '*', '\u2026': '...',
            '\u00a0': ' ',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        return text.encode('latin-1', errors='replace').decode('latin-1')


def generate_phase_slides(
    phase_name: str,
    phase_number: int,
    content_sections: dict[str, str | list[str]]
) -> BytesIO:
    """
    Generate a PDF slide deck for a specific JPP phase.

    Args:
        phase_name: Name of the phase (e.g., "Mission Analysis")
        phase_number: Phase number (1-7)
        content_sections: Dict of section_title -> content (str or list of bullets)

    Returns:
        BytesIO buffer containing the PDF
    """
    pdf = WARGATESlidePDF(f"Phase {phase_number}: {phase_name}")

    # Title slide
    pdf.add_title_slide(
        f"STEP {phase_number}: {phase_name.upper()}",
        "Joint Planning Process"
    )

    # Content slides
    for section_title, content in content_sections.items():
        if isinstance(content, list):
            pdf.add_slide(section_title, bullets=content)
        else:
            pdf.add_slide(section_title, content=content)

    # Generate to BytesIO
    buffer = BytesIO()
    pdf_output = pdf.output()
    buffer.write(pdf_output)
    buffer.seek(0)

    return buffer


# =============================================================================
# JPP PHASE DEFINITIONS
# =============================================================================

class JPPPhase(Enum):
    """Joint Planning Process phases."""
    PLANNING_INITIATION = 1
    MISSION_ANALYSIS = 2
    COA_DEVELOPMENT = 3
    COA_ANALYSIS = 4
    COA_COMPARISON = 5
    COA_APPROVAL = 6
    PLAN_DEVELOPMENT = 7


JPP_PHASE_INFO = {
    JPPPhase.PLANNING_INITIATION: {
        "name": "Planning Initiation",
        "description": "Establish planning organization, develop initial staff estimates, and receive commander's initial guidance.",
        "outputs": ["Strategic Guidance Summary", "Problem Framing", "Planning Constraints/Restraints", "Initial CCIRs", "Key Assumptions"],
    },
    JPPPhase.MISSION_ANALYSIS: {
        "name": "Mission Analysis",
        "description": "Analyze the mission, develop facts and assumptions, and produce the restated mission.",
        "outputs": ["METT-TC Analysis", "Problem Statement", "Restated Mission", "CCIRs", "Assumptions"],
    },
    JPPPhase.COA_DEVELOPMENT: {
        "name": "COA Development",
        "description": "Develop multiple courses of action that are suitable, feasible, acceptable, distinguishable, and complete.",
        "outputs": ["COA Statements", "COA Sketches", "COA Comparison Criteria", "Initial Risk Assessment"],
    },
    JPPPhase.COA_ANALYSIS: {
        "name": "COA Analysis & Wargaming",
        "description": "Wargame each COA against enemy COAs to identify strengths, weaknesses, and required modifications.",
        "outputs": ["Wargame Results", "Decision Points", "Critical Events", "Modified COAs"],
    },
    JPPPhase.COA_COMPARISON: {
        "name": "COA Comparison",
        "description": "Compare COAs against evaluation criteria to identify the preferred COA.",
        "outputs": ["Comparison Matrix", "Advantages/Disadvantages", "Risk Comparison", "Staff Recommendation"],
    },
    JPPPhase.COA_APPROVAL: {
        "name": "COA Approval",
        "description": "Present COAs to commander for decision and approval of the selected COA.",
        "outputs": ["Decision Brief", "Commander's Decision", "Refined Commander's Intent", "Planning Guidance"],
    },
    JPPPhase.PLAN_DEVELOPMENT: {
        "name": "Plan/Order Development",
        "description": "Develop the detailed plan or order based on the approved COA.",
        "outputs": ["Draft OPORD/OPLAN", "Annexes", "Synchronization Matrix", "Execution Timeline"],
    },
}


# =============================================================================
# SESSION STATE MANAGEMENT
# =============================================================================

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "planning_result": None,
        "scenario_text": "",
        "is_running": False,
        "current_phase": None,
        "phase_outputs": {},
        "dialogue_history": [],
        "pdf_slides": {},
        "model_name": "gpt-4.1",
        "temperature": 0.7,
        "persona_seed": 0,
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_header():
    """Render the main header with WARGATE logo."""
    logo_path = get_asset_path(WARGATE_LOGO)

    st.markdown('<div class="wargate-header">', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 5])

    with col1:
        if logo_path:
            st.image(logo_path, width=100)
        else:
            st.markdown("**[WARGATE]**")

    with col2:
        st.markdown("""
        <div class="wargate-header-content">
            <div>
                <h1 class="wargate-title" style="color: white !important; margin: 0; border: none;">Project WARGATE</h1>
                <p class="wargate-subtitle" style="color: #C5A572 !important;">Joint Staff Planning Interface | Multi-Agent AI System</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_logo_strip():
    """Render the horizontal logo strip."""
    cols = st.columns(len(LOGO_FILES))

    for col, logo_file in zip(cols, LOGO_FILES):
        with col:
            logo_path = get_asset_path(logo_file)
            if logo_path:
                st.image(logo_path, use_container_width=True)


def render_sidebar():
    """Render the sidebar with controls."""
    with st.sidebar:
        # Sidebar logo
        logo_path = get_asset_path(WARGATE_LOGO)
        if logo_path:
            st.image(logo_path, width=150)

        sidebar_seal = get_asset_path(SIDEBAR_LOGO)
        if sidebar_seal:
            st.image(sidebar_seal, width=100)

        st.title("WARGATE Controls")
        st.markdown("---")

        # Model settings
        st.subheader("Model Settings")

        model_name = st.text_input(
            "Model Name",
            value=st.session_state.model_name,
            help="OpenAI model (e.g., gpt-4.1, gpt-4o)"
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
        )

        persona_seed = st.number_input(
            "Persona Seed",
            min_value=0,
            max_value=99999,
            value=st.session_state.persona_seed,
        )

        st.markdown("---")

        # Scenario input
        st.subheader("Scenario Input")

        input_method = st.radio("Input Method", ["Text Input", "File Upload"], horizontal=True)

        if input_method == "Text Input":
            scenario = st.text_area(
                "Enter Scenario",
                height=200,
                placeholder="Paste your operational scenario here...",
                value=st.session_state.scenario_text
            )
        else:
            uploaded_file = st.file_uploader("Upload Scenario", type=["txt", "md"])
            if uploaded_file:
                scenario = uploaded_file.read().decode("utf-8")
                st.text_area("Preview", scenario, height=100, disabled=True)
            else:
                scenario = ""

        st.markdown("---")

        # Action buttons
        col1, col2 = st.columns(2)

        with col1:
            run_clicked = st.button(
                "RUN PLANNING",
                type="primary",
                disabled=st.session_state.is_running,
                use_container_width=True
            )

        with col2:
            if st.button("RESET", use_container_width=True):
                for key in ["planning_result", "scenario_text", "phase_outputs",
                           "dialogue_history", "pdf_slides", "current_phase"]:
                    st.session_state[key] = None if key != "phase_outputs" else {}
                st.rerun()

        # API status
        st.markdown("---")
        if os.environ.get("OPENAI_API_KEY"):
            st.success("API Key: Configured")
        else:
            st.error("API Key: Not Set")

        return {
            "model_name": model_name,
            "temperature": temperature,
            "persona_seed": persona_seed if persona_seed > 0 else None,
            "scenario": scenario,
            "run_clicked": run_clicked,
        }


def render_phase_section(
    phase: JPPPhase,
    phase_info: dict,
    is_current: bool = False,
    is_complete: bool = False
):
    """Render a JPP phase section with dialogue and PDF outputs."""
    phase_num = phase.value
    phase_name = phase_info["name"]

    # Phase header
    status_icon = "🔄" if is_current else ("✅" if is_complete else "⏳")

    with st.expander(f"STEP {phase_num}: {phase_name.upper()} {status_icon}", expanded=is_current):
        st.markdown(f"**Description:** {phase_info['description']}")

        if is_complete and phase.name in st.session_state.phase_outputs:
            output = st.session_state.phase_outputs[phase.name]

            # Sub-tabs for each sub-step
            tabs = st.tabs(["Staff Meeting", "Slides", "Brief Commander", "Commander Guidance"])

            with tabs[0]:
                st.subheader("Staff Meeting Dialogue")
                if "dialogues" in output:
                    for role, content in output["dialogues"]:
                        persona = DEFAULT_PERSONAS.get(role, DEFAULT_PERSONAS["commander"])
                        render_dialogue_bubble(persona, content, role == "commander")

            with tabs[1]:
                st.subheader("Planning Slides")
                if "pdf" in output:
                    st.download_button(
                        f"Download {phase_name} Slides (PDF)",
                        data=output["pdf"],
                        file_name=f"wargate_step{phase_num}_{phase_name.lower().replace(' ', '_')}.pdf",
                        mime="application/pdf"
                    )
                    st.info("PDF slide deck generated. Click to download.")

            with tabs[2]:
                st.subheader("Briefing to Commander")
                if "brief" in output:
                    for role, content in output["brief"]:
                        persona = DEFAULT_PERSONAS.get(role, DEFAULT_PERSONAS["commander"])
                        render_dialogue_bubble(persona, content, role == "commander")

            with tabs[3]:
                st.subheader("Commander's Guidance")
                if "guidance" in output:
                    render_dialogue_bubble(
                        DEFAULT_PERSONAS["commander"],
                        output["guidance"],
                        is_commander=True
                    )

        elif is_current:
            st.info("🔄 This phase is currently being processed...")
            st.progress(0.5)

        else:
            st.markdown("*Awaiting completion of previous phases*")

        # Next step button
        if is_complete and phase.value < 7:
            if st.button(f"PROCEED TO STEP {phase.value + 1}", key=f"next_{phase.value}"):
                st.session_state.current_phase = JPPPhase(phase.value + 1)
                st.rerun()


def render_welcome():
    """Render the welcome/landing page."""
    st.markdown("""
    ## Welcome to Project WARGATE

    **WARGATE** (War Gaming and Analysis for Responsive Tactical Engagement) is a
    multi-agent AI system that executes the full **Joint Planning Process (JPP)**.

    ### The 7-Step Joint Planning Process

    This system guides you through each phase with:
    - **Interactive Staff Dialogues** - Watch agents collaborate in real-time
    - **PDF Slide Decks** - Exportable briefing materials for each phase
    - **Commander Briefs** - Staff presentations to the commander
    - **Commander Guidance** - Direction back to the staff

    | Step | Phase | Key Outputs |
    |------|-------|-------------|
    | 1 | Planning Initiation | Strategic guidance, problem framing |
    | 2 | Mission Analysis | METT-TC, restated mission, CCIRs |
    | 3 | COA Development | Multiple courses of action |
    | 4 | COA Analysis | Wargaming results, decision points |
    | 5 | COA Comparison | Comparison matrix, recommendation |
    | 6 | COA Approval | Commander's decision |
    | 7 | Plan Development | Draft OPORD/OPLAN |

    ---

    ### Getting Started

    1. **Enter a Scenario** in the sidebar
    2. **Click "RUN PLANNING"** to begin
    3. **Review** each phase as it completes
    4. **Download** PDF slides for any phase
    5. **Proceed** through all 7 steps

    ---

    *Enter your scenario in the sidebar and click **RUN PLANNING** to begin the Joint Planning Process.*
    """)


def render_pipeline_instructions():
    """Render the pipeline instructions and control flow diagram."""
    st.markdown("## Pipeline Instructions")
    st.markdown("""
    This tab provides an overview of the WARGATE agent pipeline control flow
    and explains how the 7-step Joint Planning Process is executed.
    """)

    st.markdown("### Control Flow Diagram")

    # Render the ASCII diagram in a code block for proper formatting
    st.code("""
WARGATE AGENT PIPELINE CONTROL FLOW
====================================

                    ┌─────────────────────┐
                    │   SCENARIO INPUT    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  PLANNING INITIATION │ ◄── Step 1
                    │  (1a) Staff Meeting  │
                    │  (1b) Slides PDF     │
                    │  (1c) Brief CMDR     │
                    │  (1d) CMDR Guidance  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  MISSION ANALYSIS    │ ◄── Step 2
                    │  (2a) Staff Meeting  │
                    │  (2b) Slides PDF     │
                    │  (2c) Brief CMDR     │
                    │  (2d) CMDR Guidance  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  COA DEVELOPMENT     │ ◄── Step 3
                    │  (3a) Brainstorming  │
                    │  (3b) COA Slides PDF │
                    │  (3c) Brief CMDR     │
                    │  (3d) CMDR Guidance  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  COA ANALYSIS        │ ◄── Step 4
                    │  (4a) Wargame Mtg    │
                    │  (4b) Wargame PDF    │
                    │  (4c) Brief CMDR     │
                    │  (4d) CMDR Guidance  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  COA COMPARISON      │ ◄── Step 5
                    │  (5a) Compare Mtg    │
                    │  (5b) Compare PDF    │
                    │  (5c) Brief CMDR     │
                    │  (5d) CMDR Selection │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  COA APPROVAL        │ ◄── Step 6
                    │  (6a) Final Coord    │
                    │  (6b) Approval PDF   │
                    │  (6c) Decision Brief │
                    │  (6d) CMDR Approval  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  PLAN DEVELOPMENT    │ ◄── Step 7
                    │  (7a) PLANDEV Work   │
                    │  (7b) OPORD PDF      │
                    │  (7c) Brief CMDR     │
                    │  (7d) CMDR Approval  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   FINAL OUTPUT       │
                    │  Complete Planning   │
                    │  Product (PDF)       │
                    └─────────────────────┘
    """, language=None)

    st.markdown("### Dialogue Bubble Flow (Per Phase)")

    st.code("""
DIALOGUE BUBBLE FLOW (per phase):
─────────────────────────────────
  Staff Agent 1 ──► speaks
  Staff Agent 2 ──► responds
  Staff Agent N ──► contributes
       │
       ▼
  [PDF Slides Generated]
       │
       ▼
  Staff ──► Brief Commander
       │
       ▼
  Commander ──► Issues Guidance
       │
       ▼
  [NEXT STEP Button]
    """, language=None)

    st.markdown("### Phase Sub-Steps Explained")

    st.markdown("""
    Each of the 7 JPP phases follows the same sub-step structure:

    | Sub-Step | Description |
    |----------|-------------|
    | **(a) Staff Meeting** | Staff agents discuss and collaborate in dialogue bubbles |
    | **(b) Slides PDF** | System generates exportable slide deck for the phase |
    | **(c) Brief Commander** | Staff presents findings to the commander |
    | **(d) Commander Guidance** | Commander issues direction for the next phase |

    After completing all sub-steps, click **"PROCEED TO NEXT STEP"** to advance.
    """)


def render_planning_dashboard():
    """Render the main planning dashboard with all phases."""
    st.markdown("## Joint Planning Process Dashboard")

    # Progress overview
    completed_phases = len([p for p in JPPPhase if p.name in st.session_state.phase_outputs])
    st.progress(completed_phases / 7, text=f"Progress: {completed_phases}/7 phases complete")

    # Render each phase
    for phase in JPPPhase:
        phase_info = JPP_PHASE_INFO[phase]
        is_current = st.session_state.current_phase == phase
        is_complete = phase.name in st.session_state.phase_outputs

        render_phase_section(phase, phase_info, is_current, is_complete)

    # Final report section
    if completed_phases == 7:
        st.markdown("---")
        st.markdown("## Final Planning Product")
        st.success("All JPP phases complete! Download the full planning product below.")

        if st.session_state.planning_result:
            result = st.session_state.planning_result

            tabs = st.tabs(["Summary", "Full Report", "Export All"])

            with tabs[0]:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Intel Estimate", f"{len(result.get('intel_estimate', '')):,} chars")
                with col2:
                    st.metric("COA Development", f"{len(result.get('coa_development', '')):,} chars")
                with col3:
                    st.metric("Staff Estimates", f"{len(result.get('staff_estimates', '')):,} chars")
                with col4:
                    st.metric("Full Report", f"{len(result.get('full_report', '')):,} chars")

            with tabs[1]:
                with st.expander("View Full Report", expanded=False):
                    st.text(result.get("full_report", "Report not available"))

            with tabs[2]:
                st.download_button(
                    "DOWNLOAD COMPLETE PLANNING PRODUCT (PDF)",
                    data=generate_full_report_pdf(result),
                    file_name="wargate_complete_planning_product.pdf",
                    mime="application/pdf",
                    type="primary"
                )


def generate_full_report_pdf(result: dict) -> BytesIO:
    """Generate the complete planning product PDF."""
    pdf = WARGATESlidePDF("Complete Planning Product")

    # Title slide
    pdf.add_title_slide("PROJECT WARGATE", "Complete Joint Planning Product")

    # Add sections for each major output
    sections = [
        ("Intelligence Estimate (J2)", result.get("intel_estimate", "")),
        ("COA Development (J5/J3)", result.get("coa_development", "")),
        ("Staff Estimates", result.get("staff_estimates", "")),
        ("Legal & Ethics Review (SJA)", result.get("legal_ethics", "")),
        ("Commander's Brief", result.get("commander_brief", "")),
    ]

    for title, content in sections:
        if content:
            pdf.add_slide(title, content=content[:2000])  # Truncate for slides

    buffer = BytesIO()
    pdf_output = pdf.output()
    buffer.write(pdf_output)
    buffer.seek(0)

    return buffer


# =============================================================================
# PLANNING EXECUTION
# =============================================================================

def execute_planning_phase(
    phase: JPPPhase,
    scenario: str,
    model_name: str,
    temperature: float,
    persona_seed: int | None
) -> dict:
    """
    Execute a single JPP phase and return outputs.

    Returns dict with: dialogues, pdf, brief, guidance
    """
    phase_info = JPP_PHASE_INFO[phase]

    # Simulate phase execution (in real implementation, this calls the backend)
    # For now, generate placeholder content

    dialogues = []
    brief = []

    # Generate staff meeting dialogue
    if phase == JPPPhase.PLANNING_INITIATION:
        dialogues = [
            ("j5_plans", f"<p>Sir, we've received the strategic guidance. Let me outline the planning parameters for {phase_info['name']}.</p><p>Key constraints include timeline and available forces.</p>"),
            ("j2_intelligence", "<p>I'll provide initial threat assessment. We're seeing indicators of adversary activity in the region.</p>"),
            ("j3_operations", "<p>Roger. I'm coordinating with component commands for initial force laydown options.</p>"),
        ]
        brief = [
            ("j5_plans", "<p>Commander, we've completed initial planning setup. Key assumptions are documented.</p>"),
            ("j2_intelligence", "<p>Initial threat picture shows moderate risk in the AO.</p>"),
        ]
        guidance = "<p>Proceed with mission analysis. I want options that minimize risk to civilians while achieving objectives. Keep me informed of any significant intelligence updates.</p>"

    else:
        # Generic phase output
        dialogues = [
            ("j5_plans", f"<p>Initiating {phase_info['name']} phase. Staff sections, provide your inputs.</p>"),
            ("j3_operations", f"<p>Operations assessment for {phase_info['name']} is underway.</p>"),
            ("j2_intelligence", "<p>Intel update: Situation remains consistent with previous assessment.</p>"),
        ]
        brief = [
            ("j5_plans", f"<p>Commander, {phase_info['name']} phase complete. Key outputs are ready for review.</p>"),
        ]
        guidance = f"<p>Good work on {phase_info['name']}. Proceed to the next phase. Continue coordinating across staff sections.</p>"

    # Generate PDF slides
    pdf_content = generate_phase_slides(
        phase_info["name"],
        phase.value,
        {output: f"Content for {output}" for output in phase_info["outputs"]}
    )

    return {
        "dialogues": dialogues,
        "pdf": pdf_content,
        "brief": brief,
        "guidance": guidance,
    }


def run_full_planning(scenario: str, model_name: str, temperature: float, persona_seed: int | None):
    """Run the full planning process through all phases."""

    # Run backend planning
    progress_bar = st.progress(0, text="Initializing planning process...")
    status = st.empty()

    def update_progress(step_name: str, step_num: int, total_steps: int):
        progress = step_num / total_steps
        progress_bar.progress(progress, text=f"Step {step_num}/{total_steps}: {step_name}")
        status.info(f"Processing: {step_name}...")

    try:
        # Run the backend planning
        result = run_joint_staff_planning_structured(
            scenario_text=scenario,
            model_name=model_name,
            temperature=temperature,
            verbose=False,
            persona_seed=persona_seed,
            progress_callback=update_progress
        )

        st.session_state.planning_result = result

        # Generate phase outputs from result
        for phase in JPPPhase:
            phase_output = execute_planning_phase(
                phase, scenario, model_name, temperature, persona_seed
            )
            st.session_state.phase_outputs[phase.name] = phase_output

        progress_bar.progress(1.0, text="Planning complete!")
        status.success("All phases complete!")

        return True

    except Exception as e:
        progress_bar.empty()
        status.error(f"Error: {str(e)}")
        st.exception(e)
        return False


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main Streamlit application entry point."""

    # Page configuration
    st.set_page_config(
        page_title="Project WARGATE - Joint Staff Planning",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject custom CSS
    st.markdown(GOVERNMENT_CSS, unsafe_allow_html=True)

    # Initialize session state
    init_session_state()

    # Render header
    render_header()

    # Logo strip
    render_logo_strip()
    st.markdown("---")

    # Sidebar
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
            st.session_state.current_phase = JPPPhase.PLANNING_INITIATION

            success = run_full_planning(
                scenario=inputs["scenario"],
                model_name=inputs["model_name"],
                temperature=inputs["temperature"],
                persona_seed=inputs["persona_seed"]
            )

            st.session_state.is_running = False

            if success:
                st.rerun()

    # Main content with tabs
    # Note: "Planning Outputs" tab comes first so it's the default active tab
    if st.session_state.phase_outputs:
        # After a run, show planning outputs as default with instructions in second tab
        planning_tab, instructions_tab = st.tabs(["Planning Outputs", "Pipeline Instructions"])

        with planning_tab:
            render_planning_dashboard()

        with instructions_tab:
            render_pipeline_instructions()
    else:
        # Before a run, show welcome page with instructions in second tab
        welcome_tab, instructions_tab = st.tabs(["Getting Started", "Pipeline Instructions"])

        with welcome_tab:
            render_welcome()

        with instructions_tab:
            render_pipeline_instructions()


# =============================================================================
# CONTROL FLOW DIAGRAM (ASCII)
# =============================================================================

"""
WARGATE AGENT PIPELINE CONTROL FLOW
====================================

                    ┌─────────────────────┐
                    │   SCENARIO INPUT    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  PLANNING INITIATION │ ◄── Step 1
                    │  (1a) Staff Meeting  │
                    │  (1b) Slides PDF     │
                    │  (1c) Brief CMDR     │
                    │  (1d) CMDR Guidance  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  MISSION ANALYSIS    │ ◄── Step 2
                    │  (2a) Staff Meeting  │
                    │  (2b) Slides PDF     │
                    │  (2c) Brief CMDR     │
                    │  (2d) CMDR Guidance  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  COA DEVELOPMENT     │ ◄── Step 3
                    │  (3a) Brainstorming  │
                    │  (3b) COA Slides PDF │
                    │  (3c) Brief CMDR     │
                    │  (3d) CMDR Guidance  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  COA ANALYSIS        │ ◄── Step 4
                    │  (4a) Wargame Mtg    │
                    │  (4b) Wargame PDF    │
                    │  (4c) Brief CMDR     │
                    │  (4d) CMDR Guidance  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  COA COMPARISON      │ ◄── Step 5
                    │  (5a) Compare Mtg    │
                    │  (5b) Compare PDF    │
                    │  (5c) Brief CMDR     │
                    │  (5d) CMDR Selection │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  COA APPROVAL        │ ◄── Step 6
                    │  (6a) Final Coord    │
                    │  (6b) Approval PDF   │
                    │  (6c) Decision Brief │
                    │  (6d) CMDR Approval  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  PLAN DEVELOPMENT    │ ◄── Step 7
                    │  (7a) PLANDEV Work   │
                    │  (7b) OPORD PDF      │
                    │  (7c) Brief CMDR     │
                    │  (7d) CMDR Approval  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   FINAL OUTPUT       │
                    │  Complete Planning   │
                    │  Product (PDF)       │
                    └─────────────────────┘

DIALOGUE BUBBLE FLOW (per phase):
─────────────────────────────────
  Staff Agent 1 ──► speaks
  Staff Agent 2 ──► responds
  Staff Agent N ──► contributes
       │
       ▼
  [PDF Slides Generated]
       │
       ▼
  Staff ──► Brief Commander
       │
       ▼
  Commander ──► Issues Guidance
       │
       ▼
  [NEXT STEP Button]
"""


if __name__ == "__main__":
    main()

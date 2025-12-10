"""
Project WARGATE - Streamlit UI v3.0

A professional government-style interface for the multi-agent joint staff planning system.
Implements the full 7-step Joint Planning Process (JPP) with:
- RICH MULTI-AGENT DIALOGUE: 10+ agents participate with 2-3+ turns each
- LIVE INCREMENTAL RENDERING: Conversations unfold in real-time
- 4-STEP MEETING FLOW: Staff Meeting -> Slides -> Brief Commander -> Guidance

How to run:
    streamlit run wargate_app.py

Requirements:
    pip install streamlit fpdf2 langchain langchain-openai

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
from typing import Any, Callable, TypedDict
from dataclasses import dataclass
from enum import Enum

# PDF generation
from fpdf import FPDF

# Import the structured backend (legacy)
from wargate_backend import run_joint_staff_planning_structured, PlanningResult

# Import the new multi-agent orchestration system
from wargate_orchestration import (
    MeetingOrchestrator,
    DialogueTurn,
    MeetingResult,
    SlideContent,
    BriefResult,
    GuidanceResult,
    PhaseResult,
    JPPPhase as OrchestratorJPPPhase,
    PHASE_CONFIGS,
    create_orchestrator,
)
from wargate import WARGATEConfig


# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

# Logo filenames
WARGATE_LOGO = "WARGATE_logo.png"
SIDEBAR_LOGO = "Joint_Chiefs_of_Staff_seal_(2).svg - Copy.png"

LOGO_FILES = [
    "CNMF Seal.png",
    "USCC Seal.png",
    "JCOS Seal.png",
    "DoW Seal.png",
    "DoAF Seal.png",
    "DoN Seal.png",
    "DoA Seal.png",
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

/* Scrollable dialogue area - prevents page from growing infinitely */
.dialogue-scroll-area {
    max-height: 500px;
    overflow-y: auto;
    padding-right: 0.5rem;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    background-color: var(--bg-primary);
}

/* Custom scrollbar styling for dialogue area */
.dialogue-scroll-area::-webkit-scrollbar {
    width: 8px;
}

.dialogue-scroll-area::-webkit-scrollbar-track {
    background: var(--bg-secondary);
    border-radius: 4px;
}

.dialogue-scroll-area::-webkit-scrollbar-thumb {
    background-color: var(--primary-purple);
    border-radius: 4px;
}

.dialogue-scroll-area::-webkit-scrollbar-thumb:hover {
    background-color: #5a0b9d;
}

/* Current phase banner */
.current-phase-banner {
    background: linear-gradient(135deg, var(--primary-purple), #8b1fd4);
    color: #FFFFFF;
    padding: 0.75rem 1.25rem;
    border-radius: 0;
    margin-bottom: 1rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    box-shadow: 0 2px 8px rgba(106, 13, 173, 0.3);
}

.current-phase-banner .phase-indicator {
    background-color: rgba(255,255,255,0.2);
    padding: 0.25rem 0.5rem;
    border-radius: 3px;
    font-size: 0.85rem;
}

.current-phase-banner .phase-name {
    font-size: 1.1rem;
}

/* Step container styling */
.step-container {
    border: 1px solid var(--border-color);
    border-radius: 0;
    margin-bottom: 1rem;
    background-color: var(--bg-secondary);
}

.step-container.current {
    border-color: var(--primary-purple);
    border-width: 2px;
    box-shadow: 0 0 12px rgba(106, 13, 173, 0.2);
}

.step-container.completed {
    opacity: 0.9;
}

.step-header {
    background-color: var(--bg-tertiary);
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.step-header .step-title {
    font-weight: 600;
    color: var(--text-primary);
}

.step-header .step-status {
    font-size: 0.85rem;
    color: var(--text-muted);
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
   MISSION TERMINAL STYLE STATUS BAR
   ================================================================= */

.mission-terminal {
    background-color: #0a0a12;
    border: 2px solid var(--primary-purple);
    border-radius: 4px;
    padding: 1rem 1.5rem;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    color: #00ff00;
    margin: 1rem 0;
    box-shadow: 0 0 20px rgba(106, 13, 173, 0.3), inset 0 0 20px rgba(0, 0, 0, 0.5);
}

.mission-terminal-header {
    color: var(--primary-purple-light);
    font-size: 0.8rem;
    margin-bottom: 0.5rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-bottom: 1px solid #333;
    padding-bottom: 0.5rem;
}

.mission-terminal-progress {
    font-size: 1.1rem;
    letter-spacing: 0.05em;
    margin: 0.5rem 0;
}

.mission-terminal-progress .completed { color: #00ff00; }
.mission-terminal-progress .remaining { color: #333; }
.mission-terminal-progress .cursor {
    animation: blink 1s step-end infinite;
    color: #00ff00;
}

.mission-terminal-status {
    color: #00ccff;
    font-size: 0.9rem;
    margin-top: 0.5rem;
}

@keyframes blink {
    50% { opacity: 0; }
}

/* =================================================================
   MICRO-PROGRESS STATUS MESSAGES
   ================================================================= */

.micro-progress {
    background: linear-gradient(135deg, #0a0a12 0%, #1a1a2e 100%);
    border-left: 3px solid var(--primary-purple);
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    color: #00ccff;
    font-size: 0.9rem;
}

.micro-progress-icon {
    display: inline-block;
    width: 20px;
    animation: pulse 1.5s ease-in-out infinite;
}

.micro-progress-text {
    color: #e0e0e0;
}

@keyframes pulse {
    0%, 100% { opacity: 0.4; }
    50% { opacity: 1; }
}

/* =================================================================
   TYPING INDICATOR ANIMATION
   ================================================================= */

.typing-indicator {
    background-color: var(--bg-secondary);
    border-radius: 0;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
    border-left: 4px solid var(--primary-purple);
    opacity: 0.8;
}

.typing-indicator-content {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--text-muted);
    font-style: italic;
}

.typing-indicator-dots {
    display: flex;
    gap: 4px;
}

.typing-indicator-dots span {
    width: 8px;
    height: 8px;
    background-color: var(--primary-purple);
    border-radius: 50%;
    animation: typing-bounce 1.4s ease-in-out infinite;
}

.typing-indicator-dots span:nth-child(1) { animation-delay: 0s; }
.typing-indicator-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator-dots span:nth-child(3) { animation-delay: 0.4s; }

@keyframes typing-bounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.3; }
    30% { transform: translateY(-4px); opacity: 1; }
}

/* =================================================================
   TRANSCRIPT ARCHIVE STYLING
   ================================================================= */

.transcript-archive {
    background-color: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    margin-top: 1rem;
}

.transcript-archive-header {
    background-color: var(--bg-tertiary);
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--border-color);
    font-weight: 600;
    color: var(--primary-purple);
}

.transcript-phase {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--border-color);
}

.transcript-phase:last-child {
    border-bottom: none;
}

.transcript-phase-title {
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}

.transcript-turn {
    padding: 0.5rem 0;
    border-top: 1px dashed var(--border-color);
    font-size: 0.9rem;
}

.transcript-turn:first-child {
    border-top: none;
}

.transcript-speaker {
    color: var(--primary-purple);
    font-weight: 600;
}

/* Enhanced dialogue scroll area with auto-scroll anchor */
.dialogue-scroll-area {
    max-height: 500px;
    overflow-y: auto;
    padding-right: 0.5rem;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    background-color: var(--bg-primary);
    scroll-behavior: smooth;
}

.dialogue-scroll-anchor {
    height: 1px;
    visibility: hidden;
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


def split_first_sentence(text: str) -> tuple[str, str]:
    """
    Split text into the first sentence (summary) and the rest (body).

    The first sentence is identified by the first occurrence of '.', '!', or '?'
    followed by a space or end of text. This allows agents' summary sentences
    to be rendered in bold for easy skimming.

    Args:
        text: The full dialogue text

    Returns:
        Tuple of (first_sentence, rest_of_text)

    Examples:
        >>> split_first_sentence("We need more intel. Here's why...")
        ("We need more intel.", "Here's why...")

        >>> split_first_sentence("Short statement.")
        ("Short statement.", "")
    """
    if not text:
        return ("", "")

    text = text.strip()

    # Look for sentence-ending punctuation followed by space or end
    import re
    # Match first sentence ending with . ! or ? followed by space or end
    match = re.search(r'^(.+?[.!?])(?:\s+|$)', text)

    if match:
        first_sentence = match.group(1).strip()
        rest = text[match.end():].strip()
        return (first_sentence, rest)
    else:
        # No clear sentence break found, return whole text as summary
        return (text, "")


def render_dialogue_bubble_from_turn(turn: DialogueTurn) -> None:
    """
    Render a dialogue bubble directly from a DialogueTurn object.

    This is the preferred method for rendering dialogue from the new
    orchestration system, as it uses the DialogueTurn type directly.

    The first sentence of each turn is rendered in bold as a summary
    to allow users to quickly skim conversations.

    Args:
        turn: DialogueTurn object from the orchestrator
    """
    # Map branch to CSS class
    branch_class_map = {
        "US Army": "army",
        "US Navy": "navy",
        "US Air Force": "air-force",
        "US Marine Corps": "marine-corps",
        "US Space Force": "space-force",
        "US Coast Guard": "coast-guard",
    }
    branch_class = branch_class_map.get(turn.get("branch", ""), "joint")
    commander_class = "commander" if turn.get("is_commander", False) else ""

    # Get and format text content with bold first sentence
    raw_text = turn.get("text", "")
    summary, body = split_first_sentence(raw_text)

    # Build content with bold summary and regular body
    if summary and body:
        content = f"<p><strong>{summary}</strong></p><p>{body}</p>"
    elif summary:
        content = f"<p><strong>{summary}</strong></p>"
    else:
        content = f"<p>{raw_text}</p>"

    html = f"""
    <div class="dialogue-bubble {branch_class} {commander_class}">
        <div class="dialogue-header">
            <span class="dialogue-badge {branch_class}">{turn.get('branch', 'Joint')}</span>
            <span class="dialogue-rank">{turn.get('rank', '')} {turn.get('speaker', 'Unknown').replace(turn.get('rank', ''), '').strip()}</span>
            <span class="dialogue-role">{turn.get('role_display', turn.get('role', 'Staff'))}</span>
        </div>
        <div class="dialogue-content">
            {content}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_turns_incrementally(
    turns: list[DialogueTurn],
    container,
    delay: float = 0.0,
    scrollable: bool = True,
    show_typing_indicator: bool = False,
    next_speaker: str = "",
    next_role: str = "",
) -> None:
    """
    Render a list of dialogue turns into a Streamlit container.

    This function is used to re-render all accumulated turns after
    each new turn is added (since st.empty() replaces its contents).
    Now supports showing a typing indicator for the next speaker.

    Args:
        turns: List of DialogueTurn objects to render
        container: Streamlit container (from st.empty() or st.container())
        delay: Optional delay between renders for visual effect
        scrollable: Whether to wrap in a scrollable container (default True)
        show_typing_indicator: Whether to show "X is typing..." after turns
        next_speaker: Name of the next speaker for typing indicator
        next_role: Role display for the next speaker
    """
    with container:
        if scrollable:
            st.markdown('<div class="dialogue-scroll-area">', unsafe_allow_html=True)

        st.markdown('<div class="dialogue-container">', unsafe_allow_html=True)
        for turn in turns:
            render_dialogue_bubble_from_turn(turn)

        # Show typing indicator if requested
        if show_typing_indicator and next_speaker:
            role_text = f" ({next_role})" if next_role else ""
            st.markdown(f"""
            <div class="typing-indicator">
                <div class="typing-indicator-content">
                    <span><strong>{next_speaker}</strong>{role_text} is typing</span>
                    <div class="typing-indicator-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        if scrollable:
            # Add scroll anchor for auto-scroll
            st.markdown('<div class="dialogue-scroll-anchor" id="scroll-anchor"></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    if delay > 0:
        time.sleep(delay)


def render_current_phase_banner(phase_num: int, phase_name: str, substep: str = "") -> None:
    """
    Render a banner showing the current phase being processed.

    Args:
        phase_num: The phase number (1-7)
        phase_name: The phase name (e.g., "Mission Analysis")
        substep: Optional substep indicator (a, b, c, d)
    """
    substep_names = {
        'a': 'Staff Meeting',
        'b': 'Slide Generation',
        'c': 'Commander Brief',
        'd': 'Commander Guidance',
    }
    substep_text = f" - {substep_names.get(substep, substep)}" if substep else ""

    html = f"""
    <div class="current-phase-banner">
        <span class="phase-indicator">STEP {phase_num}{substep.upper() if substep else ''}</span>
        <span class="phase-name">{phase_name}{substep_text}</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def format_transcript_text(turns: list[DialogueTurn]) -> str:
    """
    Format dialogue turns into a plain text transcript for download.

    Args:
        turns: List of DialogueTurn objects

    Returns:
        Formatted transcript as plain text string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("WARGATE STAFF MEETING TRANSCRIPT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")

    for turn in turns:
        speaker = turn.get('speaker', 'Unknown')
        role = turn.get('role_display', turn.get('role', 'Staff'))
        branch = turn.get('branch', 'Joint')
        text = turn.get('text', '')

        lines.append("-" * 50)
        lines.append(f"{speaker} ({branch}) - {role}")
        lines.append("-" * 50)
        lines.append(text)
        lines.append("")

    lines.append("=" * 70)
    lines.append("END OF TRANSCRIPT")
    lines.append("=" * 70)

    return "\n".join(lines)


def store_phase_transcript(phase_name: str, substep: str, turns: list[DialogueTurn]) -> None:
    """
    Store dialogue transcript in session state for later access.

    Args:
        phase_name: Name of the phase (e.g., "MISSION_ANALYSIS")
        substep: The substep ('a', 'b', 'c', 'd')
        turns: List of DialogueTurn objects
    """
    if "transcripts" not in st.session_state:
        st.session_state.transcripts = {}

    key = f"{phase_name}_{substep}"
    st.session_state.transcripts[key] = turns

    # Also store in phase_transcripts for the archive feature
    if "phase_transcripts" not in st.session_state:
        st.session_state.phase_transcripts = {}

    phase_key = f"{phase_name}_{substep}"
    st.session_state.phase_transcripts[phase_key] = turns


# =============================================================================
# MICRO-PROGRESS INDICATORS
# =============================================================================

# Phase-specific micro-progress messages that rotate during dialogue generation
MICRO_PROGRESS_MESSAGES: dict[str, list[str]] = {
    "PLANNING_INITIATION": [
        "J5 reviewing strategic guidance...",
        "J2 preparing initial threat assessment...",
        "J3 coordinating with component commands...",
        "Establishing planning timeline...",
        "Synchronizing staff inputs...",
        "Generating dialogue...",
    ],
    "MISSION_ANALYSIS": [
        "J2 preparing intelligence estimates...",
        "J3 reviewing operational factors...",
        "J5 drafting problem statement...",
        "Analyzing METT-TC factors...",
        "Developing CCIRs and PIRs...",
        "Synchronizing staff inputs...",
        "Generating dialogue...",
    ],
    "COA_DEVELOPMENT": [
        "J5 facilitating brainstorming session...",
        "J3 sketching initial COA concepts...",
        "Fires OIC integrating effects...",
        "J4 assessing sustainment requirements...",
        "Evaluating feasibility factors...",
        "Generating dialogue...",
    ],
    "COA_ANALYSIS": [
        "J2 preparing enemy COA templates...",
        "J3 setting up wargame framework...",
        "Running action-reaction cycles...",
        "Identifying decision points...",
        "Analyzing branches and sequels...",
        "Generating dialogue...",
    ],
    "COA_COMPARISON": [
        "J5 building comparison matrix...",
        "Evaluating COAs against criteria...",
        "Weighing advantages and risks...",
        "Formulating staff recommendation...",
        "Generating dialogue...",
    ],
    "COA_APPROVAL": [
        "Preparing decision brief...",
        "Final coordination across staff...",
        "Risk acceptance assessment...",
        "Generating dialogue...",
    ],
    "PLAN_DEVELOPMENT": [
        "J3 drafting OPORD framework...",
        "J5 integrating annexes...",
        "J4 finalizing sustainment annex...",
        "Building synchronization matrix...",
        "Final staff review...",
        "Generating dialogue...",
    ],
}


def run_micro_progress(
    container,
    phase_name: str,
    delay: float = 2.0,
    max_messages: int = 4,
) -> None:
    """
    Display a sequence of short status messages in a 'mission terminal'
    style area while background work is running.

    This creates the illusion of staged progress even if the actual
    work is a single blocking LLM call.

    Args:
        container: Streamlit container (from st.empty()) to display in
        phase_name: The phase name key for MICRO_PROGRESS_MESSAGES
        delay: Seconds between message updates (2-3 recommended)
        max_messages: Maximum number of messages to show before stopping
    """
    messages = MICRO_PROGRESS_MESSAGES.get(phase_name, [
        "Coordinating staff inputs...",
        "Synchronizing analysis...",
        "Generating dialogue...",
    ])

    for i, message in enumerate(messages[:max_messages]):
        with container:
            html = f"""
            <div class="micro-progress">
                <span class="micro-progress-icon">▶</span>
                <span class="micro-progress-text">{message}</span>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)
        time.sleep(delay if i < len(messages) - 1 else 1.0)


def render_micro_progress_message(container, message: str) -> None:
    """
    Render a single micro-progress status message.

    Args:
        container: Streamlit container to display in
        message: The status message to show
    """
    with container:
        html = f"""
        <div class="micro-progress">
            <span class="micro-progress-icon">▶</span>
            <span class="micro-progress-text">{message}</span>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# MISSION TERMINAL STYLE STATUS BAR
# =============================================================================

def render_mission_status(
    container,
    bar_fraction: float,
    message: str,
    phase_name: str = "",
    substep: str = "",
) -> None:
    """
    Render a terminal-like status bar using block characters to indicate progress.

    This replaces the generic Streamlit progress bar with a more thematic
    mission-terminal aesthetic.

    Args:
        container: Streamlit container to render in
        bar_fraction: Progress fraction from 0.0 to 1.0
        message: Current status message (e.g., "Generating Joint COA Matrix...")
        phase_name: Optional phase name for header
        substep: Optional substep indicator (a, b, c, d)

    Example output:
        [ ████████▒▒▒▒▒▒▒▒ ] 45% | Generating Joint COA Matrix...
    """
    # Calculate bar segments
    total_segments = 20
    completed = int(bar_fraction * total_segments)
    remaining = total_segments - completed

    # Build ASCII progress bar
    completed_chars = "█" * completed
    remaining_chars = "░" * remaining
    percentage = int(bar_fraction * 100)

    # Build header
    header_parts = ["WARGATE MISSION STATUS"]
    if phase_name:
        header_parts.append(f"// PHASE: {phase_name.replace('_', ' ')}")
    if substep:
        substep_names = {'a': 'STAFF MTG', 'b': 'SLIDES', 'c': 'CDR BRIEF', 'd': 'GUIDANCE'}
        header_parts.append(f"// {substep_names.get(substep, substep.upper())}")

    header = " ".join(header_parts)

    with container:
        html = f"""
        <div class="mission-terminal">
            <div class="mission-terminal-header">{header}</div>
            <div class="mission-terminal-progress">
                [ <span class="completed">{completed_chars}</span><span class="remaining">{remaining_chars}</span> ]
                <span class="cursor">_</span> {percentage}%
            </div>
            <div class="mission-terminal-status">&gt; {message}</div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# TYPING INDICATOR FOR ANIMATED DIALOGUE
# =============================================================================

def render_typing_indicator(
    container,
    speaker_name: str,
    role_display: str = "",
) -> None:
    """
    Render a typing indicator bubble showing who is about to speak.

    This creates the "X is typing..." effect before a new dialogue
    bubble appears, making the conversation feel more dynamic.

    Args:
        container: Streamlit container to render in
        speaker_name: Name of the speaker (e.g., "COL Young")
        role_display: Optional role display (e.g., "J2 - Intelligence")
    """
    with container:
        role_text = f" ({role_display})" if role_display else ""
        html = f"""
        <div class="typing-indicator">
            <div class="typing-indicator-content">
                <span><strong>{speaker_name}</strong>{role_text} is typing</span>
                <div class="typing-indicator-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)


def render_turns_with_typing_animation(
    turns: list[DialogueTurn],
    container,
    typing_delay: float = 0.4,
    turn_delay: float = 0.2,
    scrollable: bool = True,
) -> None:
    """
    Render dialogue turns with typing indicator animation before each new message.

    This function shows a brief "X is typing..." indicator before each
    dialogue bubble appears, creating a more dynamic conversation feel.

    Args:
        turns: List of DialogueTurn objects to render
        container: Streamlit container (from st.empty() or st.container())
        typing_delay: Seconds to show typing indicator (0.3-0.7 recommended)
        turn_delay: Delay between turns after bubble appears
        scrollable: Whether to wrap in a scrollable container
    """
    rendered_turns: list[DialogueTurn] = []

    for turn in turns:
        # First, show typing indicator with previously rendered turns
        with container:
            if scrollable:
                st.markdown('<div class="dialogue-scroll-area">', unsafe_allow_html=True)

            st.markdown('<div class="dialogue-container">', unsafe_allow_html=True)

            # Render all previously completed turns
            for prev_turn in rendered_turns:
                render_dialogue_bubble_from_turn(prev_turn)

            # Show typing indicator for current speaker
            render_typing_indicator(
                st.container(),
                turn.get('speaker', 'Staff'),
                turn.get('role_display', ''),
            )

            st.markdown('</div>', unsafe_allow_html=True)
            if scrollable:
                # Auto-scroll anchor
                st.markdown('<div class="dialogue-scroll-anchor" id="scroll-anchor"></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        time.sleep(typing_delay)

        # Now render with the actual turn
        rendered_turns.append(turn)

        with container:
            if scrollable:
                st.markdown('<div class="dialogue-scroll-area">', unsafe_allow_html=True)

            st.markdown('<div class="dialogue-container">', unsafe_allow_html=True)
            for prev_turn in rendered_turns:
                render_dialogue_bubble_from_turn(prev_turn)
            st.markdown('</div>', unsafe_allow_html=True)

            if scrollable:
                st.markdown('<div class="dialogue-scroll-anchor" id="scroll-anchor"></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        if turn_delay > 0:
            time.sleep(turn_delay)


# =============================================================================
# COLLAPSIBLE TRANSCRIPT ARCHIVE
# =============================================================================

def get_pretty_phase_name(phase_id: str) -> str:
    """
    Convert a phase_substep ID to a human-readable name.

    Args:
        phase_id: Phase ID like "MISSION_ANALYSIS_a" or "COA_DEVELOPMENT_c"

    Returns:
        Pretty name like "Step 2: Mission Analysis - Staff Meeting"
    """
    # Parse phase and substep
    parts = phase_id.rsplit('_', 1)
    if len(parts) == 2:
        phase_name, substep = parts
    else:
        phase_name = phase_id
        substep = ""

    # Phase number mapping
    phase_nums = {
        "PLANNING_INITIATION": 1,
        "MISSION_ANALYSIS": 2,
        "COA_DEVELOPMENT": 3,
        "COA_ANALYSIS": 4,
        "COA_COMPARISON": 5,
        "COA_APPROVAL": 6,
        "PLAN_DEVELOPMENT": 7,
    }

    phase_display = {
        "PLANNING_INITIATION": "Planning Initiation",
        "MISSION_ANALYSIS": "Mission Analysis",
        "COA_DEVELOPMENT": "COA Development",
        "COA_ANALYSIS": "COA Analysis & Wargaming",
        "COA_COMPARISON": "COA Comparison",
        "COA_APPROVAL": "COA Approval",
        "PLAN_DEVELOPMENT": "Plan/Order Development",
    }

    substep_names = {
        'a': 'Staff Meeting',
        'b': 'Slide Generation',
        'c': 'Commander Brief',
        'd': 'Commander Guidance',
    }

    num = phase_nums.get(phase_name, 0)
    display = phase_display.get(phase_name, phase_name.replace('_', ' ').title())
    substep_display = substep_names.get(substep, substep)

    if substep:
        return f"Step {num}: {display} - {substep_display}"
    return f"Step {num}: {display}"


def render_transcript_archive() -> None:
    """
    Render the collapsible transcript archive showing all completed phase transcripts.

    This creates an expander with all stored transcripts organized by phase,
    separate from the live streaming area to keep the main view focused.
    """
    phase_transcripts = st.session_state.get("phase_transcripts", {})

    if not phase_transcripts:
        return

    with st.expander("📜 Transcript Archive", expanded=False):
        st.markdown("""
        <div class="transcript-archive-header">
            Complete dialogue transcripts from all completed phases
        </div>
        """, unsafe_allow_html=True)

        # Sort transcripts by phase order
        phase_order = [
            "PLANNING_INITIATION", "MISSION_ANALYSIS", "COA_DEVELOPMENT",
            "COA_ANALYSIS", "COA_COMPARISON", "COA_APPROVAL", "PLAN_DEVELOPMENT"
        ]
        substep_order = ['a', 'b', 'c', 'd']

        # Group by phase
        for phase_name in phase_order:
            phase_has_content = False
            for substep in substep_order:
                key = f"{phase_name}_{substep}"
                if key in phase_transcripts and phase_transcripts[key]:
                    if not phase_has_content:
                        st.markdown(f"#### {get_pretty_phase_name(f'{phase_name}_a').rsplit(' - ', 1)[0]}")
                        phase_has_content = True

                    turns = phase_transcripts[key]
                    substep_names = {'a': 'Staff Meeting', 'c': 'Commander Brief'}

                    # Only show substantive dialogue (a and c)
                    if substep in ['a', 'c']:
                        with st.container():
                            st.markdown(f"**{substep_names.get(substep, substep)}** ({len(turns)} turns)")

                            for turn in turns:
                                speaker = turn.get('speaker', 'Unknown')
                                role = turn.get('role_display', '')
                                branch = turn.get('branch', '')
                                text = turn.get('text', '')

                                # Truncate long text for archive view
                                if len(text) > 300:
                                    text = text[:300] + "..."

                                st.markdown(f"""
                                <div class="transcript-turn">
                                    <span class="transcript-speaker">{speaker}</span>
                                    <span style="color: #666;">({branch}, {role})</span>:
                                    {text}
                                </div>
                                """, unsafe_allow_html=True)

                            st.markdown("---")


def render_transcript_access(phase_name: str, substep: str) -> None:
    """
    Render transcript access options (expander and download button).

    Args:
        phase_name: Name of the phase
        substep: The substep ('a', 'b', 'c', 'd')
    """
    key = f"{phase_name}_{substep}"
    transcripts = st.session_state.get("transcripts", {})
    turns = transcripts.get(key, [])

    if not turns:
        return

    col1, col2 = st.columns([3, 1])

    with col1:
        with st.expander("View Full Transcript", expanded=False):
            for turn in turns:
                speaker = turn.get('speaker', 'Unknown')
                role = turn.get('role_display', '')
                text = turn.get('text', '')
                st.markdown(f"**{speaker}** ({role}):")
                st.markdown(text)
                st.markdown("---")

    with col2:
        transcript_text = format_transcript_text(turns)
        st.download_button(
            "Download Transcript",
            data=transcript_text,
            file_name=f"wargate_{phase_name}_{substep}_transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )


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
# IMPROVED PDF GENERATION - REAL CONTENT, IMMEDIATE GENERATION
# =============================================================================

class SlideSection(TypedDict):
    """
    Represents a section in a slide/report PDF.

    Attributes:
        title: Section heading (e.g., "Mission Statement", "Key Assumptions")
        subtitle: Optional subtitle or descriptor (e.g., "From J2 Analysis")
        body: Main content - may contain bullets/newlines, will be preserved
    """
    title: str
    subtitle: str | None
    body: str


# Map JPPPhase enum names to human-readable phase IDs for PDF storage
PHASE_ID_MAP = {
    "PLANNING_INITIATION": "step1_planning_initiation",
    "MISSION_ANALYSIS": "step2_mission_analysis",
    "COA_DEVELOPMENT": "step3_coa_development",
    "COA_ANALYSIS": "step4_coa_analysis",
    "COA_COMPARISON": "step5_coa_comparison",
    "COA_APPROVAL": "step6_coa_approval",
    "PLAN_DEVELOPMENT": "step7_plan_development",
}


class WARGATEReportPDF(FPDF):
    """
    PDF class for WARGATE phase reports - text-focused, NotebookLM-friendly format.

    Unlike WARGATESlidePDF (landscape/slide format), this creates portrait
    documents optimized for reading and AI ingestion.
    """

    def __init__(
        self,
        op_name: str = "Operation WARGATE",
        phase_label: str = "Planning Phase",
        classification: str = "UNCLASSIFIED // FOR EXERCISE PURPOSES ONLY"
    ):
        super().__init__(orientation='P')  # Portrait for reports
        self.op_name = op_name
        self.phase_label = phase_label
        self.classification = classification
        self.set_auto_page_break(auto=True, margin=20)
        self.add_page()

    def header(self):
        """Report header with classification banner."""
        # Classification banner - purple
        self.set_fill_color(106, 13, 173)
        self.rect(0, 0, 210, 10, 'F')
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(255, 255, 255)
        self.set_xy(0, 2)
        self.cell(210, 6, self.classification, align="C")

        # Operation name and phase
        self.set_xy(10, 14)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(106, 13, 173)
        self.cell(0, 8, self.op_name, ln=True)

        self.set_font("Helvetica", "", 11)
        self.set_text_color(60, 60, 60)
        self.cell(0, 6, self.phase_label, ln=True)

        # Separator line
        self.set_draw_color(106, 13, 173)
        self.set_line_width(0.5)
        self.line(10, 30, 200, 30)
        self.ln(18)

    def footer(self):
        """Report footer with page number and classification."""
        self.set_y(-15)

        # Classification repeat
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 5, self.classification, align="C", ln=True)

        # Page number
        self.cell(0, 5, f"Page {self.page_no()}", align="C")

    def add_section(self, title: str, subtitle: str | None, body: str):
        """
        Add a section to the report.

        Args:
            title: Section heading (rendered large/bold)
            subtitle: Optional subtitle (rendered smaller/italic)
            body: Content text - newlines and bullets preserved
        """
        # Section title
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(26, 26, 46)
        self.cell(0, 8, self._clean_text(title), ln=True)

        # Subtitle if present
        if subtitle:
            self.set_font("Helvetica", "I", 10)
            self.set_text_color(100, 100, 100)
            self.cell(0, 5, self._clean_text(subtitle), ln=True)

        self.ln(2)

        # Body content
        self.set_font("Helvetica", "", 10)
        self.set_text_color(0, 0, 0)

        # Process body - preserve structure
        clean_body = self._clean_text(body)
        lines = clean_body.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                self.ln(3)
                continue

            # Detect bullets/numbered lists
            if line.startswith(('-', '*', '•')):
                # Bullet point
                self.set_x(15)
                bullet_text = line.lstrip('-*• ').strip()
                self.multi_cell(180, 5, f"• {bullet_text}")
            elif len(line) > 2 and line[0].isdigit() and line[1] in '.):':
                # Numbered item
                self.set_x(15)
                self.multi_cell(180, 5, line)
            else:
                # Regular paragraph
                self.multi_cell(0, 5, line)

        self.ln(5)

    def add_transcript_entry(
        self,
        speaker: str,
        rank: str,
        branch: str,
        role: str,
        text: str
    ):
        """
        Add a transcript entry (single dialogue turn).

        Args:
            speaker: Speaker name (e.g., "COL Smith")
            rank: Rank abbreviation (e.g., "COL")
            branch: Service branch (e.g., "US Army")
            role: Staff role (e.g., "J2 - Intelligence")
            text: The spoken content
        """
        # Speaker header
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(106, 13, 173)
        header = f"{speaker} ({branch}, {role}):"
        self.cell(0, 6, self._clean_text(header), ln=True)

        # Content
        self.set_font("Helvetica", "", 10)
        self.set_text_color(0, 0, 0)
        self.set_x(15)
        self.multi_cell(180, 5, self._clean_text(text))
        self.ln(4)

    def _clean_text(self, text: str) -> str:
        """Clean text for PDF output - handle special chars and markdown."""
        if not text:
            return ""

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


def build_phase_pdf(
    op_name: str,
    phase_label: str,
    classification: str,
    sections: list[SlideSection]
) -> BytesIO:
    """
    Build a text-focused PDF representing all slide/report content for a single phase.

    This creates a NotebookLM-friendly document with proper headings, preserved
    bullet structures, and clear section divisions.

    Args:
        op_name: Operation name (e.g., "Operation GenAI")
        phase_label: Phase descriptor (e.g., "Step 2: Mission Analysis")
        classification: Classification marking (e.g., "UNCLASSIFIED // FOUO")
        sections: List of SlideSection dicts with title, subtitle, and body

    Returns:
        BytesIO buffer containing the PDF

    Example:
        sections = [
            {
                "title": "Mission Statement",
                "subtitle": "Restated from Commander's Intent",
                "body": "Conduct operations to secure...\n- Objective 1\n- Objective 2"
            },
            {
                "title": "Key Assumptions",
                "subtitle": None,
                "body": "1. Enemy forces will...\n2. Allied support..."
            }
        ]
        pdf_buffer = build_phase_pdf("Op WARGATE", "Step 2: Mission Analysis",
                                      "UNCLASSIFIED", sections)
    """
    pdf = WARGATEReportPDF(
        op_name=op_name,
        phase_label=phase_label,
        classification=classification
    )

    for section in sections:
        pdf.add_section(
            title=section.get("title", "Section"),
            subtitle=section.get("subtitle"),
            body=section.get("body", "")
        )

    # Generate to BytesIO
    buffer = BytesIO()
    pdf_output = pdf.output()
    buffer.write(pdf_output)
    buffer.seek(0)

    return buffer


def build_transcript_pdf(
    op_name: str,
    phase_label: str,
    classification: str,
    turns: list[DialogueTurn]
) -> BytesIO:
    """
    Build a PDF containing the full dialogue transcript for a single phase.

    Format includes speaker name, rank, branch, and role for each turn,
    maintaining chronological order.

    Args:
        op_name: Operation name (e.g., "Operation GenAI")
        phase_label: Phase descriptor (e.g., "Step 2: Mission Analysis - Staff Meeting")
        classification: Classification marking
        turns: List of DialogueTurn dicts from the meeting

    Returns:
        BytesIO buffer containing the PDF

    Example:
        turns = [
            {
                "speaker": "COL Young",
                "role": "j2_intelligence",
                "role_display": "J2 - Intelligence",
                "branch": "US Air Force",
                "rank": "COL",
                "text": "Based on current intel...",
                "turn_number": 1,
                "is_commander": False
            },
            ...
        ]
        pdf_buffer = build_transcript_pdf("Op WARGATE", "Step 2: Mission Analysis",
                                          "UNCLASSIFIED", turns)
    """
    pdf = WARGATEReportPDF(
        op_name=op_name,
        phase_label=f"{phase_label} - Transcript",
        classification=classification
    )

    # Add metadata header
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, f"Dialogue Transcript - {len(turns)} turns", ln=True)
    pdf.cell(0, 5, f"Generated: {datetime.now().strftime('%d %B %Y %H:%M')}", ln=True)
    pdf.ln(8)

    # Add each turn
    for turn in turns:
        speaker = turn.get("speaker", "Unknown")
        rank = turn.get("rank", "")
        branch = turn.get("branch", "")
        role = turn.get("role_display", turn.get("role", "Staff"))
        text = turn.get("text", "")

        # Handle commander specially
        if turn.get("is_commander", False):
            role = "Commander"
            branch = "Joint"

        pdf.add_transcript_entry(
            speaker=speaker,
            rank=rank,
            branch=branch,
            role=role,
            text=text
        )

    # Generate to BytesIO
    buffer = BytesIO()
    pdf_output = pdf.output()
    buffer.write(pdf_output)
    buffer.seek(0)

    return buffer


def extract_sections_from_phase_result(phase_result: PhaseResult) -> list[SlideSection]:
    """
    Extract SlideSection list from a PhaseResult's slides and other content.

    This converts the orchestrator's output format into sections suitable
    for build_phase_pdf.

    Args:
        phase_result: The PhaseResult from the orchestrator

    Returns:
        List of SlideSection dicts
    """
    sections: list[SlideSection] = []

    # Convert slides to sections
    for slide in phase_result.get("slides", []):
        title = slide.get("title", "Section")
        bullets = slide.get("bullets", [])
        notes = slide.get("notes", "")

        # Convert bullets to body text
        body_lines = []
        for bullet in bullets:
            body_lines.append(f"• {bullet}")
        if notes:
            body_lines.append("")
            body_lines.append(f"Notes: {notes}")

        sections.append({
            "title": title,
            "subtitle": None,
            "body": "\n".join(body_lines)
        })

    # Add commander guidance as a section
    guidance = phase_result.get("guidance", {})
    if guidance:
        guidance_text = guidance.get("guidance_text", "")
        priority_tasks = guidance.get("priority_tasks", [])

        body_parts = []
        if guidance_text:
            body_parts.append(guidance_text)
        if priority_tasks:
            body_parts.append("\nPriority Tasks:")
            for task in priority_tasks:
                body_parts.append(f"• {task}")

        if body_parts:
            sections.append({
                "title": "Commander's Guidance",
                "subtitle": "Direction for Next Phase",
                "body": "\n".join(body_parts)
            })

    return sections


def generate_and_store_phase_pdfs(
    phase_name: str,
    phase_result: PhaseResult,
    op_name: str = "Operation WARGATE",
    classification: str = "UNCLASSIFIED // FOR EXERCISE PURPOSES ONLY"
) -> dict[str, BytesIO]:
    """
    Generate both PDFs for a phase and store them in session state.

    Called immediately after a phase completes, BEFORE moving to the next phase.
    This ensures PDFs are available even if later phases fail.

    Args:
        phase_name: The JPPPhase.name (e.g., "MISSION_ANALYSIS")
        phase_result: The PhaseResult from the orchestrator
        op_name: Operation name for PDF headers
        classification: Classification marking

    Returns:
        Dict with "slides" and "transcript" BytesIO buffers

    Session State:
        Stores in st.session_state["phase_pdfs"][phase_id] = {
            "slides": BytesIO,
            "transcript": BytesIO,
            "generated_at": datetime
        }
    """
    # Initialize phase_pdfs storage if needed
    if "phase_pdfs" not in st.session_state:
        st.session_state["phase_pdfs"] = {}

    # Get phase ID
    phase_id = PHASE_ID_MAP.get(phase_name, phase_name.lower())
    phase_display = phase_result.get("phase_name", phase_name.replace("_", " ").title())
    phase_num = {
        "PLANNING_INITIATION": 1,
        "MISSION_ANALYSIS": 2,
        "COA_DEVELOPMENT": 3,
        "COA_ANALYSIS": 4,
        "COA_COMPARISON": 5,
        "COA_APPROVAL": 6,
        "PLAN_DEVELOPMENT": 7,
    }.get(phase_name, 1)
    phase_label = f"Step {phase_num}: {phase_display}"

    # Extract sections from phase result
    sections = extract_sections_from_phase_result(phase_result)

    # Build slide/report PDF
    slides_pdf = build_phase_pdf(
        op_name=op_name,
        phase_label=phase_label,
        classification=classification,
        sections=sections
    )

    # Combine all dialogue turns for transcript
    all_turns: list[DialogueTurn] = []

    # Staff meeting turns
    meeting = phase_result.get("meeting", {})
    meeting_turns = meeting.get("turns", [])
    all_turns.extend(meeting_turns)

    # Brief turns
    brief = phase_result.get("brief", {})
    brief_turns = brief.get("turns", [])
    all_turns.extend(brief_turns)

    # Build transcript PDF
    transcript_pdf = build_transcript_pdf(
        op_name=op_name,
        phase_label=phase_label,
        classification=classification,
        turns=all_turns
    )

    # Store in session state - WILL NOT BE WIPED by later errors
    st.session_state["phase_pdfs"][phase_id] = {
        "slides": slides_pdf,
        "transcript": transcript_pdf,
        "phase_name": phase_display,
        "phase_num": phase_num,
        "generated_at": datetime.now(),
        "turn_count": len(all_turns),
        "section_count": len(sections),
    }

    return {
        "slides": slides_pdf,
        "transcript": transcript_pdf,
    }


def render_phase_pdf_downloads(phase_name: str) -> None:
    """
    Render download buttons for a phase's PDFs if available.

    This should be called in render_phase_section for each completed phase.
    Shows two buttons: Slide Report and Transcript.

    Args:
        phase_name: The JPPPhase.name (e.g., "MISSION_ANALYSIS")
    """
    phase_id = PHASE_ID_MAP.get(phase_name, phase_name.lower())
    phase_store = st.session_state.get("phase_pdfs", {})
    phase_data = phase_store.get(phase_id)

    if not phase_data:
        return

    phase_display = phase_data.get("phase_name", phase_name.replace("_", " ").title())
    phase_num = phase_data.get("phase_num", 1)
    turn_count = phase_data.get("turn_count", 0)
    section_count = phase_data.get("section_count", 0)

    st.markdown("---")
    st.markdown("#### 📄 Phase Documents")
    st.caption(f"Generated with {section_count} sections and {turn_count} dialogue turns")

    col1, col2 = st.columns(2)

    with col1:
        # Need to reset buffer position before download
        slides_buffer = phase_data["slides"]
        slides_buffer.seek(0)

        st.download_button(
            label=f"📊 Download {phase_display} Report (PDF)",
            data=slides_buffer,
            file_name=f"wargate_step{phase_num}_{phase_name.lower()}_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with col2:
        # Need to reset buffer position before download
        transcript_buffer = phase_data["transcript"]
        transcript_buffer.seek(0)

        st.download_button(
            label=f"📝 Download {phase_display} Transcript (PDF)",
            data=transcript_buffer,
            file_name=f"wargate_step{phase_num}_{phase_name.lower()}_transcript.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


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
        # Legacy planning result
        "planning_result": None,
        "scenario_text": "",
        "is_running": False,
        "current_phase": None,
        "phase_outputs": {},
        "dialogue_history": [],
        "pdf_slides": {},
        "model_name": "gpt-4o",
        "temperature": 0.8,  # Higher default for natural dialogue
        "persona_seed": 0,
        # New orchestration state
        "orchestrator": None,
        "live_turns": [],           # Current live dialogue turns
        "current_substep": None,    # 'a', 'b', 'c', or 'd'
        "substep_status": "",       # Status message for current substep
        "phase_results": {},        # PhaseResult objects by phase name
        "prior_context": "",        # Accumulated context from prior phases
        # Enhanced UX state
        "phase_transcripts": {},    # Full transcripts per phase for archive
        "micro_progress_index": 0,  # Current micro-progress message index
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Safety reset: If is_running is True but we're starting fresh,
    # reset it to prevent the RUN button from being stuck disabled
    # (This can happen if a previous run crashed/was interrupted)
    if st.session_state.is_running and not st.session_state.phase_outputs:
        st.session_state.is_running = False


def get_or_create_orchestrator() -> MeetingOrchestrator:
    """Get the cached orchestrator or create a new one."""
    if st.session_state.orchestrator is None:
        # Handle persona_seed being None or 0
        persona_seed = st.session_state.persona_seed
        if persona_seed is None or persona_seed == 0:
            persona_seed = None

        st.session_state.orchestrator = create_orchestrator(
            model_name=st.session_state.model_name,
            temperature=st.session_state.temperature,
            persona_seed=persona_seed,
        )
    return st.session_state.orchestrator


def reset_orchestrator():
    """Reset the orchestrator (e.g., when settings change)."""
    st.session_state.orchestrator = None
    st.session_state.live_turns = []
    st.session_state.phase_results = {}
    st.session_state.prior_context = ""


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
                # Reset all planning state
                reset_keys = {
                    "planning_result": None,
                    "scenario_text": "",
                    "phase_outputs": {},
                    "dialogue_history": [],
                    "pdf_slides": {},
                    "current_phase": None,
                    "orchestrator": None,
                    "live_turns": [],
                    "phase_results": {},
                    "prior_context": "",
                    "is_running": False,
                }
                for key, default_value in reset_keys.items():
                    st.session_state[key] = default_value
                st.rerun()

        # Status indicators
        st.markdown("---")
        st.subheader("Status")

        # API Key status
        api_key_set = bool(os.environ.get("OPENAI_API_KEY"))
        if api_key_set:
            st.success("✅ API Key: Configured")
        else:
            st.error("❌ API Key: Not Set")

        # Scenario status
        if scenario and scenario.strip():
            st.success(f"✅ Scenario: {len(scenario)} chars")
        else:
            st.warning("⚠️ Scenario: Not entered")

        # Ready status
        if api_key_set and scenario and scenario.strip():
            st.info("🚀 Ready to run!")
        elif st.session_state.is_running:
            st.info("🔄 Planning in progress...")

        return {
            "model_name": model_name,
            "temperature": temperature,
            "persona_seed": persona_seed if persona_seed is not None and persona_seed > 0 else None,
            "scenario": scenario,
            "run_clicked": run_clicked,
        }


def render_phase_section(
    phase: JPPPhase,
    phase_info: dict,
    is_current: bool = False,
    is_complete: bool = False
):
    """
    Render a JPP phase section with dialogue and PDF outputs.

    Features:
    - Scrollable dialogue containers (max 500px height)
    - Bold first-sentence summaries for skimming
    - Transcript download buttons
    - Collapsible expanders for completed phases
    """
    phase_num = phase.value
    phase_name = phase_info["name"]

    # Phase header
    status_icon = "🔄" if is_current else ("✅" if is_complete else "⏳")

    # Completed phases are collapsed by default, current phase is expanded
    with st.expander(f"STEP {phase_num}: {phase_name.upper()} {status_icon}", expanded=is_current):
        st.markdown(f"**Description:** {phase_info['description']}")

        if is_complete and phase.name in st.session_state.phase_outputs:
            output = st.session_state.phase_outputs[phase.name]

            # Check if we have new PhaseResult format
            phase_result = output.get("phase_result") if isinstance(output, dict) else None

            # Sub-tabs for each sub-step
            tabs = st.tabs(["Staff Meeting", "Slides", "Brief Commander", "Commander Guidance"])

            with tabs[0]:
                st.subheader("Staff Meeting Dialogue")

                # Try to use PhaseResult format first (new format)
                if phase_result and "meeting" in phase_result:
                    meeting_turns = phase_result["meeting"]["turns"]
                    turn_count = len(meeting_turns)
                    unique_agents = len(set(t['role'] for t in meeting_turns))
                    st.caption(f"{turn_count} dialogue turns from {unique_agents} staff agents")

                    # Render in scrollable container
                    st.markdown('<div class="dialogue-scroll-area">', unsafe_allow_html=True)
                    st.markdown('<div class="dialogue-container">', unsafe_allow_html=True)
                    for turn in meeting_turns:
                        render_dialogue_bubble_from_turn(turn)
                    st.markdown('</div></div>', unsafe_allow_html=True)

                    # Transcript access
                    render_transcript_access(phase.name, 'a')

                # Fallback to legacy format
                elif "dialogues" in output:
                    st.markdown('<div class="dialogue-scroll-area">', unsafe_allow_html=True)
                    for role, content in output["dialogues"]:
                        persona = DEFAULT_PERSONAS.get(role, DEFAULT_PERSONAS["commander"])
                        render_dialogue_bubble(persona, content, role == "commander")
                    st.markdown('</div>', unsafe_allow_html=True)

            with tabs[1]:
                st.subheader("Planning Slides")

                # Show slide preview if available
                if phase_result and "slides" in phase_result:
                    st.markdown('<div class="dialogue-scroll-area">', unsafe_allow_html=True)
                    for slide in phase_result["slides"]:
                        with st.container():
                            st.markdown(f"**{slide['title']}**")
                            for bullet in slide["bullets"]:
                                st.markdown(f"- {bullet}")
                            st.markdown("---")
                    st.markdown('</div>', unsafe_allow_html=True)

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

                # Try PhaseResult format first
                if phase_result and "brief" in phase_result:
                    brief_turns = phase_result["brief"]["turns"]

                    # Render in scrollable container
                    st.markdown('<div class="dialogue-scroll-area">', unsafe_allow_html=True)
                    st.markdown('<div class="dialogue-container">', unsafe_allow_html=True)
                    for turn in brief_turns:
                        render_dialogue_bubble_from_turn(turn)
                    st.markdown('</div></div>', unsafe_allow_html=True)

                    # Show Q&A summary
                    if phase_result["brief"]["questions_asked"]:
                        st.markdown("#### Commander's Questions:")
                        for q in phase_result["brief"]["questions_asked"]:
                            st.markdown(f"- {q}")

                    # Transcript access
                    render_transcript_access(phase.name, 'c')

                # Fallback to legacy format
                elif "brief" in output:
                    st.markdown('<div class="dialogue-scroll-area">', unsafe_allow_html=True)
                    for role, content in output["brief"]:
                        persona = DEFAULT_PERSONAS.get(role, DEFAULT_PERSONAS["commander"])
                        render_dialogue_bubble(persona, content, role == "commander")
                    st.markdown('</div>', unsafe_allow_html=True)

            with tabs[3]:
                st.subheader("Commander's Guidance")

                # Try PhaseResult format first
                if phase_result and "guidance" in phase_result:
                    guidance = phase_result["guidance"]

                    # Render as commander dialogue bubble
                    guidance_turn = {
                        "speaker": "CMDR",
                        "role": "commander",
                        "role_display": "Commander",
                        "branch": "Joint",
                        "rank": "LTG",
                        "text": guidance["guidance_text"],
                        "turn_number": 1,
                        "is_commander": True,
                    }
                    render_dialogue_bubble_from_turn(guidance_turn)

                    # Show priority tasks
                    if guidance.get("priority_tasks"):
                        st.markdown("#### Priority Tasks for Next Phase:")
                        for task in guidance["priority_tasks"]:
                            st.markdown(f"- {task}")

                    # Show section-specific guidance
                    if guidance.get("guidance_by_section"):
                        st.markdown("#### Guidance by Section:")
                        for section, text in guidance["guidance_by_section"].items():
                            st.markdown(f"**{section.upper()}:** {text}")

                # Fallback to legacy format
                elif "guidance" in output:
                    render_dialogue_bubble(
                        DEFAULT_PERSONAS["commander"],
                        output["guidance"],
                        is_commander=True
                    )

            # PDF Download Section - always at the end of tabs
            # Shows two buttons: Report PDF and Transcript PDF
            render_phase_pdf_downloads(phase.name)

        elif is_current:
            st.info("🔄 This phase is currently being processed...")

            # Show live dialogue if available
            if st.session_state.live_turns:
                st.markdown("### Live Dialogue:")
                for turn in st.session_state.live_turns:
                    render_dialogue_bubble_from_turn(turn)
            else:
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
    """Render the main planning dashboard with all phases and transcript archive."""
    st.markdown("## Joint Planning Process Dashboard")

    # Progress overview using mission terminal style
    completed_phases = len([p for p in JPPPhase if p.name in st.session_state.phase_outputs])
    progress_container = st.empty()
    render_mission_status(
        progress_container,
        completed_phases / 7,
        f"{completed_phases}/7 phases complete",
        phase_name="DASHBOARD",
    )

    # Render each phase
    for phase in JPPPhase:
        phase_info = JPP_PHASE_INFO[phase]
        is_current = st.session_state.current_phase == phase
        is_complete = phase.name in st.session_state.phase_outputs

        render_phase_section(phase, phase_info, is_current, is_complete)

    # Add the collapsible transcript archive
    render_transcript_archive()

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
# PLANNING EXECUTION (NEW ORCHESTRATOR-BASED)
# =============================================================================

def map_jpp_phase_to_orchestrator(phase: JPPPhase) -> OrchestratorJPPPhase:
    """Map the app's JPPPhase enum to the orchestrator's version."""
    mapping = {
        JPPPhase.PLANNING_INITIATION: OrchestratorJPPPhase.PLANNING_INITIATION,
        JPPPhase.MISSION_ANALYSIS: OrchestratorJPPPhase.MISSION_ANALYSIS,
        JPPPhase.COA_DEVELOPMENT: OrchestratorJPPPhase.COA_DEVELOPMENT,
        JPPPhase.COA_ANALYSIS: OrchestratorJPPPhase.COA_ANALYSIS,
        JPPPhase.COA_COMPARISON: OrchestratorJPPPhase.COA_COMPARISON,
        JPPPhase.COA_APPROVAL: OrchestratorJPPPhase.COA_APPROVAL,
        JPPPhase.PLAN_DEVELOPMENT: OrchestratorJPPPhase.PLAN_DEVELOPMENT,
    }
    return mapping[phase]


def run_phase_with_live_dialogue(
    phase: JPPPhase,
    scenario: str,
    dialogue_container,
    status_container,
    banner_container=None,
    micro_progress_container=None,
) -> PhaseResult | None:
    """
    Execute a single JPP phase with live incremental dialogue rendering.

    This function runs the full 4-step meeting flow (a-d) for a phase,
    rendering each dialogue turn as it happens in a scrollable container.
    Now includes:
    - Micro-progress indicators while generating
    - Typing indicators before each new speaker
    - Automatic transcript archiving

    Args:
        phase: The JPP phase to execute
        scenario: The operational scenario text
        dialogue_container: Streamlit container for live dialogue
        status_container: Streamlit container for status messages
        banner_container: Optional container for phase banner
        micro_progress_container: Optional container for micro-progress messages

    Returns:
        PhaseResult with all phase outputs, or None on error
    """
    orchestrator = get_or_create_orchestrator()
    orchestrator_phase = map_jpp_phase_to_orchestrator(phase)
    phase_info = JPP_PHASE_INFO[phase]

    # Get prior context from previous phases
    prior_context = st.session_state.prior_context

    # Track live turns for incremental rendering
    live_turns: list[DialogueTurn] = []
    current_substep = 'a'
    is_first_turn_in_substep = True

    def on_turn_callback(turn: DialogueTurn):
        """Called for each dialogue turn - renders with typing indicator animation."""
        nonlocal is_first_turn_in_substep

        # Clear micro-progress when first turn arrives
        if is_first_turn_in_substep and micro_progress_container:
            micro_progress_container.empty()
            is_first_turn_in_substep = False

        # Show typing indicator briefly before the actual turn
        render_turns_incrementally(
            live_turns,
            dialogue_container,
            delay=0,
            scrollable=True,
            show_typing_indicator=True,
            next_speaker=turn.get('speaker', 'Staff'),
            next_role=turn.get('role_display', ''),
        )
        time.sleep(0.35)  # Brief pause to see typing indicator

        # Now add the turn and render without indicator
        live_turns.append(turn)
        st.session_state.live_turns = live_turns

        # Re-render all turns in the scrollable dialogue container
        render_turns_incrementally(live_turns, dialogue_container, delay=0, scrollable=True)

    def on_substep_callback(substep: str, description: str):
        """Called when starting a new substep - shows micro-progress."""
        nonlocal current_substep, live_turns, is_first_turn_in_substep
        current_substep = substep
        is_first_turn_in_substep = True
        st.session_state.current_substep = substep
        st.session_state.substep_status = description

        # Store transcript from previous substep before clearing
        if live_turns and substep != 'a':
            prev_substep = chr(ord(substep) - 1)  # Get previous substep letter
            store_phase_transcript(phase.name, prev_substep, live_turns.copy())

        # Clear turns for new substep (except for 'a' which starts fresh)
        if substep in ['c', 'd']:  # Brief and Guidance are new conversations
            live_turns = []
            st.session_state.live_turns = []

        # Update status with micro-progress style
        if micro_progress_container and substep in ['a', 'c']:
            substep_descriptions = {
                'a': f"Staff assembling for {phase_info['name']}...",
                'c': "Preparing commander briefing...",
            }
            render_micro_progress_message(
                micro_progress_container,
                substep_descriptions.get(substep, description)
            )

        # Update main status
        with status_container:
            substep_names = {'a': 'Staff Meeting', 'b': 'Slide Generation', 'c': 'Commander Brief', 'd': 'Commander Guidance'}
            st.info(f"**Step {phase.value}{substep.upper()}**: {substep_names.get(substep, substep)}")

        # Update banner if provided
        if banner_container:
            with banner_container:
                render_current_phase_banner(phase.value, phase_info['name'], substep)

    try:
        # Clear live turns for this phase
        live_turns = []
        st.session_state.live_turns = []

        # Show initial micro-progress messages while waiting for dialogue to start
        if micro_progress_container:
            messages = MICRO_PROGRESS_MESSAGES.get(phase.name, [
                "Coordinating staff inputs...",
                "Synchronizing analysis...",
            ])
            # Show first 2 micro-progress messages quickly
            for msg in messages[:2]:
                render_micro_progress_message(micro_progress_container, msg)
                time.sleep(1.5)

        # Show initial banner
        if banner_container:
            with banner_container:
                render_current_phase_banner(phase.value, phase_info['name'], 'a')

        # Run the full phase with live callbacks
        phase_result = orchestrator.run_full_phase(
            phase=orchestrator_phase,
            scenario=scenario,
            prior_context=prior_context,
            on_turn_callback=on_turn_callback,
            on_substep_callback=on_substep_callback,
            turn_delay=0.1,  # Faster since we have typing indicator delay
        )

        # Store final transcripts
        if phase_result:
            # Store meeting transcript
            store_phase_transcript(phase.name, 'a', phase_result['meeting']['turns'])
            # Store brief transcript
            store_phase_transcript(phase.name, 'c', phase_result['brief']['turns'])

            # Update prior context for next phase
            meeting_transcript = phase_result['meeting']['transcript']
            guidance_text = phase_result['guidance']['guidance_text']
            st.session_state.prior_context += f"\n\n=== {phase_result['phase_name']} ===\n{meeting_transcript[:2000]}\n\nCommander Guidance: {guidance_text[:500]}"

        return phase_result

    except Exception as e:
        with status_container:
            st.error(f"Error during {phase.name}: {str(e)}")
        st.exception(e)
        return None


def convert_phase_result_to_legacy_format(phase_result: PhaseResult) -> dict:
    """Convert PhaseResult to the legacy format used by render_phase_section."""
    if not phase_result:
        return {}

    # Convert meeting turns to legacy dialogue format
    dialogues = []
    for turn in phase_result['meeting']['turns']:
        role_key = turn.get('role', 'commander')
        content = f"<p>{turn.get('text', '')}</p>"
        dialogues.append((role_key, content))

    # Convert brief turns to legacy format
    brief = []
    for turn in phase_result['brief']['turns']:
        role_key = turn.get('role', 'commander')
        content = f"<p>{turn.get('text', '')}</p>"
        brief.append((role_key, content))

    # Generate PDF from slides
    slide_sections = {}
    for slide in phase_result['slides']:
        slide_sections[slide['title']] = slide['bullets']

    phase_name = phase_result['phase_name']
    phase_num = {
        "Planning Initiation": 1,
        "Mission Analysis": 2,
        "COA Development": 3,
        "COA Analysis & Wargaming": 4,
        "COA Comparison": 5,
        "COA Approval": 6,
        "Plan/Order Development": 7,
    }.get(phase_name, 1)

    pdf_content = generate_phase_slides(phase_name, phase_num, slide_sections)

    return {
        "dialogues": dialogues,
        "pdf": pdf_content,
        "brief": brief,
        "guidance": f"<p>{phase_result['guidance']['guidance_text']}</p>",
        "phase_result": phase_result,  # Keep original for advanced use
    }


def run_single_phase_interactive(phase: JPPPhase, scenario: str) -> bool:
    """
    Run a single phase interactively with live dialogue rendering.

    This is called when the user clicks "Run Phase X" button.
    """
    st.session_state.is_running = True
    st.session_state.current_phase = phase

    # Create containers for live updates
    phase_info = JPP_PHASE_INFO[phase]

    st.markdown(f"## Running: Step {phase.value} - {phase_info['name']}")

    status_container = st.empty()
    dialogue_container = st.empty()

    with status_container:
        st.info(f"Starting {phase_info['name']}... Staff agents are assembling.")

    try:
        # Run the phase with live dialogue
        phase_result = run_phase_with_live_dialogue(
            phase=phase,
            scenario=scenario,
            dialogue_container=dialogue_container,
            status_container=status_container,
        )

        if phase_result:
            # Convert to legacy format and store
            legacy_output = convert_phase_result_to_legacy_format(phase_result)
            st.session_state.phase_outputs[phase.name] = legacy_output
            st.session_state.phase_results[phase.name] = phase_result

            # IMMEDIATELY generate and store PDFs - error resilient
            # These will persist even if later phases fail
            try:
                pdf_result = generate_and_store_phase_pdfs(
                    phase_name=phase.name,
                    phase_result=phase_result,
                    op_name="Operation WARGATE",
                    classification="UNCLASSIFIED // FOR EXERCISE PURPOSES ONLY"
                )
                with status_container:
                    st.success(f"{phase_info['name']} complete! PDFs generated.")
            except Exception as pdf_error:
                # PDF generation failure should NOT block phase completion
                with status_container:
                    st.warning(f"{phase_info['name']} complete. PDF generation failed: {pdf_error}")

            st.session_state.is_running = False
            return True
        else:
            st.session_state.is_running = False
            return False

    except Exception as e:
        with status_container:
            st.error(f"Error: {str(e)}")
        st.session_state.is_running = False
        return False


def run_full_planning_orchestrated(scenario: str) -> bool:
    """
    Run all 7 JPP phases sequentially with the new orchestrator.

    Enhanced Features:
    - Mission terminal style status bar (replaces generic progress bar)
    - Micro-progress indicators during generation
    - Typing indicators for animated dialogue
    - Live scrollable dialogue containers per phase
    - Automatic transcript archiving
    - Current phase banner

    This replaces the legacy run_full_planning function.
    """
    st.session_state.is_running = True

    # Create containers for live updates with enhanced UX
    terminal_container = st.empty()  # Mission terminal status bar
    micro_progress_container = st.empty()  # Micro-progress messages
    banner_container = st.empty()  # Current phase banner
    status_container = st.empty()
    dialogue_container = st.empty()

    phases = list(JPPPhase)
    total_phases = len(phases)

    try:
        for idx, phase in enumerate(phases):
            phase_info = JPP_PHASE_INFO[phase]

            # Update mission terminal status bar
            progress = idx / total_phases
            render_mission_status(
                terminal_container,
                progress,
                f"Executing {phase_info['name']}...",
                phase_name=phase.name,
                substep='a',
            )

            with status_container:
                st.info(f"Starting Phase {idx + 1}: {phase_info['name']}")

            # Clear dialogue for new phase
            st.session_state.live_turns = []

            # Run the phase with all enhanced features
            phase_result = run_phase_with_live_dialogue(
                phase=phase,
                scenario=scenario,
                dialogue_container=dialogue_container,
                status_container=status_container,
                banner_container=banner_container,
                micro_progress_container=micro_progress_container,
            )

            if phase_result:
                # Store results
                legacy_output = convert_phase_result_to_legacy_format(phase_result)
                st.session_state.phase_outputs[phase.name] = legacy_output
                st.session_state.phase_results[phase.name] = phase_result

                # Update terminal to show phase completion
                render_mission_status(
                    terminal_container,
                    (idx + 1) / total_phases,
                    f"Phase {idx + 1} complete: {phase_info['name']}",
                    phase_name=phase.name,
                    substep='d',
                )

                # IMMEDIATELY generate and store PDFs - error resilient
                # These will persist in session state even if later phases fail
                try:
                    generate_and_store_phase_pdfs(
                        phase_name=phase.name,
                        phase_result=phase_result,
                        op_name="Operation WARGATE",
                        classification="UNCLASSIFIED // FOR EXERCISE PURPOSES ONLY"
                    )
                except Exception as pdf_error:
                    # Log but don't stop - PDFs are secondary to phase completion
                    st.warning(f"PDF generation for {phase_info['name']} failed: {pdf_error}")
            else:
                st.error(f"Phase {phase_info['name']} failed. Stopping.")
                # Note: Previously completed phases' PDFs are still in session state
                st.session_state.is_running = False
                return False

        # Final completion status
        render_mission_status(
            terminal_container,
            1.0,
            "MISSION COMPLETE - All phases executed successfully",
            phase_name="COMPLETE",
        )
        banner_container.empty()  # Clear the banner
        micro_progress_container.empty()  # Clear micro-progress
        with status_container:
            st.success("Joint Planning Process complete! All 7 phases executed successfully.")

        st.session_state.is_running = False
        return True

    except Exception as e:
        progress_bar.empty()
        with status_container:
            st.error(f"Error: {str(e)}")
        st.exception(e)
        st.session_state.is_running = False
        return False


# Legacy function - kept for backward compatibility
def run_full_planning(scenario: str, model_name: str, temperature: float, persona_seed: int | None):
    """
    Legacy planning function - now redirects to orchestrated version.

    For the legacy backend, use run_joint_staff_planning_structured directly.
    """
    # Update session state with settings
    st.session_state.model_name = model_name
    st.session_state.temperature = temperature
    st.session_state.persona_seed = persona_seed

    # Reset orchestrator to use new settings
    reset_orchestrator()

    # Run the new orchestrated version
    return run_full_planning_orchestrated(scenario)


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
            st.warning("⚠️ Please enter a scenario before running the planning process.")
        elif not os.environ.get("OPENAI_API_KEY"):
            st.error("🔑 OPENAI_API_KEY environment variable is not set. Please set it and restart.")
        else:
            st.session_state.scenario_text = inputs["scenario"]
            st.session_state.is_running = True
            st.session_state.current_phase = JPPPhase.PLANNING_INITIATION

            try:
                success = run_full_planning(
                    scenario=inputs["scenario"],
                    model_name=inputs["model_name"],
                    temperature=inputs["temperature"],
                    persona_seed=inputs["persona_seed"]
                )
            except Exception as e:
                st.error(f"❌ Planning failed with error: {str(e)}")
                st.exception(e)
                success = False
            finally:
                # Always reset is_running to prevent stuck button
                st.session_state.is_running = False

            if success:
                st.rerun()
            return  # Don't render tabs after run - will rerun

    # Main content - show different views based on state
    if st.session_state.is_running:
        # Show running indicator (shouldn't normally reach here, but safety fallback)
        st.info("🔄 Planning in progress... Please wait.")
        if st.session_state.live_turns:
            st.markdown("### Live Dialogue")
            st.markdown('<div class="dialogue-container">', unsafe_allow_html=True)
            for turn in st.session_state.live_turns:
                render_dialogue_bubble_from_turn(turn)
            st.markdown('</div>', unsafe_allow_html=True)

    elif st.session_state.phase_outputs:
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


# NOTE: The pipeline diagram is rendered ONLY in render_pipeline_instructions()
# and is called ONLY inside the "Pipeline Instructions" tab context above.
# No other location should render the diagram.


if __name__ == "__main__":
    main()

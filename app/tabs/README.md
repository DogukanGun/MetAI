# Streamlit Tabs Module

This directory contains the modular tab components for the Multimodal Emotion Recognition System.

## Structure

```
tabs/
├── __init__.py          # Package exports
├── upload_tab.py        # Upload & Analyze functionality
├── results_tab.py       # Results display
└── help_tab.py          # Help & information
```

## Tab Files

### upload_tab.py
**Purpose**: Handle video upload and analysis

**Key Functions:**
- `render_upload_tab(config, extractors)`: Main render function
- `process_video_pipeline(video_path, config, extractors)`: Processing pipeline

**Features:**
- File upload interface
- Video preview
- 4-stage processing (Input → Features → Fusion → Results)
- Progress tracking
- Error handling

### results_tab.py
**Purpose**: Display analysis results

**Key Functions:**
- `render_results_tab()`: Main render function

**Features:**
- Video metadata display
- Emotion prediction visualization
- Confidence distribution charts
- Hybrid model analysis
  - Modality importance weights
  - Model agreement/disagreement
- Feature extraction summary
- Transcription display
- Export options (JSON, Report)

### help_tab.py
**Purpose**: Provide system information and guidance

**Key Functions:**
- `render_help_tab(config)`: Main render function

**Features:**
- System architecture explanation
- Fusion strategies comparison
- Supported emotions display
- Usage requirements
- Best practices
- Training guidance
- Technical specifications

## Usage

### In app.py
```python
from tabs import render_upload_tab, render_results_tab, render_help_tab

# In main()
tab1, tab2, tab3 = st.tabs(["Upload & Analyze", "Results", "Help"])

with tab1:
    render_upload_tab(config, extractors)

with tab2:
    render_results_tab()

with tab3:
    render_help_tab(config)
```

### Adding a New Tab

1. Create new file: `tabs/your_tab.py`

```python
"""Your Tab Description"""

import streamlit as st

def render_your_tab(config):
    """Render your custom tab."""
    st.header("Your Tab Title")
    
    # Your tab content here
    st.write("Hello from your tab!")
```

2. Update `tabs/__init__.py`:

```python
from .your_tab import render_your_tab

__all__ = [..., 'render_your_tab']
```

3. Update `app.py`:

```python
from tabs import ..., render_your_tab

# In main()
tab1, tab2, tab3, tab4 = st.tabs([..., "Your Tab"])

with tab4:
    render_your_tab(config)
```

## Design Principles

1. **Modularity**: Each tab is self-contained
2. **Reusability**: Functions can be called independently
3. **Consistency**: All tabs follow same structure
4. **Simplicity**: Clean, readable code
5. **Documentation**: Clear docstrings and comments

## Dependencies

Each tab file may use:
- `streamlit`: UI components
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `json`: JSON handling
- Application modules from `modules/`

## State Management

Tabs use Streamlit's session state to share data:
- `st.session_state['results']`: Analysis results
- `st.session_state['config']`: Configuration
- `st.session_state['audio_features']`: Audio features
- `st.session_state['visual_features']`: Visual features
- `st.session_state['text_features']`: Text features

## Best Practices

1. **Keep tabs focused**: Each tab handles one major function
2. **Use session state**: Share data between tabs properly
3. **Error handling**: Use try-except and user-friendly messages
4. **Progress feedback**: Show progress bars and status updates
5. **Clean imports**: Only import what's needed
6. **Documentation**: Document all public functions

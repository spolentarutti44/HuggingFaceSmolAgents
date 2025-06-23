#!/bin/bash
# First uninstall conflicting packages
pip uninstall -y smolagents selenium helium

# Install base packages
pip install smolagents==1.18.0

# Install selenium that's compatible with helium
pip install selenium==3.141.0

# Install helium
pip install helium==3.0.9

# Install other dependencies
pip install python-env plotly kaleido matplotlib pandas geopandas shapely

# Install additional dependencies for smolagents
pip install openai requests beautifulsoup4 lxml
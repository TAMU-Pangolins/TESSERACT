# nucres or CrossSectionThingyizer3000Prime 


# Dependencies
- Python 3
- mpmath




# Installation Instructions
    
1. Download the code:
        
        git clone https://github.com/aldusv/crosseccy


2. Install mpmath:

        pip install mpmath

3. Download the RIPL-3 data using the provided helper:

    You may edit the "USER SETTINGS" block according to your preferences found in `download_ripl.py`

    In the project folder, run

    ```
    python download_ripl.py
    ```


# Project Structure

- **nucres/**  
  - `api.py`, `__init__.py`, `constants.py`, `units.py`, `kinematics.py` — public API, physical constants, unit conversions, and basic kinematics helpers.  
  - `types.py`, `bw.py`, `penetrability.py` — resonance dataclass plus Breit–Wigner cross-section primitives.  
  - `sampling.py`, `widths.py`, `generator.py` — Porter–Thomas samplers, width utilities, and the HFB-driven cross section generator.  
  - `model.py`, `rates.py`, `hfb_adapter.py` — high-level modeling API, cross section -> reaction rate conversion, and HFB density bridges.  
  - `plotting.py` — optional visualization helpers for resonances/spectra.

- **common/**  
  HFB density table parsers (`densities_retrieval.py`).

- **tests/**  
  Unit tests for the generator and model APIs.


# nucres or CrossSectionThingyizer3000Prime 



# Installation Instructions
    
1. Download the code:
        
        git clone https://github.com/aldusv/crosseccy


2. Download the RIPL-3 data using the provided helper: (inside the project root)

    ```
    python download_ripl.py
    ```

3. Install into your environment. (inside the project root)

    ```
    pip install -e .
    ```

4. Verify the setup. (inside the project root)

    ```
    python -c "import nucres, json; print('nucres ready, data root =', nucres.config.resolve_data_root())"
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


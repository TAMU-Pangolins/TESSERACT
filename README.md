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
  - `api.py`, `__init__.py` - public API.  
  
  - `physics.py` - public API, physical constants, unit conversions, and basic kinematics helpers.  

  - `resonance.py` - resonance dataclass plus single level Breit-Wigner sigma(E) (constant and energy-dependent widths), Coulomb penetrability via Coulomb wave functions f and g. 

  - `sampling.py` - Porter–Thomas and inverse-CDF samplers, Poisson placement of levels, mean widths from strength/penetrability.

  - `generator.py` - Monte Carlo synthesis of spectra using HFB level densities. Integrates sampled resonances with Breit-Wigner shapes. 

  - `model.py`, `rates.py` - high-level modeling API, cross section -> reaction rate conversion

  - `hfb_adapter.py` - parsing HFB combinatorial level density tables.

  - `plotting.py` - optional visualization helpers for resonances/spectra.

- **common/**  

  HFB density table parsers (`densities_retrieval.py`).

- **data/**

  All data, downloaded or otherwise set in the repository. 

- **tests/**  
  Unit tests for the generator and model APIs. (temporary) 


# Developer Standards


- Prefer stdlib first (`csv`, `pathlib.Path`, etc.) and keep dependencies to the existing set (`numpy`, `mpmath`, `matplotlib`)

- Keep units consistent (MeV/eV barns as used in `nucres`), document inputs/outputs in docstrings, and return informative errors for missing datasets or bad inputs.


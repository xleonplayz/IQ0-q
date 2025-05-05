# SimOS NV Simulator Documentation

This directory contains the documentation for the SimOS NV Simulator project. The documentation is built using [Sphinx](https://www.sphinx-doc.org/en/master/) and [Doxygen](https://www.doxygen.nl/).

## Building the Documentation

### Prerequisites

- Python 3.8+
- Doxygen
- Graphviz (for diagrams)

### Setup

Install required Python packages:

```bash
pip install sphinx sphinx-rtd-theme myst-parser
```

Install Doxygen and Graphviz:

- **Linux (Ubuntu/Debian):**
  ```bash
  sudo apt-get install doxygen graphviz
  ```

- **macOS:**
  ```bash
  brew install doxygen graphviz
  ```

- **Windows:**
  Download and install [Doxygen](https://www.doxygen.nl/download.html) and [Graphviz](https://graphviz.org/download/)

### Building

To build the documentation:

1. Run Doxygen to generate the API documentation:
   ```bash
   cd docs
   doxygen Doxyfile
   ```

2. Build the Sphinx documentation:
   ```bash
   sphinx-build -b html . _build/html
   ```

3. View the documentation by opening `_build/html/index.html` in your browser

## Documentation Structure

- `physical_model.md`: Detailed explanation of the NV center physical model
- `api_reference.md`: API documentation generated from code docstrings
- `conf.py`: Sphinx configuration file
- `Doxyfile`: Doxygen configuration file
- `index.md`: Documentation homepage

## Updating the Documentation

- **Code Documentation:** Update docstrings in the code files using Doxygen-style comments
- **Physical Model Documentation:** Edit `physical_model.md` to reflect changes in the physics implementation
- **General Documentation:** Update relevant markdown files in this directory

## GitHub Wiki Integration

This documentation is automatically published to the GitHub wiki of the repository when changes are pushed to the main branch. The workflow is defined in `.github/workflows/docs-to-wiki.yml`.
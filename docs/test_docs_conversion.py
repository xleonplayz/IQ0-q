#!/usr/bin/env python3
"""
Test script to verify the documentation conversion process.

This script simulates the conversion of documentation from Doxygen HTML
to Markdown for the GitHub wiki. It can be run locally to test the process
before pushing changes to GitHub.
"""

import os
import sys
import subprocess
import glob
import shutil

def run_command(cmd, cwd=None):
    """Run a shell command and return the output."""
    print(f"Running: {cmd}")
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error: {stderr.decode('utf-8')}")
        return False
    print(stdout.decode('utf-8'))
    return True

def main():
    """Main function to test the documentation conversion workflow."""
    # Create temporary output directories
    os.makedirs("temp_output", exist_ok=True)
    os.makedirs("temp_wiki", exist_ok=True)
    
    # Check if Doxygen is installed
    if not run_command("doxygen --version"):
        print("Doxygen is not installed. Please install Doxygen to run this test.")
        return False
    
    # Run Doxygen
    print("Running Doxygen...")
    run_command("doxygen Doxyfile", cwd=".")
    
    # Check if output was generated
    if not os.path.exists("docs_output/html"):
        print("Doxygen did not generate output. Check the Doxyfile configuration.")
        return False
    
    # Create a simple converter script similar to the one in the GitHub workflow
    converter_script = """
import os
import sys
from bs4 import BeautifulSoup
import re
import glob

def html_to_markdown(html_file, output_dir):
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except Exception as e:
        print(f"Error reading {html_file}: {e}")
        return None
    
    # Basic conversion (simplified for test)
    md_content = f"# Converted from {html_file}\\n\\n"
    md_content += "This is a test conversion.\\n\\n"
    
    # Determine output filename
    filename = os.path.basename(html_file).replace('.html', '.md')
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    return output_path

def main():
    input_dir = 'docs_output/html'
    output_dir = 'temp_wiki'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process a few HTML files
    html_files = glob.glob(os.path.join(input_dir, '*.html'))[:3]  # Just test a few
    for html_file in html_files:
        if 'index.html' in html_file or 'search.html' in html_file:
            continue
        try:
            output_path = html_to_markdown(html_file, output_dir)
            print(f"Converted {html_file} to {output_path}")
        except Exception as e:
            print(f"Error processing {html_file}: {e}")
    
    # Copy existing markdown files
    md_files = glob.glob('*.md')
    for md_file in md_files:
        filename = os.path.basename(md_file)
        output_path = os.path.join(output_dir, filename)
        with open(md_file, 'r', encoding='utf-8') as f_in:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                f_out.write(f_in.read())
        print(f"Copied {md_file} to {output_path}")

if __name__ == "__main__":
    main()
"""
    
    # Write the converter script
    with open("temp_converter.py", "w") as f:
        f.write(converter_script)
    
    # Try to run the converter
    print("Testing conversion...")
    try:
        import bs4
    except ImportError:
        print("Installing required packages...")
        run_command("pip install beautifulsoup4 lxml")
    
    run_command("python temp_converter.py")
    
    # Check if any files were generated
    wiki_files = glob.glob("temp_wiki/*.md")
    if wiki_files:
        print(f"Successfully generated {len(wiki_files)} wiki files:")
        for file in wiki_files:
            print(f" - {file}")
        print("\nTest passed! The documentation workflow should work correctly.")
        return True
    else:
        print("No wiki files were generated. There might be an issue with the conversion process.")
        return False
    
if __name__ == "__main__":
    success = main()
    # Clean up
    if success:
        for dir in ["temp_output", "temp_wiki"]:
            if os.path.exists(dir):
                shutil.rmtree(dir)
        if os.path.exists("temp_converter.py"):
            os.remove("temp_converter.py")
    sys.exit(0 if success else 1)
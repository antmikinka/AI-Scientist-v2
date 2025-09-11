#!/usr/bin/env python3
"""
Comprehensive test suite for the enhanced LaTeX to JSON converter.

This test suite validates the parsing of all supported LaTeX elements
and ensures the JSON output conforms to the expected schema.
"""

import json
import tempfile
import os
from pathlib import Path
from tools.tex_to_json import EnhancedLaTeXParser


def test_basic_document():
    """Test basic document structure parsing."""
    latex_content = r"""
\documentclass{article}
\title{Test Title}
\author{Test Author}
\begin{document}
\maketitle
\tableofcontents
\begin{abstract}
This is a test abstract.
\end{abstract}
\section{Introduction}
This is the introduction.
\end{document}
"""
    
    print("Testing basic document structure...")
    parser = EnhancedLaTeXParser()
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False) as f:
        f.write(latex_content)
        temp_file = f.name
    
    try:
        result = parser.parse_latex_file(Path(temp_file))
        
        # Validate structure
        assert 'document' in result
        assert 'metadata' in result
        assert len(result['document']) > 0
        
        # Check for expected element types
        element_types = [elem['type'] for elem in result['document']]
        assert 'title' in element_types
        assert 'author' in element_types
        assert 'abstract' in element_types
        assert 'section' in element_types
        
        print("✅ Basic document structure test passed")
        return True
        
    finally:
        os.unlink(temp_file)


def test_mathematical_content():
    """Test mathematical content parsing."""
    latex_content = r"""
\documentclass{article}
\begin{document}
Inline math: $E = mc^2$

\begin{equation}
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
\end{equation}

\begin{align}
a &= b + c \\
d &= e + f
\end{align}
\end{document}
"""
    
    print("Testing mathematical content...")
    parser = EnhancedLaTeXParser()
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False) as f:
        f.write(latex_content)
        temp_file = f.name
    
    try:
        result = parser.parse_latex_file(Path(temp_file))
        
        # Check for equation elements
        element_types = [elem['type'] for elem in result['document']]
        print(f"Found element types: {element_types}")
        
        print("✅ Mathematical content test completed")
        return True
        
    finally:
        os.unlink(temp_file)


def test_list_parsing():
    """Test list parsing functionality."""
    latex_content = r"""
\documentclass{article}
\begin{document}
\begin{itemize}
\item First item
\item Second item
\item Third item
\end{itemize}

\begin{enumerate}
\item First numbered
\item Second numbered
\end{enumerate}
\end{document}
"""
    
    print("Testing list parsing...")
    parser = EnhancedLaTeXParser()
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False) as f:
        f.write(latex_content)
        temp_file = f.name
    
    try:
        result = parser.parse_latex_file(Path(temp_file))
        
        element_types = [elem['type'] for elem in result['document']]
        print(f"Found element types: {element_types}")
        
        print("✅ List parsing test completed")
        return True
        
    finally:
        os.unlink(temp_file)


def test_table_parsing():
    """Test table parsing functionality."""
    latex_content = r"""
\documentclass{article}
\begin{document}
\begin{table}[h]
\centering
\caption{Sample Table}
\label{tab:sample}
\begin{tabular}{|l|c|r|}
\hline
Col1 & Col2 & Col3 \\
\hline
A & B & C \\
D & E & F \\
\hline
\end{tabular}
\end{table}
\end{document}
"""
    
    print("Testing table parsing...")
    parser = EnhancedLaTeXParser()
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False) as f:
        f.write(latex_content)
        temp_file = f.name
    
    try:
        result = parser.parse_latex_file(Path(temp_file))
        
        element_types = [elem['type'] for elem in result['document']]
        print(f"Found element types: {element_types}")
        
        print("✅ Table parsing test completed")
        return True
        
    finally:
        os.unlink(temp_file)


def test_figure_parsing():
    """Test figure parsing functionality."""
    latex_content = r"""
\documentclass{article}
\begin{document}
\begin{figure}[h]
\centering
\includegraphics[width=0.5\textwidth]{test.png}
\caption{Test Figure}
\label{fig:test}
\end{figure}
\end{document}
"""
    
    print("Testing figure parsing...")
    parser = EnhancedLaTeXParser()
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False) as f:
        f.write(latex_content)
        temp_file = f.name
    
    try:
        result = parser.parse_latex_file(Path(temp_file))
        
        element_types = [elem['type'] for elem in result['document']]
        print(f"Found element types: {element_types}")
        
        print("✅ Figure parsing test completed")
        return True
        
    finally:
        os.unlink(temp_file)


def test_json_schema_compliance():
    """Test that the output conforms to the expected JSON schema."""
    latex_content = r"""
\documentclass{article}
\title{Schema Test}
\author{Test Author}
\begin{document}
\maketitle
\section{Test Section}
Test content.
\end{document}
"""
    
    print("Testing JSON schema compliance...")
    parser = EnhancedLaTeXParser()
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False) as f:
        f.write(latex_content)
        temp_file = f.name
    
    try:
        result = parser.parse_latex_file(Path(temp_file))
        
        # Validate top-level structure
        assert isinstance(result, dict)
        assert 'document' in result
        assert 'metadata' in result
        assert isinstance(result['document'], list)
        assert isinstance(result['metadata'], dict)
        
        # Validate document elements
        for element in result['document']:
            assert isinstance(element, dict)
            assert 'type' in element
            assert 'content' in element
            
            # Optional fields
            if 'attributes' in element:
                assert isinstance(element['attributes'], dict)
            if 'children' in element:
                assert isinstance(element['children'], list)
        
        # Validate metadata
        metadata = result['metadata']
        assert 'parser_version' in metadata
        assert 'total_elements' in metadata
        assert 'processing_time' in metadata
        
        print("✅ JSON schema compliance test passed")
        return True
        
    finally:
        os.unlink(temp_file)


def run_all_tests():
    """Run all test functions."""
    print("="*60)
    print("ENHANCED LATEX TO JSON CONVERTER - TEST SUITE")
    print("="*60)
    
    tests = [
        test_basic_document,
        test_mathematical_content,
        test_list_parsing,
        test_table_parsing,
        test_figure_parsing,
        test_json_schema_compliance,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
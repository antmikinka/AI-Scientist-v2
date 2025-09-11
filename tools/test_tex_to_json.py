#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced LaTeX to JSON Converter

This test suite validates all parsing capabilities of the enhanced tex_to_json.py
converter, ensuring robust handling of every LaTeX element type.

Author: AI-Scientist-v2 Project
License: MIT
"""

import unittest
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
import sys
import os

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from tex_to_json import (
    LaTeXTokenizer, 
    LaTeXElementParser, 
    EnhancedLaTeXParser,
    SectionParser,
    EquationParser,
    ListParser,
    JSONSchemaBuilder,
    HookManager
)


class TestLaTeXTokenizer(unittest.TestCase):
    """Test cases for LaTeX tokenization."""
    
    def setUp(self):
        self.tokenizer = LaTeXTokenizer()
    
    def test_command_tokenization(self):
        """Test parsing of LaTeX commands."""
        source = r'\section{Introduction} \textbf{bold}'
        tokens = self.tokenizer.tokenize(source)
        
        # Find command tokens
        command_tokens = [t for t in tokens if t.type == 'command']
        self.assertEqual(len(command_tokens), 2)
        self.assertEqual(command_tokens[0].content, 'section')
        self.assertEqual(command_tokens[1].content, 'textbf')
    
    def test_math_tokenization(self):
        """Test parsing of mathematical expressions."""
        source = r'This is $E = mc^2$ and $$\frac{1}{2}$$'
        tokens = self.tokenizer.tokenize(source)
        
        # Find math tokens
        math_tokens = [t for t in tokens if t.type in ['inline_math', 'display_math']]
        self.assertEqual(len(math_tokens), 2)
        self.assertEqual(math_tokens[0].content, 'E = mc^2')
        self.assertEqual(math_tokens[1].content, r'\frac{1}{2}')
    
    def test_brace_group_tokenization(self):
        """Test parsing of brace groups."""
        source = r'\section{Introduction} \textbf{bold text}'
        tokens = self.tokenizer.tokenize(source)
        
        # Find brace group tokens
        brace_tokens = [t for t in tokens if t.type == 'brace_group']
        self.assertEqual(len(brace_tokens), 2)
        self.assertEqual(brace_tokens[0].content, 'Introduction')
        self.assertEqual(brace_tokens[1].content, 'bold text')
    
    def test_text_tokenization(self):
        """Test parsing of regular text."""
        source = r'This is regular text with some words.'
        tokens = self.tokenizer.tokenize(source)
        
        # Should have text tokens
        text_tokens = [t for t in tokens if t.type == 'text']
        self.assertGreater(len(text_tokens), 0)
    
    def test_comment_removal(self):
        """Test removal of LaTeX comments."""
        source = r'''
        \section{Introduction}  % This is a comment
        This is text.  % Another comment
        '''
        tokens = self.tokenizer.tokenize(source)
        
        # Comments should be removed
        comment_tokens = [t for t in tokens if 'comment' in t.content.lower()]
        self.assertEqual(len(comment_tokens), 0)


class TestElementParsers(unittest.TestCase):
    """Test cases for element-specific parsers."""
    
    def setUp(self):
        self.tokenizer = LaTeXTokenizer()
    
    def test_section_parsing(self):
        """Test parsing of sections and subsections."""
        source = r'''
        \section{Introduction}
        This is the introduction text.
        
        \subsection{Background}
        This is background information.
        '''
        
        tokens = self.tokenizer.tokenize(source)
        parser = SectionParser()
        
        # Find section start
        section_pos = None
        for i, token in enumerate(tokens):
            if token.type == 'command' and token.content == 'section':
                section_pos = i
                break
        
        self.assertIsNotNone(section_pos)
        
        # Parse section
        element, new_pos = parser.parse(tokens, section_pos)
        
        self.assertEqual(element.type, 'section')
        self.assertEqual(element.attributes['title'], 'Introduction')
        self.assertEqual(element.attributes['level'], 1)
        self.assertGreater(len(element.content), 0)
    
    def test_equation_parsing(self):
        """Test parsing of equation environments."""
        source = r'''
        \begin{equation}
        E = mc^2
        \end{equation}
        '''
        
        tokens = self.tokenizer.tokenize(source)
        parser = EquationParser()
        
        # Find equation start
        eq_pos = None
        for i, token in enumerate(tokens):
            if parser.can_parse(tokens, i):
                eq_pos = i
                break
        
        self.assertIsNotNone(eq_pos)
        
        # Parse equation
        element, new_pos = parser.parse(tokens, eq_pos)
        
        self.assertEqual(element.type, 'equation')
        self.assertEqual(element.attributes['environment'], 'equation')
        self.assertTrue(element.attributes['numbered'])
        self.assertIn('mc^2', element.content)
    
    def test_list_parsing(self):
        """Test parsing of itemize and enumerate lists."""
        source = r'''
        \begin{itemize}
        \item First item
        \item Second item
        \end{itemize}
        '''
        
        tokens = self.tokenizer.tokenize(source)
        parser = ListParser()
        
        # Find list start
        list_pos = None
        for i, token in enumerate(tokens):
            if parser.can_parse(tokens, i):
                list_pos = i
                break
        
        self.assertIsNotNone(list_pos)
        
        # Parse list
        element, new_pos = parser.parse(tokens, list_pos)
        
        self.assertEqual(element.type, 'itemize_list')
        self.assertEqual(element.attributes['list_type'], 'itemize')
        self.assertEqual(element.attributes['item_count'], 2)
        self.assertEqual(len(element.content), 2)


class TestJSONSchemaBuilder(unittest.TestCase):
    """Test cases for JSON schema building and validation."""
    
    def setUp(self):
        self.builder = JSONSchemaBuilder()
    
    def test_json_structure(self):
        """Test basic JSON structure building."""
        from tex_to_json import ParsedElement
        
        # Create sample elements
        elements = [
            ParsedElement(
                type='section',
                content=[],
                attributes={'title': 'Introduction', 'level': 1}
            ),
            ParsedElement(
                type='paragraph',
                content='This is a paragraph.',
                attributes={'word_count': 4}
            )
        ]
        
        json_output = self.builder.build_json(elements, processing_time=1.5)
        
        # Validate structure
        self.assertIn('document', json_output)
        self.assertIn('metadata', json_output)
        self.assertEqual(len(json_output['document']), 2)
        self.assertEqual(json_output['metadata']['total_elements'], 2)
        self.assertEqual(json_output['metadata']['processing_time'], 1.5)
    
    def test_schema_validation(self):
        """Test JSON schema validation."""
        # Valid JSON structure
        valid_json = {
            'document': [
                {'type': 'paragraph', 'content': 'Test content'}
            ],
            'metadata': {
                'parser_version': '2.0.0',
                'total_elements': 1
            }
        }
        
        self.assertTrue(self.builder.validate_schema(valid_json))
        
        # Invalid JSON structure
        invalid_json = {
            'document': [
                {'content': 'Missing type field'}
            ]
        }
        
        self.assertFalse(self.builder.validate_schema(invalid_json))


class TestHookManager(unittest.TestCase):
    """Test cases for hook management system."""
    
    def setUp(self):
        self.hook_manager = HookManager()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_pre_hook_execution(self):
        """Test pre-hook script execution."""
        # Create a simple pre-hook script
        hook_script = self.temp_dir / 'pre_hook.py'
        hook_script.write_text('''
import sys
print(f"Processing file: {sys.argv[1]}")
''')
        
        # Create a dummy tex file
        tex_file = self.temp_dir / 'test.tex'
        tex_file.write_text(r'\section{Test}')
        
        # Execute hook
        result = self.hook_manager.execute_pre_hook(str(hook_script), str(tex_file))
        self.assertTrue(result)
    
    def test_post_hook_execution(self):
        """Test post-hook script execution."""
        # Create a simple post-hook script
        hook_script = self.temp_dir / 'post_hook.py'
        hook_script.write_text('''
import sys
import json

# Read JSON file
with open(sys.argv[1], 'r') as f:
    data = json.load(f)

print(f"Processed {len(data.get('document', []))} elements")
''')
        
        # Create a dummy JSON file
        json_file = self.temp_dir / 'test.json'
        json_file.write_text('{"document": [{"type": "test", "content": "test"}]}')
        
        # Execute hook
        result = self.hook_manager.execute_post_hook(str(hook_script), str(json_file))
        self.assertTrue(result)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete parsing workflow."""
    
    def setUp(self):
        self.parser = EnhancedLaTeXParser()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_complete_document_parsing(self):
        """Test parsing of a complete LaTeX document."""
        # Create a comprehensive test LaTeX document
        latex_content = r'''
\documentclass{article}
\title{Test Document}
\author{Test Author}

\begin{document}

\maketitle

\begin{abstract}
This is the abstract of the test document.
\end{abstract}

\tableofcontents

\section{Introduction}
This is the introduction section with some text and a citation \cite{test2023}.

The fundamental equation is $E = mc^2$ which represents energy-mass equivalence.

\subsection{Mathematical Content}
Here is a display equation:
\begin{equation}
\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
\end{equation}

\section{Lists and Structure}
Here are some key points:
\begin{itemize}
\item First important point
\item Second important point with $\alpha + \beta$
\item Third point
\end{itemize}

Numbered list:
\begin{enumerate}
\item Step one
\item Step two
\item Step three
\end{enumerate}

\section{Conclusion}
This concludes our test document.

\end{document}
        '''
        
        # Write to temporary file
        tex_file = self.temp_dir / 'test_document.tex'
        tex_file.write_text(latex_content)
        
        # Parse the document
        result = self.parser.parse_latex_file(tex_file)
        
        # Validate results
        self.assertIn('document', result)
        self.assertIn('metadata', result)
        self.assertGreater(result['metadata']['total_elements'], 0)
        
        # Check for specific element types
        elements = result['document']
        element_types = {elem['type'] for elem in elements}
        
        # Should contain various element types
        expected_types = {'section', 'paragraph', 'equation', 'itemize_list', 'enumerate_list'}
        found_types = element_types & expected_types
        self.assertGreater(len(found_types), 0, f"Expected some of {expected_types}, found {element_types}")
    
    def test_malformed_latex_handling(self):
        """Test handling of malformed LaTeX input."""
        # Create LaTeX with intentional errors
        malformed_latex = r'''
\section{Test Section
This section has unmatched braces {{{ and $$$ math errors.
\begin{itemize}
\item Unclosed item
\begin{equation}
Unclosed equation
'''
        
        # Write to temporary file
        tex_file = self.temp_dir / 'malformed.tex'
        tex_file.write_text(malformed_latex)
        
        # Parse should not crash
        result = self.parser.parse_latex_file(tex_file)
        
        # Should have error information
        self.assertIn('metadata', result)
        self.assertTrue(result['metadata'].get('has_errors', False))
        
        # Should still extract some content
        self.assertIn('document', result)
    
    def test_empty_document(self):
        """Test handling of empty or minimal documents."""
        # Create minimal LaTeX document
        minimal_latex = r'\documentclass{article}'
        
        # Write to temporary file
        tex_file = self.temp_dir / 'minimal.tex'
        tex_file.write_text(minimal_latex)
        
        # Parse should work
        result = self.parser.parse_latex_file(tex_file)
        
        # Should have valid structure
        self.assertIn('document', result)
        self.assertIn('metadata', result)
        self.assertEqual(result['metadata']['parser_version'], '2.0.0')


class TestCLIInterface(unittest.TestCase):
    """Test cases for command-line interface."""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_cli_basic_usage(self):
        """Test basic CLI functionality."""
        import subprocess
        
        # Create a simple LaTeX file
        tex_file = self.temp_dir / 'test.tex'
        tex_file.write_text(r'''
\section{Test}
This is a test document.
        ''')
        
        json_file = self.temp_dir / 'output.json'
        
        # Run the CLI
        result = subprocess.run([
            sys.executable, 
            str(Path(__file__).parent / 'tex_to_json.py'),
            '--tex_file', str(tex_file),
            '--json_file', str(json_file)
        ], capture_output=True, text=True)
        
        # Should succeed
        self.assertEqual(result.returncode, 0)
        
        # Output file should exist and be valid JSON
        self.assertTrue(json_file.exists())
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.assertIn('document', data)
        self.assertIn('metadata', data)


class TestPerformance(unittest.TestCase):
    """Performance tests for large documents."""
    
    def test_large_document_performance(self):
        """Test performance with a large document."""
        import time
        
        # Generate a large LaTeX document
        sections = []
        for i in range(50):  # 50 sections
            sections.append(f'''
\\section{{Section {i+1}}}
This is section {i+1} with multiple paragraphs and content.

Here is some mathematical content: $x^{i+1} + y^{i+1} = z^{i+1}$.

\\begin{{itemize}}
\\item First item in section {i+1}
\\item Second item in section {i+1}  
\\item Third item in section {i+1}
\\end{{itemize}}

\\begin{{equation}}
\\sum_{{k=1}}^{{{i+1}}} k^2 = \\frac{{{i+1}({i+1}+1)(2 \\cdot {i+1}+1)}}{{6}}
\\end{{equation}}
''')
        
        large_latex = r'''
\documentclass{article}
\title{Large Test Document}
\begin{document}
''' + '\n'.join(sections) + r'''
\end{document}
'''
        
        # Parse and time it
        parser = EnhancedLaTeXParser()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False) as f:
            f.write(large_latex)
            temp_file = Path(f.name)
        
        try:
            start_time = time.time()
            result = parser.parse_latex_file(temp_file)
            processing_time = time.time() - start_time
            
            # Should complete in reasonable time (less than 30 seconds)
            self.assertLess(processing_time, 30.0)
            
            # Should extract significant number of elements
            self.assertGreater(result['metadata']['total_elements'], 100)
            
        finally:
            temp_file.unlink()


def create_test_sample_files():
    """Create sample LaTeX files for manual testing."""
    samples_dir = Path(__file__).parent / 'test_samples'
    samples_dir.mkdir(exist_ok=True)
    
    # Simple academic paper
    simple_paper = r'''
\documentclass{article}
\usepackage{amsmath}

\title{A Simple Academic Paper}
\author{Test Author \\ University of Testing}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
This is a simple academic paper for testing the enhanced LaTeX to JSON converter.
It contains various elements including sections, equations, lists, and references.
\end{abstract}

\section{Introduction}
\label{sec:intro}

This paper demonstrates the capabilities of our enhanced LaTeX parser.
The fundamental equation of relativity is $E = mc^2$.

\subsection{Problem Statement}

We need to parse every element of a LaTeX document into structured JSON format.

\section{Methodology}

Our approach involves several steps:
\begin{enumerate}
\item Tokenize the LaTeX source
\item Parse elements hierarchically  
\item Generate structured JSON output
\end{enumerate}

\section{Mathematical Formulation}

The loss function is defined as:
\begin{equation}
\label{eq:loss}
L(\theta) = \frac{1}{n} \sum_{i=1}^{n} \ell(f_\theta(x_i), y_i)
\end{equation}

For optimization, we use gradient descent:
\begin{align}
\theta_{t+1} &= \theta_t - \alpha \nabla L(\theta_t) \\
&= \theta_t - \alpha \frac{1}{n} \sum_{i=1}^{n} \nabla \ell(f_\theta(x_i), y_i)
\end{align}

\section{Results}

Key findings include:
\begin{itemize}
\item Comprehensive element extraction
\item Hierarchical structure preservation
\item Robust error handling
\item Extensible hook system
\end{itemize}

\section{Conclusion}

We have successfully implemented a comprehensive LaTeX to JSON converter
that extracts every element with high fidelity. Future work includes
adding support for tables and figures.

\end{document}
'''
    
    (samples_dir / 'simple_paper.tex').write_text(simple_paper)
    
    # Complex mathematical document
    math_doc = r'''
\documentclass{article}
\usepackage{amsmath, amssymb, amsthm}

\title{Mathematical Document Test}
\begin{document}

\section{Advanced Mathematics}

\subsection{Complex Equations}

Multi-line equation:
\begin{align}
\nabla \times \mathbf{E} &= -\frac{\partial \mathbf{B}}{\partial t} \\
\nabla \times \mathbf{B} &= \mu_0 \mathbf{J} + \mu_0 \epsilon_0 \frac{\partial \mathbf{E}}{\partial t} \\
\nabla \cdot \mathbf{E} &= \frac{\rho}{\epsilon_0} \\
\nabla \cdot \mathbf{B} &= 0
\end{align}

Matrix equation:
\begin{equation}
\mathbf{A} = \begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{pmatrix}
\end{equation}

\subsection{Inline Mathematics}

The function $f(x) = \sum_{n=0}^{\infty} \frac{x^n}{n!} = e^x$ is the exponential function.
We have $\lim_{x \to 0} \frac{\sin x}{x} = 1$ and $\int_0^1 x^2 dx = \frac{1}{3}$.

\end{document}
'''
    
    (samples_dir / 'math_document.tex').write_text(math_doc)
    
    print(f"Created test samples in: {samples_dir}")
    return samples_dir


def run_all_tests():
    """Run all test cases."""
    print("Running Enhanced LaTeX Parser Test Suite")
    print("=" * 50)
    
    # Create test samples
    samples_dir = create_test_sample_files()
    
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('test_tex_to_json')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed successfully!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
    
    print(f"\nTest samples created in: {samples_dir}")
    print("You can test manually with:")
    print(f"python tex_to_json.py -t {samples_dir}/simple_paper.tex -j output.json -v")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run all tests if called directly
    success = run_all_tests()
    sys.exit(0 if success else 1)
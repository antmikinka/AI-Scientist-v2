#!/usr/bin/env python3
"""
Enhanced LaTeX to JSON Converter for AI-Scientist-v2

This tool provides comprehensive extraction of every element from LaTeX documents
and converts them to structured JSON format for advanced AI analysis and processing.

Features:
- Granular element extraction (sections, paragraphs, equations, lists, tables, figures)
- Hierarchical document structure preservation
- Pre/post-processing hooks for extensibility
- Robust error handling and validation
- Comprehensive JSON schema with metadata

Author: AI-Scientist-v2 Project (Enhanced Version)
License: MIT
"""

import argparse
import json
import re
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Token:
    """Represents a LaTeX token with metadata."""
    type: str
    content: str
    line_number: int
    column: int
    raw_text: str = ""


@dataclass 
class ParsedElement:
    """Represents a parsed LaTeX element."""
    type: str
    content: Any
    attributes: Dict[str, Any] = field(default_factory=dict)
    children: List['ParsedElement'] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert element to dictionary representation."""
        result = {
            "type": self.type,
            "content": self.content
        }
        
        if self.attributes:
            result["attributes"] = self.attributes
            
        if self.children:
            result["children"] = [child.to_dict() for child in self.children]
            
        return result


class LaTeXTokenizer:
    """
    Advanced LaTeX tokenizer that handles complex LaTeX structures.
    
    Tokenizes LaTeX source into structured tokens for hierarchical parsing.
    Handles commands, environments, text, math, comments, and special characters.
    """
    
    def __init__(self):
        self.tokens = []
        self.position = 0
        self.line_number = 1
        self.column = 1
        
        # Token patterns for different LaTeX constructs
        self.patterns = {
            'command': re.compile(r'\\([a-zA-Z]+)(\*?)'),
            'begin_env': re.compile(r'\\begin\s*\{([^}]+)\}'),
            'end_env': re.compile(r'\\end\s*\{([^}]+)\}'),
            'inline_math': re.compile(r'\$([^$]+)\$'),
            'display_math': re.compile(r'\$\$([^$]+)\$\$'),
            'comment': re.compile(r'%.*$', re.MULTILINE),
            'brace_group': re.compile(r'\{([^{}]*)\}'),
            'bracket_group': re.compile(r'\[([^\[\]]*)\]'),
        }
    
    def tokenize(self, latex_source: str) -> List[Token]:
        """
        Tokenize LaTeX source into structured tokens.
        
        Args:
            latex_source: Raw LaTeX document content
            
        Returns:
            List of tokens representing the document structure
        """
        self.tokens = []
        self.position = 0
        self.line_number = 1
        self.column = 1
        
        # Remove comments first but preserve line structure
        cleaned_source = self._preprocess_source(latex_source)
        
        i = 0
        while i < len(cleaned_source):
            char = cleaned_source[i]
            
            # Track line and column numbers
            if char == '\n':
                self.line_number += 1
                self.column = 1
            else:
                self.column += 1
            
            # Tokenize based on character
            if char == '\\':
                token, consumed = self._tokenize_command(cleaned_source[i:])
                if token:
                    self.tokens.append(token)
                i += consumed
            elif char == '$':
                token, consumed = self._tokenize_math(cleaned_source[i:])
                if token:
                    self.tokens.append(token)
                i += consumed
            elif char == '{':
                token, consumed = self._tokenize_brace_group(cleaned_source[i:])
                if token:
                    self.tokens.append(token)
                i += consumed
            elif char == '[':
                token, consumed = self._tokenize_bracket_group(cleaned_source[i:])
                if token:
                    self.tokens.append(token)
                i += consumed
            elif char.isspace():
                # Skip whitespace but track position
                i += 1
            else:
                # Regular text
                token, consumed = self._tokenize_text(cleaned_source[i:])
                if token:
                    self.tokens.append(token)
                i += consumed
        
        return self.tokens
    
    def _preprocess_source(self, source: str) -> str:
        """Preprocess source to handle comments and normalize whitespace."""
        # Remove LaTeX comments but preserve line structure
        lines = source.split('\n')
        cleaned_lines = []
        for line in lines:
            comment_pos = line.find('%')
            if comment_pos != -1:
                # Check if % is escaped
                if comment_pos == 0 or line[comment_pos-1] != '\\':
                    line = line[:comment_pos]
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _tokenize_command(self, text: str) -> Tuple[Optional[Token], int]:
        """Tokenize LaTeX commands like \\section, \\begin, etc."""
        if not text.startswith('\\'):
            return None, 1
        
        # Simple command matching
        i = 1  # Skip the backslash
        while i < len(text) and text[i].isalpha():
            i += 1
        
        if i > 1:  # Found a command
            command = text[1:i]
            star = ""
            
            # Check for starred version
            if i < len(text) and text[i] == '*':
                star = "*"
                i += 1
            
            full_command = f"\\{command}{star}"
            
            token = Token(
                type='command',
                content=command,
                line_number=self.line_number,
                column=self.column,
                raw_text=full_command
            )
            
            return token, i
        
        return None, 1
    
    def _tokenize_math(self, text: str) -> Tuple[Optional[Token], int]:
        """Tokenize mathematical expressions."""
        # Check for display math ($$...$$)
        if text.startswith('$$'):
            match = self.patterns['display_math'].match(text)
            if match:
                content = match.group(1)
                token = Token(
                    type='display_math',
                    content=content,
                    line_number=self.line_number,
                    column=self.column,
                    raw_text=match.group(0)
                )
                return token, len(match.group(0))
        
        # Check for inline math ($...$)
        elif text.startswith('$'):
            match = self.patterns['inline_math'].match(text)
            if match:
                content = match.group(1)
                token = Token(
                    type='inline_math',
                    content=content,
                    line_number=self.line_number,
                    column=self.column,
                    raw_text=match.group(0)
                )
                return token, len(match.group(0))
        
        return None, 1
    
    def _tokenize_brace_group(self, text: str) -> Tuple[Optional[Token], int]:
        """Tokenize brace groups {...}."""
        if not text.startswith('{'):
            return None, 1
        
        # Find matching closing brace
        brace_count = 0
        i = 0
        for char in text:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    content = text[1:i]  # Exclude braces
                    token = Token(
                        type='brace_group',
                        content=content,
                        line_number=self.line_number,
                        column=self.column,
                        raw_text=text[:i+1]
                    )
                    return token, i + 1
            i += 1
        
        # Unmatched brace
        return None, 1
    
    def _tokenize_bracket_group(self, text: str) -> Tuple[Optional[Token], int]:
        """Tokenize bracket groups [...]."""
        if not text.startswith('['):
            return None, 1
        
        # Find matching closing bracket
        bracket_count = 0
        i = 0
        for char in text:
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    content = text[1:i]  # Exclude brackets
                    token = Token(
                        type='bracket_group',
                        content=content,
                        line_number=self.line_number,
                        column=self.column,
                        raw_text=text[:i+1]
                    )
                    return token, i + 1
            i += 1
        
        # Unmatched bracket
        return None, 1
    
    def _tokenize_text(self, text: str) -> Tuple[Optional[Token], int]:
        """Tokenize regular text content."""
        # Find next special character
        special_chars = ['\\', '$', '{', '}', '[', ']', '\n']
        
        i = 0
        while i < len(text) and text[i] not in special_chars:
            i += 1
        
        if i > 0:
            content = text[:i]
            token = Token(
                type='text',
                content=content.strip(),
                line_number=self.line_number,
                column=self.column,
                raw_text=content
            )
            return token, i
        
        return None, 1


class ElementParser(ABC):
    """Abstract base class for element-specific parsers."""
    
    @abstractmethod
    def can_parse(self, tokens: List[Token], position: int) -> bool:
        """Check if this parser can handle the current token sequence."""
        pass
    
    @abstractmethod
    def parse(self, tokens: List[Token], position: int) -> Tuple[ParsedElement, int]:
        """Parse element and return parsed element with new position."""
        pass


class SectionParser(ElementParser):
    """Parser for section, subsection, and subsubsection elements."""
    
    SECTION_COMMANDS = {
        'section': 1,
        'subsection': 2, 
        'subsubsection': 3,
        'paragraph': 4,
        'subparagraph': 5
    }
    
    def can_parse(self, tokens: List[Token], position: int) -> bool:
        if position >= len(tokens):
            return False
        
        token = tokens[position]
        return (token.type == 'command' and 
                token.content in self.SECTION_COMMANDS)
    
    def parse(self, tokens: List[Token], position: int) -> Tuple[ParsedElement, int]:
        token = tokens[position]
        section_type = token.content
        level = self.SECTION_COMMANDS[section_type]
        
        # Look for title in next brace group
        title = ""
        pos = position + 1
        
        if pos < len(tokens) and tokens[pos].type == 'brace_group':
            title = tokens[pos].content
            pos += 1
        
        # Parse section content until next section or end
        content_elements = []
        while pos < len(tokens):
            next_token = tokens[pos]
            
            # Stop at next section of same or higher level
            if (next_token.type == 'command' and 
                next_token.content in self.SECTION_COMMANDS and
                self.SECTION_COMMANDS[next_token.content] <= level):
                break
            
            # Skip this for now to avoid circular dependency
            # We'll collect content differently
            pos += 1
        
        element = ParsedElement(
            type=section_type,
            content=content_elements,
            attributes={
                'title': title,
                'level': level,
                'command': f'\\{section_type}',
                'line_number': token.line_number
            }
        )
        
        return element, pos


class EquationParser(ElementParser):
    """Parser for equation environments and mathematical content."""
    
    MATH_ENVIRONMENTS = {
        'equation', 'align', 'gather', 'split', 'multline',
        'eqnarray', 'alignat', 'flalign'
    }
    
    def can_parse(self, tokens: List[Token], position: int) -> bool:
        if position >= len(tokens):
            return False
        
        token = tokens[position]
        if token.type == 'command' and token.content == 'begin':
            # Check if next token is math environment
            if position + 1 < len(tokens):
                env_token = tokens[position + 1]
                if env_token.type == 'brace_group':
                    return env_token.content in self.MATH_ENVIRONMENTS
        
        return False
    
    def parse(self, tokens: List[Token], position: int) -> Tuple[ParsedElement, int]:
        # Skip \begin
        position += 1
        
        # Get environment name
        env_token = tokens[position]
        env_name = env_token.content
        position += 1
        
        # Collect equation content until \end{env_name}
        equation_content = []
        start_line = tokens[position].line_number
        while position < len(tokens):
            token = tokens[position]
            
            if (token.type == 'command' and token.content == 'end' and
                position + 1 < len(tokens) and
                tokens[position + 1].type == 'brace_group' and
                tokens[position + 1].content == env_name):
                # Found matching end
                position += 2  # Skip \end and environment name
                break
            
            # Collect content
            if token.type in ['text', 'command', 'brace_group', 'inline_math', 'display_math']:
                equation_content.append(token.raw_text)
            
            position += 1
        
        content = ''.join(equation_content).strip()
        
        element = ParsedElement(
            type='equation',
            content=content,
            attributes={
                'environment': env_name,
                'numbered': env_name != 'equation*',
                'line_number': start_line
            }
        )
        
        return element, position


class ListParser(ElementParser):
    """Parser for itemize, enumerate, and description lists."""
    
    LIST_ENVIRONMENTS = {'itemize', 'enumerate', 'description'}
    
    def can_parse(self, tokens: List[Token], position: int) -> bool:
        if position >= len(tokens):
            return False
        
        token = tokens[position]
        if token.type == 'command' and token.content == 'begin':
            if position + 1 < len(tokens):
                env_token = tokens[position + 1]
                if env_token.type == 'brace_group':
                    return env_token.content in self.LIST_ENVIRONMENTS
        
        return False
    
    def parse(self, tokens: List[Token], position: int) -> Tuple[ParsedElement, int]:
        # Skip \begin
        position += 1
        
        # Get list environment name
        env_token = tokens[position]
        list_type = env_token.content
        position += 1
        
        # Parse list items
        list_items = []
        
        while position < len(tokens):
            token = tokens[position]
            
            # Check for end of list
            if (token.type == 'command' and token.content == 'end' and
                position + 1 < len(tokens) and
                tokens[position + 1].type == 'brace_group' and
                tokens[position + 1].content == list_type):
                position += 2  # Skip \end and environment name
                break
            
            # Check for new item
            if token.type == 'command' and token.content == 'item':
                item_element, position = self._parse_list_item(tokens, position + 1, list_type, len(list_items) + 1)
                if item_element:
                    list_items.append(item_element)
                continue

            position += 1
        
        element = ParsedElement(
            type=f'{list_type}_list',
            content=[item.to_dict() for item in list_items],
            attributes={
                'list_type': list_type,
                'item_count': len(list_items),
                'max_depth': 1
            }
        )
        
        return element, position

    def _parse_list_item(self, tokens: List[Token], position: int, list_type: str, item_number: int) -> Tuple[Optional[ParsedElement], int]:
        content_elements = []
        while position < len(tokens):
            token = tokens[position]
            if (token.type == 'command' and (token.content == 'item' or (token.content == 'end' and position + 1 < len(tokens) and tokens[position+1].type == 'brace_group' and tokens[position+1].content == list_type))):
                break

            # Collect raw text content for now
            if tokens[position].type == 'text' and tokens[position].content.strip():
                content_elements.append(tokens[position].content.strip())
            position += 1
        
        if not content_elements:
            return None, position

        item = ParsedElement(
            type='list_item',
            content=' '.join(content_elements),
            attributes={
                'marker': self._get_list_marker(list_type, item_number),
                'level': 1
            }
        )
        return item, position

    def _get_list_marker(self, list_type: str, item_number: int) -> str:
        """Get appropriate marker for list item."""
        if list_type == 'itemize':
            return 'bullet'
        elif list_type == 'enumerate':
            return f'{item_number}.'
        elif list_type == 'description':
            return 'description'
        return 'unknown'


class TableParser(ElementParser):
    """Parser for table environments and tabular content."""
    
    TABLE_ENVIRONMENTS = {'table', 'tabular', 'longtable', 'tabulary', 'supertabular'}
    
    def can_parse(self, tokens: List[Token], position: int) -> bool:
        if position >= len(tokens):
            return False
        
        token = tokens[position]
        if token.type == 'command' and token.content == 'begin':
            if position + 1 < len(tokens):
                env_token = tokens[position + 1]
                if env_token.type == 'brace_group':
                    return env_token.content in self.TABLE_ENVIRONMENTS
        
        return False
    
    def parse(self, tokens: List[Token], position: int) -> Tuple[ParsedElement, int]:
        # Skip \begin
        position += 1
        
        # Get table environment name
        env_token = tokens[position]
        table_type = env_token.content
        position += 1
        
        # Look for table options (e.g., column specifications)
        table_options = ""
        if position < len(tokens) and tokens[position].type == 'bracket_group':
            table_options = tokens[position].content
            position += 1
        
        # Collect table content
        table_content = []
        caption = ""
        label = ""
        
        while position < len(tokens):
            token = tokens[position]
            
            # Check for end of table
            if (token.type == 'command' and token.content == 'end' and
                position + 1 < len(tokens) and
                tokens[position + 1].type == 'brace_group' and
                tokens[position + 1].content == table_type):
                position += 2  # Skip \end and environment name
                break
            
            # Check for caption
            if token.type == 'command' and token.content == 'caption':
                if position + 1 < len(tokens) and tokens[position + 1].type == 'brace_group':
                    caption = tokens[position + 1].content
                    position += 2
                    continue
            
            # Check for label
            if token.type == 'command' and token.content == 'label':
                if position + 1 < len(tokens) and tokens[position + 1].type == 'brace_group':
                    label = tokens[position + 1].content
                    position += 2
                    continue
            
            # Collect table content
            table_content.append(token.raw_text)
            position += 1
        
        # Parse table structure (simplified)
        rows = []
        current_row = []
        for content in table_content:
            if '\\\\' in content:  # End of row
                if current_row:
                    rows.append(current_row)
                    current_row = []
            elif '&' in content:  # Column separator
                cells = content.split('&')
                current_row.extend([cell.strip() for cell in cells])
            elif content.strip():
                current_row.append(content.strip())
        
        if current_row:  # Last row
            rows.append(current_row)
        
        element = ParsedElement(
            type='table',
            content=rows,
            attributes={
                'environment': table_type,
                'caption': caption,
                'label': label,
                'options': table_options,
                'rows': len(rows),
                'columns': max(len(row) for row in rows) if rows else 0
            }
        )
        
        return element, position


class FigureParser(ElementParser):
    """Parser for figure environments and graphics content."""
    
    FIGURE_ENVIRONMENTS = {'figure', 'figure*'}
    
    def can_parse(self, tokens: List[Token], position: int) -> bool:
        if position >= len(tokens):
            return False
        
        token = tokens[position]
        if token.type == 'command' and token.content == 'begin':
            if position + 1 < len(tokens):
                env_token = tokens[position + 1]
                if env_token.type == 'brace_group':
                    return env_token.content in self.FIGURE_ENVIRONMENTS
        
        return False
    
    def parse(self, tokens: List[Token], position: int) -> Tuple[ParsedElement, int]:
        # Skip \begin
        position += 1
        
        # Get figure environment name
        env_token = tokens[position]
        figure_type = env_token.content
        position += 1
        
        # Look for figure placement options
        figure_options = ""
        if position < len(tokens) and tokens[position].type == 'bracket_group':
            figure_options = tokens[position].content
            position += 1
        
        # Collect figure content
        caption = ""
        label = ""
        graphics_files = []
        other_content = []
        
        while position < len(tokens):
            token = tokens[position]
            
            # Check for end of figure
            if (token.type == 'command' and token.content == 'end' and
                position + 1 < len(tokens) and
                tokens[position + 1].type == 'brace_group' and
                tokens[position + 1].content == figure_type):
                position += 2  # Skip \end and environment name
                break
            
            # Check for includegraphics
            if token.type == 'command' and token.content == 'includegraphics':
                graphics_options = ""
                if position + 1 < len(tokens) and tokens[position + 1].type == 'bracket_group':
                    graphics_options = tokens[position + 1].content
                    position += 1
                
                if position + 1 < len(tokens) and tokens[position + 1].type == 'brace_group':
                    graphics_file = tokens[position + 1].content
                    graphics_files.append({
                        'file': graphics_file,
                        'options': graphics_options
                    })
                    position += 2
                    continue
            
            # Check for caption
            if token.type == 'command' and token.content == 'caption':
                if position + 1 < len(tokens) and tokens[position + 1].type == 'brace_group':
                    caption = tokens[position + 1].content
                    position += 2
                    continue
            
            # Check for label
            if token.type == 'command' and token.content == 'label':
                if position + 1 < len(tokens) and tokens[position + 1].type == 'brace_group':
                    label = tokens[position + 1].content
                    position += 2
                    continue
            
            # Collect other content
            if token.type == 'text' and token.content.strip():
                other_content.append(token.content.strip())
            
            position += 1
        
        element = ParsedElement(
            type='figure',
            content=graphics_files,
            attributes={
                'environment': figure_type,
                'caption': caption,
                'label': label,
                'placement': figure_options,
                'graphics_count': len(graphics_files),
                'other_content': ' '.join(other_content) if other_content else ""
            }
        )
        
        return element, position


class TitleAuthorParser(ElementParser):
    """Parser for document title, author, and metadata commands."""
    
    TITLE_COMMANDS = {
        'title', 'author', 'date', 'maketitle', 'thanks',
        'icmltitle', 'icmlauthor', 'icmlaffiliation'
    }
    
    def can_parse(self, tokens: List[Token], position: int) -> bool:
        if position >= len(tokens):
            return False
        
        token = tokens[position]
        return (token.type == 'command' and 
                token.content in self.TITLE_COMMANDS)
    
    def parse(self, tokens: List[Token], position: int) -> Tuple[ParsedElement, int]:
        token = tokens[position]
        command = token.content
        position += 1
        
        # Get content if present
        content = ""
        if position < len(tokens) and tokens[position].type == 'brace_group':
            content = tokens[position].content
            position += 1
        
        # Map command to element type
        element_type = command
        if command in ['icmltitle']:
            element_type = 'title'
        elif command in ['icmlauthor']:
            element_type = 'author'
        elif command in ['icmlaffiliation']:
            element_type = 'affiliation'
        
        element = ParsedElement(
            type=element_type,
            content=content,
            attributes={
                'command': f'\\{command}',
                'line_number': token.line_number
            }
        )
        
        return element, position


class AbstractParser(ElementParser):
    """Parser for abstract environment."""
    
    def can_parse(self, tokens: List[Token], position: int) -> bool:
        if position >= len(tokens):
            return False
        
        token = tokens[position]
        if token.type == 'command' and token.content == 'begin':
            if position + 1 < len(tokens):
                env_token = tokens[position + 1]
                if env_token.type == 'brace_group':
                    return env_token.content == 'abstract'
        
        return False
    
    def parse(self, tokens: List[Token], position: int) -> Tuple[ParsedElement, int]:
        # Skip \begin
        position += 1
        
        # Skip {abstract}
        position += 1
        
        # Collect abstract content
        abstract_content = []
        
        while position < len(tokens):
            token = tokens[position]
            
            # Check for end of abstract
            if (token.type == 'command' and token.content == 'end' and
                position + 1 < len(tokens) and
                tokens[position + 1].type == 'brace_group' and
                tokens[position + 1].content == 'abstract'):
                position += 2  # Skip \end and {abstract}
                break
            
            # Collect content
            if token.type == 'text' and token.content.strip():
                abstract_content.append(token.content.strip())
            
            position += 1
        
        content = ' '.join(abstract_content)
        
        element = ParsedElement(
            type='abstract',
            content=[{
                'type': 'paragraph',
                'content': content
            }],
            attributes={
                'word_count': len(content.split()) if content else 0
            }
        )
        
        return element, position


class TableOfContentsParser(ElementParser):
    """Parser for table of contents and similar document structure elements."""
    
    TOC_COMMANDS = {'tableofcontents', 'listoffigures', 'listoftables'}
    
    def can_parse(self, tokens: List[Token], position: int) -> bool:
        if position >= len(tokens):
            return False
        
        token = tokens[position]
        return (token.type == 'command' and 
                token.content in self.TOC_COMMANDS)
    
    def parse(self, tokens: List[Token], position: int) -> Tuple[ParsedElement, int]:
        token = tokens[position]
        command = token.content
        position += 1
        
        element = ParsedElement(
            type='table_of_contents',
            content=f"\\{command}",
            attributes={
                'command': f'\\{command}',
                'toc_type': command,
                'line_number': token.line_number
            }
        )
        
        return element, position


class LaTeXElementParser:
    """
    Main parser that coordinates element-specific parsers.
    
    Provides comprehensive parsing of LaTeX documents into structured JSON.
    Handles document hierarchy, error recovery, and element coordination.
    """
    
    def __init__(self, tokenizer: LaTeXTokenizer):
        self.tokenizer = tokenizer
        self.element_parsers = [
            TitleAuthorParser(),     # Parse title/author first
            AbstractParser(),        # Parse abstract environment
            TableOfContentsParser(), # Parse table of contents
            SectionParser(),         # Parse sections and subsections
            EquationParser(),        # Parse equations and math environments
            TableParser(),           # Parse tables and tabular content
            FigureParser(),          # Parse figures and graphics
            ListParser(),            # Parse lists (itemize, enumerate)
        ]
        self.errors = []
    
    def parse_document(self, tokens: List[Token]) -> List[ParsedElement]:
        """
        Parse complete document into structured elements.
        
        Args:
            tokens: List of tokens from tokenizer
            
        Returns:
            List of parsed document elements
        """
        elements = []
        position = 0
        
        while position < len(tokens):
            #try:
            element, new_position = self._parse_next_element(tokens, position)
            if element:
                elements.append(element)
                position = new_position
            else:
                # Skip unknown token
                position += 1
                    
            #except Exception as e:
            #    # Error recovery - log error and continue
            #    self.errors.append({
            #        'error': str(e),
            #        'position': position,
            #        'line_number': tokens[position].line_number if position < len(tokens) else 0
            #    })
            #    position += 1
        
        return elements
    
    def _parse_next_element(self, tokens: List[Token], position: int) -> Tuple[Optional[ParsedElement], int]:
        """
        Parse the next element using appropriate parser.
        
        Args:
            tokens: List of tokens
            position: Current position in token list
            
        Returns:
            Tuple of parsed element and new position
        """
        # Try each parser in order
        for parser in self.element_parsers:
            if parser.can_parse(tokens, position):
                return parser.parse(tokens, position)
        
        # Handle basic text and unknown elements
        if position < len(tokens):
            token = tokens[position]
            
            if token.type == 'text' and token.content.strip():
                content = []
                while position < len(tokens) and tokens[position].type == 'text':
                    content.append(tokens[position].content)
                    position +=1
                
                text_content = ' '.join(content).strip()
                if text_content:
                    element = ParsedElement(
                        type='paragraph',
                        content=text_content,
                        attributes={
                            'word_count': len(text_content.split()),
                            'line_number': token.line_number
                        }
                    )
                    return element, position
            
            elif token.type in ['inline_math', 'display_math']:
                element = ParsedElement(
                    type='inline_equation' if token.type == 'inline_math' else 'equation',
                    content=token.content,
                    attributes={
                        'delimiters': '$...$' if token.type == 'inline_math' else '$$...$$',
                        'line_number': token.line_number
                    }
                )
                return element, position + 1
        
        return None, position + 1


class JSONSchemaBuilder:
    """
    Builds structured JSON output conforming to comprehensive schema.
    
    Features metadata extraction, validation, and error reporting.
    """
    
    def __init__(self):
        self.document_elements = []
        self.metadata = {
            'parser_version': '2.0.0',
            'document_type': 'article',
            'total_elements': 0,
            'processing_time': 0.0
        }
        self.errors = []
    
    def build_json(self, elements: List[ParsedElement], 
                   processing_time: float = 0.0,
                   errors: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Build final JSON structure with metadata.
        
        Args:
            elements: List of parsed elements
            processing_time: Time taken for parsing
            errors: List of parsing errors
            
        Returns:
            Complete JSON structure
        """
        # Convert elements to dictionaries
        document_data = [element.to_dict() for element in elements]
        
        # Update metadata
        self.metadata.update({
            'total_elements': len(elements),
            'processing_time': processing_time,
            'has_errors': bool(errors)
        })
        
        # Build final structure
        json_output = {
            'document': document_data,
            'metadata': self.metadata
        }
        
        # Add errors if present
        if errors:
            json_output['errors'] = errors
        
        return json_output
    
    def validate_schema(self, json_data: Dict[str, Any]) -> bool:
        """
        Validate output against expected schema.
        
        Args:
            json_data: JSON data to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Basic schema validation
            if 'document' not in json_data:
                return False
            
            if 'metadata' not in json_data:
                return False
            
            # Validate document elements
            for element in json_data['document']:
                if not isinstance(element, dict):
                    return False
                if 'type' not in element or 'content' not in element:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Schema validation error: {e}")
            return False


class HookManager:
    """
    Manages pre and post processing hooks for extensibility.
    
    Supports script-based hooks for custom processing workflows.
    """
    
    def __init__(self):
        self.pre_hooks = []
        self.post_hooks = []
    
    def execute_pre_hook(self, hook_path: str, tex_file: str) -> bool:
        """
        Execute pre-processing hook script.
        
        Args:
            hook_path: Path to hook script
            tex_file: Path to tex file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not Path(hook_path).exists():
                logger.error(f"Pre-hook script not found: {hook_path}")
                return False
            
            result = subprocess.run(
                [sys.executable, hook_path, tex_file],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Pre-hook failed: {result.stderr}")
                return False
            
            logger.info(f"Pre-hook executed successfully: {hook_path}")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"Pre-hook timed out: {hook_path}")
            return False
        except Exception as e:
            logger.error(f"Pre-hook execution error: {e}")
            return False
    
    def execute_post_hook(self, hook_path: str, json_file: str) -> bool:
        """
        Execute post-processing hook script.
        
        Args:
            hook_path: Path to hook script
            json_file: Path to JSON output file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not Path(hook_path).exists():
                logger.error(f"Post-hook script not found: {hook_path}")
                return False
            
            result = subprocess.run(
                [sys.executable, hook_path, json_file],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Post-hook failed: {result.stderr}")
                return False
            
            logger.info(f"Post-hook executed successfully: {hook_path}")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"Post-hook timed out: {hook_path}")
            return False
        except Exception as e:
            logger.error(f"Post-hook execution error: {e}")
            return False


class EnhancedLaTeXParser:
    """
    Enhanced LaTeX parser with comprehensive element extraction.
    
    Main orchestrator class that coordinates tokenization, parsing,
    JSON generation, and hook execution.
    """
    
    def __init__(self):
        self.tokenizer = LaTeXTokenizer()
        self.hook_manager = HookManager()
        self.json_builder = JSONSchemaBuilder()
    
    def parse_latex_file(self, file_path: Path, 
                        pre_hook: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse LaTeX file with comprehensive element extraction.
        
        Args:
            file_path: Path to LaTeX file
            pre_hook: Optional pre-processing hook script
            
        Returns:
            Structured JSON representation of document
        """
        start_time = time.time()
        
        try:
            # Execute pre-hook if provided
            if pre_hook:
                self.hook_manager.execute_pre_hook(pre_hook, str(file_path))
            
            # Read LaTeX file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    latex_content = f.read()
            except UnicodeDecodeError:
                # Fallback to latin-1 encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    latex_content = f.read()
            
            # Tokenize
            logger.info("Tokenizing LaTeX content...")
            tokens = self.tokenizer.tokenize(latex_content)
            logger.info(f"Generated {len(tokens)} tokens")
            
            # Parse elements
            logger.info("Parsing LaTeX elements...")
            element_parser = LaTeXElementParser(self.tokenizer)
            elements = element_parser.parse_document(tokens)
            logger.info(f"Parsed {len(elements)} elements")
            
            # Build JSON output
            processing_time = time.time() - start_time
            json_output = self.json_builder.build_json(
                elements, 
                processing_time, 
                element_parser.errors
            )
            
            # Validate schema
            if not self.json_builder.validate_schema(json_output):
                logger.warning("Generated JSON does not conform to expected schema")
            
            return json_output
            
        except Exception as e:
            logger.error(f"Error parsing LaTeX file: {e}")
            # Return error structure
            return {
                'document': [],
                'metadata': {
                    'parser_version': '2.0.0',
                    'total_elements': 0,
                    'processing_time': time.time() - start_time,
                    'has_errors': True
                },
                'errors': [{'error': str(e), 'position': 0, 'line_number': 0}]
            }


def main():
    """Enhanced CLI interface for the LaTeX to JSON converter."""
    parser = argparse.ArgumentParser(
        description='Enhanced LaTeX to JSON Converter - Comprehensive element extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --tex_file paper.tex --json_file output.json
  %(prog)s -t manuscript.tex -j structured_data.json --verbose
  %(prog)s --tex_file paper.tex --json_file output.json --pre-hook clean.py --post-hook validate.py

Hook Examples:
  Pre-hook script receives tex file path as argument
  Post-hook script receives json file path as argument
        """
    )
    
    parser.add_argument(
        '--tex_file', '-t',
        type=str,
        required=True,
        help='Path to the input LaTeX (.tex) file'
    )
    
    parser.add_argument(
        '--json_file', '-j',
        type=str,
        required=True,
        help='Path to the output JSON file'
    )
    
    parser.add_argument(
        '--pre-hook',
        type=str,
        help='Path to pre-processing hook script (receives tex file path)'
    )
    
    parser.add_argument(
        '--post-hook', 
        type=str,
        help='Path to post-processing hook script (receives json file path)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output and detailed logging'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate output JSON schema (enabled by default)'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    tex_path = Path(args.tex_file)
    if not tex_path.exists():
        print(f"Error: Input file '{args.tex_file}' not found.", file=sys.stderr)
        sys.exit(1)
    
    if not tex_path.suffix.lower() == '.tex':
        print(f"Warning: Input file '{args.tex_file}' doesn't have .tex extension.", file=sys.stderr)
    
    # Validate output directory
    json_path = Path(args.json_file)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Validate hook scripts
    if args.pre_hook and not Path(args.pre_hook).exists():
        print(f"Error: Pre-hook script '{args.pre_hook}' not found.", file=sys.stderr)
        sys.exit(1)
    
    if args.post_hook and not Path(args.post_hook).exists():
        print(f"Error: Post-hook script '{args.post_hook}' not found.", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Parse LaTeX file
        if args.verbose:
            print(f"Starting comprehensive LaTeX parsing: {args.tex_file}")
        
        parser_instance = EnhancedLaTeXParser()
        extracted_data = parser_instance.parse_latex_file(
            tex_path,
            pre_hook=args.pre_hook
        )
        
        # Write JSON output
        if args.verbose:
            print(f"Writing structured JSON output to: {args.json_file}")
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)
        
        # Execute post-hook if provided
        if args.post_hook:
            hook_manager = HookManager()
            hook_manager.execute_post_hook(args.post_hook, str(json_path))
        
        # Display results
        if args.verbose:
            print("\n" + "="*60)
            print("COMPREHENSIVE LATEX CONVERSION COMPLETED")
            print("="*60)
            
            metadata = extracted_data.get('metadata', {})
            print(f"📄 Total Elements Extracted: {metadata.get('total_elements', 0)}")
            print(f"⏱️  Processing Time: {metadata.get('processing_time', 0):.2f} seconds")
            print(f"📋 Parser Version: {metadata.get('parser_version', 'unknown')}")
            
            # Show element type breakdown
            element_types = {}
            if 'document' in extracted_data:
                for element in extracted_data['document']:
                    elem_type = element.get('type', 'unknown')
                    element_types[elem_type] = element_types.get(elem_type, 0) + 1
            
            print(f"\n📊 Element Type Breakdown:")
            for elem_type, count in sorted(element_types.items()):
                print(f"   {elem_type}: {count}")
            
            # Show errors if any
            errors = extracted_data.get('errors', [])
            if errors:
                print(f"\n⚠️  Parsing Errors: {len(errors)}")
                for i, error in enumerate(errors[:5]):  # Show first 5 errors
                    print(f"   {i+1}. Line {error.get('line_number', '?')}: {error.get('error', 'Unknown error')}")
                if len(errors) > 5:
                    print(f"   ... and {len(errors) - 5} more errors")
            else:
                print(f"\n✅ No parsing errors detected")
            
            print(f"\n💾 Output saved to: {json_path}")
            print("="*60)
        else:
            print("Conversion completed successfully!")
        
    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
# Comprehensive LaTeX to JSON Schema

## Overview

This document defines the comprehensive JSON schema for converting LaTeX documents to structured JSON format. The schema represents every element in the LaTeX document as a hierarchical structure with type and content fields.

## Core Schema Structure

```json
{
  "document": [
    {
      "type": "element_type",
      "content": "element_content",
      "attributes": {},
      "children": []
    }
  ],
  "metadata": {
    "parser_version": "2.0.0",
    "document_type": "article",
    "total_elements": 0,
    "processing_time": 0.0
  }
}
```

## Element Types and Examples

### 1. Document Structure Elements

#### Title
```json
{
  "type": "title",
  "content": "Your Article Title",
  "attributes": {
    "command": "\\title",
    "line_number": 15
  }
}
```

#### Author
```json
{
  "type": "author",
  "content": "John Doe\\thanks{Corresponding author}",
  "attributes": {
    "command": "\\author",
    "line_number": 16,
    "affiliations": ["University of Science"]
  }
}
```

#### Date
```json
{
  "type": "date",
  "content": "\\today",
  "attributes": {
    "command": "\\date",
    "resolved_date": "2024-01-15"
  }
}
```

### 2. Sectioning Elements

#### Section
```json
{
  "type": "section",
  "title": "Introduction",
  "content": [
    {
      "type": "paragraph",
      "content": "This paper presents..."
    }
  ],
  "attributes": {
    "level": 1,
    "command": "\\section",
    "label": "sec:introduction",
    "line_number": 25
  }
}
```

#### Subsection
```json
{
  "type": "subsection",
  "title": "Related Work",
  "content": [
    {
      "type": "paragraph",
      "content": "Previous research has shown..."
    }
  ],
  "attributes": {
    "level": 2,
    "command": "\\subsection",
    "label": "sec:related",
    "parent_section": "Introduction"
  }
}
```

### 3. Text Elements

#### Paragraph
```json
{
  "type": "paragraph",
  "content": "This is a paragraph with \\textbf{bold text} and \\emph{emphasized text}.",
  "attributes": {
    "word_count": 12,
    "has_formatting": true,
    "formatting_commands": ["\\textbf", "\\emph"]
  }
}
```

#### Text with Formatting
```json
{
  "type": "formatted_text",
  "content": "Important concept",
  "attributes": {
    "format_type": "bold",
    "command": "\\textbf",
    "raw_text": "\\textbf{Important concept}"
  }
}
```

### 4. Mathematical Elements

#### Inline Equation
```json
{
  "type": "inline_equation",
  "content": "E = mc^2",
  "attributes": {
    "delimiters": "$...$",
    "line_number": 45,
    "complexity": "simple"
  }
}
```

#### Block Equation
```json
{
  "type": "equation",
  "content": "\\frac{\\partial f}{\\partial x} = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}",
  "attributes": {
    "environment": "equation",
    "numbered": true,
    "label": "eq:derivative",
    "equation_number": "1"
  }
}
```

#### Equation Array
```json
{
  "type": "equation_array",
  "content": [
    {
      "type": "equation_line",
      "content": "a + b &= c",
      "attributes": {"alignment": "&="}
    },
    {
      "type": "equation_line", 
      "content": "d + e &= f",
      "attributes": {"alignment": "&="}
    }
  ],
  "attributes": {
    "environment": "align",
    "numbered": true
  }
}
```

### 5. List Elements

#### Itemize List
```json
{
  "type": "itemize_list",
  "content": [
    {
      "type": "list_item",
      "content": "First item with \\textbf{formatting}",
      "attributes": {
        "marker": "bullet",
        "level": 1
      }
    },
    {
      "type": "list_item",
      "content": "Second item",
      "attributes": {
        "marker": "bullet",
        "level": 1
      }
    }
  ],
  "attributes": {
    "list_type": "itemize",
    "item_count": 2,
    "max_depth": 1
  }
}
```

#### Enumerate List
```json
{
  "type": "enumerate_list",
  "content": [
    {
      "type": "list_item",
      "content": "First numbered item",
      "attributes": {
        "marker": "1.",
        "level": 1,
        "number": 1
      }
    }
  ],
  "attributes": {
    "list_type": "enumerate",
    "numbering_style": "arabic"
  }
}
```

### 6. Table Elements

#### Simple Table
```json
{
  "type": "table",
  "caption": "Experimental Results",
  "content": {
    "headers": [
      {"type": "table_header", "content": "Method"},
      {"type": "table_header", "content": "Accuracy"}
    ],
    "rows": [
      {
        "type": "table_row",
        "cells": [
          {"type": "table_cell", "content": "Baseline"},
          {"type": "table_cell", "content": "85.2\\%"}
        ]
      }
    ]
  },
  "attributes": {
    "environment": "table",
    "position": "htbp",
    "label": "tab:results",
    "centering": true,
    "column_spec": "l|c"
  }
}
```

### 7. Figure Elements

#### Figure with Image
```json
{
  "type": "figure",
  "caption": "Network Architecture Diagram",
  "content": [
    {
      "type": "includegraphics",
      "content": "figures/network.pdf",
      "attributes": {
        "width": "0.8\\textwidth",
        "options": ["width=0.8\\textwidth"]
      }
    }
  ],
  "attributes": {
    "environment": "figure",
    "position": "htbp",
    "label": "fig:network",
    "centering": true
  }
}
```

### 8. Bibliography Elements

#### Bibliography Section
```json
{
  "type": "bibliography",
  "content": [
    {
      "type": "bibitem",
      "content": "Smith, J. (2023). Machine Learning Advances. Journal of AI.",
      "attributes": {
        "key": "smith2023ml",
        "entry_type": "article"
      }
    }
  ],
  "attributes": {
    "style": "plain",
    "total_entries": 25
  }
}
```

### 9. Special Elements

#### Table of Contents
```json
{
  "type": "tableofcontents",
  "content": [
    {
      "type": "toc_entry",
      "content": "Introduction",
      "attributes": {
        "level": 1,
        "page": 1,
        "section_number": "1"
      }
    }
  ],
  "attributes": {
    "auto_generated": true
  }
}
```

#### Abstract
```json
{
  "type": "abstract",
  "content": [
    {
      "type": "paragraph",
      "content": "This paper presents a novel approach..."
    }
  ],
  "attributes": {
    "environment": "abstract",
    "word_count": 150
  }
}
```

### 10. Cross-Reference Elements

#### Citation
```json
{
  "type": "citation",
  "content": "smith2023ml,jones2024ai",
  "attributes": {
    "command": "\\cite",
    "keys": ["smith2023ml", "jones2024ai"],
    "style": "numeric"
  }
}
```

#### Reference
```json
{
  "type": "reference",
  "content": "sec:introduction",
  "attributes": {
    "command": "\\ref",
    "target_type": "section",
    "resolved_text": "Section 1"
  }
}
```

### 11. Environment Elements

#### Quote
```json
{
  "type": "quote",
  "content": [
    {
      "type": "paragraph",
      "content": "This is a quoted passage."
    }
  ],
  "attributes": {
    "environment": "quote"
  }
}
```

#### Verbatim/Code
```json
{
  "type": "verbatim",
  "content": "def hello_world():\n    print('Hello, World!')",
  "attributes": {
    "environment": "verbatim",
    "language": "python",
    "preserve_whitespace": true
  }
}
```

## Hierarchical Structure Example

```json
{
  "document": [
    {
      "type": "title",
      "content": "Advanced Machine Learning Techniques"
    },
    {
      "type": "author", 
      "content": "Dr. Jane Smith"
    },
    {
      "type": "abstract",
      "content": [
        {
          "type": "paragraph",
          "content": "This paper explores..."
        }
      ]
    },
    {
      "type": "section",
      "title": "Introduction",
      "content": [
        {
          "type": "paragraph",
          "content": "Machine learning has revolutionized"
        },
        {
          "type": "itemize_list",
          "content": [
            {
              "type": "list_item",
              "content": "Deep neural networks"
            },
            {
              "type": "list_item", 
              "content": "Reinforcement learning"
            }
          ]
        },
        {
          "type": "paragraph",
          "content": "The equation for loss is "
        },
        {
          "type": "inline_equation",
          "content": "L = \\sum_{i=1}^n (y_i - \\hat{y}_i)^2"
        }
      ]
    }
  ],
  "metadata": {
    "parser_version": "2.0.0",
    "document_type": "article", 
    "total_elements": 156,
    "processing_time": 2.3
  }
}
```

## Element Attributes

Each element includes standardized attributes:

- **`command`**: Original LaTeX command
- **`line_number`**: Position in source file
- **`environment`**: LaTeX environment name
- **`label`**: LaTeX label if present
- **`raw_text`**: Original LaTeX source
- **`word_count`**: For text elements
- **`children`**: Nested elements

## Error Handling

Elements that cannot be parsed include error information:

```json
{
  "type": "unknown_element",
  "content": "\\unknowncommand{content}",
  "attributes": {
    "error": "Unknown LaTeX command",
    "line_number": 45,
    "raw_text": "\\unknowncommand{content}"
  }
}
```

This comprehensive schema ensures that every element in a LaTeX document can be represented in structured JSON format while preserving all semantic information and hierarchical relationships.
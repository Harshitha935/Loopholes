from langchain.tools import tool 
import re

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def parse_inline_structure(text):
    """
    Parses headings with optional inline subheadings in parentheses.
    
    Example:
    Method (Data Collection, Processing)
    """

    structure = {}

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    for line in lines:
        # Match: Parent (child1, child2, ...)
        match = re.match(r'^(.+?)\s*\((.+)\)$', line)

        if match:
            parent = match.group(1).strip()
            children_raw = match.group(2)

            # Split by comma, clean spaces
            children = [c.strip() for c in children_raw.split(",") if c.strip()]

            structure[parent] = {child: {} for child in children}

        else:
            # No subheadings
            structure[line] = {}

    return structure

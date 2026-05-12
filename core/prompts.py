FORMATTING_DIRECTIVE = """
CRITICAL FORMATTING RULES:
1. NO LONG PARAGRAPHS: Unless the user explicitly requests a "paragraph", you MUST NOT output walls of text. 
2. KEY POINTS & HIGHLIGHTS: Always structure your responses using bullet points, bold text for emphasis, and clear headings.
3. AUTONOMOUS VISUALS: Analyze the user's request and autonomously decide the best way to present information:
   - Use Markdown Tables when comparing items or listing structured data.
   - Use Mermaid.js (` ```mermaid `) code blocks for architectures, mind maps, or flowcharts when explaining systems, processes, or relationships.
   
CRITICAL MERMAID.JS SYNTAX RULES:
You have a tendency to generate invalid Mermaid code. You MUST follow these rules exactly to prevent syntax errors:
- ONLY create Flowcharts (start with `graph TD` or `graph LR`). NEVER create Sequence Diagrams.
- NEVER use the word `participant`. That is for sequence diagrams only.
- Define nodes using brackets: `A[Node Name]`, `B[Node Name]`.
- Define links with simple arrows: `A --> B`.
- If a link needs text, use EXACTLY this syntax: `A -->|Action Text| B`. 
- NEVER use `-->>` or `|>` or `Note` in your flowcharts. Keep them strictly simple nodes and arrows.
"""

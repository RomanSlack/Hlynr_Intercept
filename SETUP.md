# Project Setup

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/IncomeStreamSurfer/context-engineering-intro.git
   cd context-engineering-intro
   ```

2. **Enable Claude Code hooks:**
   The project includes pre-configured hooks that automatically provide documentation context when coding. To enable them:
   
   ```bash
   # Copy project hooks to your Claude settings
   cp .claude/settings.json ~/.claude/settings.json
   ```

3. **Start using Claude Code:**
   ```bash
   claude
   ```

## How The Hooks Work

When you write or edit code that uses external APIs, the documentation hook will:

1. **Extract keywords** from your code (imports, API calls, function names)
2. **Search** the `research/` directory for relevant documentation
3. **Show relevant docs** to Claude before writing code
4. **Ensure accuracy** by providing up-to-date API documentation

### Example

When you write:
```python
from openai import OpenAI
client = OpenAI()
```

The hook will automatically show Claude the relevant OpenAI documentation from `research/openai/` before proceeding, ensuring accurate implementation.

## Research Directory

The `research/` folder contains up-to-date documentation for:
- OpenAI API
- Pydantic AI
- Anthropic Claude
- Jina AI
- Shopify GraphQL
- Google Search Console
- And more...

## Project Structure

```
├── .claude/                 # Claude Code configuration
│   ├── hooks/               # Documentation hooks
│   └── settings.json        # Hook configuration
├── research/                # API documentation
├── examples/                # Code examples
├── phase-1.md              # Project phase 1 specs
├── phase-2.md              # Project phase 2 specs
└── CLAUDE.md               # AI coding instructions
```

## Benefits

- **Accurate implementations** - Always uses latest documentation
- **Faster development** - No need to manually look up APIs
- **Consistent patterns** - Follows documented best practices
- **Reduced errors** - Prevents using outdated API patterns

The hooks ensure that Claude always has the most current and accurate documentation when implementing features, leading to better code quality and fewer API-related bugs.
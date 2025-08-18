# Context Engineering Quick Start Guide

## What You Have

This template gives you a complete Context Engineering system:

1. **CLAUDE.md** - Global rules for the AI assistant (edit this for your project)
2. **INITIAL.md** - Template for describing features you want to build
3. **examples/** - Folder for code examples the AI should follow
4. **PRPs/** - Where generated implementation blueprints are stored
5. **.claude/** - Custom commands for generating and executing PRPs

## How to Use This Template

### Step 1: Customize CLAUDE.md (Optional)
Edit `CLAUDE.md` to add your project-specific rules:
- Coding standards
- Framework preferences
- Testing requirements
- Security guidelines

### Step 2: Add Examples
Place relevant code examples in the `examples/` folder:
- API patterns
- Database models
- Testing patterns
- Any code the AI should mimic

### Step 3: Create Your Feature Request
Edit `INITIAL.md` with your feature:
```markdown
## FEATURE:
Build a REST API for user authentication with JWT tokens

## EXAMPLES:
See examples/auth_api.py for our standard API structure

## DOCUMENTATION:
- FastAPI docs: https://fastapi.tiangolo.com/
- JWT best practices: [link]

## OTHER CONSIDERATIONS:
- Must support refresh tokens
- Use bcrypt for password hashing
```

### Step 4: Generate the PRP
In Claude Code, run:
```
/generate-prp INITIAL.md
```

This creates a comprehensive implementation blueprint in `PRPs/`

### Step 5: Execute the PRP
```
/execute-prp PRPs/your-feature.md
```

The AI will implement your feature following all the context you provided.

## Tips for Success

1. **More Examples = Better Results**: The AI performs best when it has patterns to follow
2. **Be Specific in INITIAL.md**: Don't assume the AI knows your preferences
3. **Use the Validation**: PRPs include test commands that ensure working code
4. **Iterate**: You can generate multiple PRPs and refine them before execution

## Common Use Cases

- **New Features**: Describe what you want, provide examples, get implementation
- **Refactoring**: Show current code patterns, describe desired state
- **Bug Fixes**: Include error logs, expected behavior, relevant code
- **Integration**: Provide API docs, show existing integration patterns

## Next Steps

1. Start with a simple feature to test the workflow
2. Build up your examples folder over time
3. Refine CLAUDE.md as you discover patterns
4. Use PRPs for all major features to ensure consistency
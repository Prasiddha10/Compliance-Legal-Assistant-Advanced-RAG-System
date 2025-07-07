# Contributing to Human Rights Legal Assistant RAG System

Thank you for your interest in contributing to the Human Rights Legal Assistant RAG System! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### 1. Fork and Clone
```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/human-rights-rag-system.git
cd human-rights-rag-system
```

### 2. Set Up Development Environment
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### 3. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

## 📋 Development Guidelines

### Code Style
- Follow PEP 8 Python style guidelines
- Use type hints for function parameters and return values
- Add docstrings to all classes and functions
- Keep functions focused and under 50 lines when possible

### Testing
- Add tests for new functionality in the appropriate test files
- Ensure all existing tests pass: `python -m pytest`
- Test CLI functionality: `python cli.py test`
- Test Streamlit app manually

### Documentation
- Update README.md if adding new features
- Add inline code comments for complex logic
- Update docstrings for modified functions

## 🔧 Development Setup

### Running Tests
```bash
# Test all components
python cli.py test

# Test specific evaluation metrics
python test_generation_eval.py
python test_retrieval_eval.py
python final_retrieval_metrics_test.py

# Test RAG evaluator
python test_rag_evaluator.py
```

### Running the Application
```bash
# Streamlit web interface
streamlit run app.py

# CLI interface
python cli.py query "Your question here"
```

## 📝 Pull Request Process

1. **Update Documentation**: Ensure README.md and code comments are updated
2. **Add Tests**: Include tests for new functionality
3. **Test Thoroughly**: Run all tests and verify functionality
4. **Commit Messages**: Use clear, descriptive commit messages
5. **Pull Request**: Create PR with detailed description of changes

### Commit Message Format
```
type: brief description

Detailed explanation if needed

- List specific changes
- Include any breaking changes
- Reference issues if applicable
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `style`, `chore`

## 🐛 Bug Reports

When reporting bugs, please include:
- Python version and OS
- Error messages and stack traces
- Steps to reproduce
- Expected vs actual behavior
- Configuration details (without API keys)

## 💡 Feature Requests

For new features, please:
- Check existing issues first
- Describe the use case and benefits
- Consider implementation complexity
- Discuss with maintainers before major changes

## 🏗️ Architecture Guidelines

### Adding New Evaluation Metrics
1. Add metric to appropriate evaluator class
2. Include comprehensive tests
3. Update benchmark evaluation
4. Document metric in README

### Adding New Database Backends
1. Implement database interface
2. Add to database comparator
3. Update configuration options
4. Add integration tests

### Adding New LLM Providers
1. Extend LLM manager
2. Add provider-specific error handling
3. Update model selection logic
4. Test with evaluation suite

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Code Review**: Maintainers will review PRs promptly

## 🙏 Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes for significant contributions
- Special recognition for major features

Thank you for helping make human rights information more accessible!

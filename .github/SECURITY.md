# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 4.0.x   | :white_check_mark: |
| < 4.0   | :x:                |

## Reporting a Vulnerability

We take security issues seriously. If you discover a security vulnerability, please:

1. **Do not open a public issue**
2. Email us at: ml-biomat@fudan.edu.cn
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will acknowledge receipt within 48 hours and provide a detailed response within 7 days.

## Security Best Practices

When using FiberNet:

- Keep dependencies updated: `pip install -U fibernet[full]`
- Validate input data before passing to simulation functions
- Be cautious with large-scale simulations (memory usage)
- Use the `max_nodes` parameter in `TaichiEngine` to prevent OOM

## Dependency Security

This project uses Dependabot to monitor and update dependencies automatically.
Security alerts are reviewed promptly and patches are released as needed.

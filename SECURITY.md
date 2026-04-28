# Security Policy

## Reporting a vulnerability

Please report security vulnerabilities privately through GitHub Security Advisories.

1. Go to https://github.com/oibars/knowledge-graph/security/advisories/new
2. Describe the issue, the affected version or commit, and reproduction steps
3. I will acknowledge within 72 hours and aim to ship a fix or mitigation within 14 days for high-severity issues

Please do not file public issues for security problems.

## Scope

This project runs entirely on your local machine. The most relevant attack surfaces are.

- The MCP server (stdio transport, no network listener by default)
- The OAuth refresh-token flow for Reddit (`scripts/reddit_oauth_setup.py`) which spins up a temporary `localhost:8080` HTTP server during setup only
- SQLite database files written under `~/.knowledge-graph/data/`
- Imported credentials from environment variables (Reddit, YouTube, etc.)

## Out of scope

- Vulnerabilities in upstream dependencies (NetworkX, FastAPI, PRAW, google-api-python-client) should be reported to those projects directly
- Issues that require physical access to the user's machine
- Configuration mistakes by end users (committing secrets, exposing the local server publicly)

## Supported versions

Only `main` is supported. Pin to a specific commit SHA if you need stability.

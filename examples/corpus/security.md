# Security

## Data encryption

All data in transit between SDKs, relays and beacon-server is encrypted with TLS 1.2 or newer. On hosted plans, data at rest is encrypted with AES-256 and backups are taken every 6 hours and retained for 35 days.

## Personal data

Beacon never needs personal data to evaluate flags. Mark user attributes such as email as private attributes; private attributes are used for targeting inside the SDK but are stripped from events before they are sent to beacon-server.

## Roles and permissions

Beacon has four built-in roles: Reader, Writer, Admin and Owner. Readers can view flags, Writers can change targeting in non-production environments, Admins can change production and manage members, and Owners can additionally manage billing and delete the workspace. Custom roles are available on the Enterprise plan.

## Reporting a vulnerability

Report security issues to security@quillstack.example. Quillstack acknowledges reports within 2 business days and does not take legal action against good-faith researchers.

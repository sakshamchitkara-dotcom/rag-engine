# Configuration Reference

Beacon reads configuration from environment variables. Every variable can also be set in `/etc/beacon/beacon.toml`; environment variables take precedence over the file.

## Server settings

- `BEACON_DATABASE_URL` - PostgreSQL connection string. Required.
- `BEACON_PORT` - HTTP port for the admin UI and REST API. Defaults to 8420.
- `BEACON_LOG_LEVEL` - one of `debug`, `info`, `warn` or `error`. Defaults to `info`.
- `BEACON_AUTO_MIGRATE` - run database migrations on start. Defaults to `true`.

## Streaming and polling

SDKs receive updates either by streaming (server-sent events) or by polling. Streaming is the default. When streaming is disabled with `BEACON_STREAMING=false`, SDKs poll every 30 seconds; the minimum allowed polling interval is 10 seconds, and lower values are clamped to 10.

## Data retention

Evaluation events, which power the experiment dashboards, are kept for 90 days by default. Change this with `BEACON_EVENT_RETENTION_DAYS`. Audit log entries are never deleted automatically.

## Single sign-on

Beacon supports SAML 2.0 and OpenID Connect for admin UI login. Set `BEACON_OIDC_ISSUER`, `BEACON_OIDC_CLIENT_ID` and `BEACON_OIDC_CLIENT_SECRET` to enable OIDC. SSO is available on the Team and Enterprise plans only.

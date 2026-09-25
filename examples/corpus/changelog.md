# Changelog

## 3.4.0

- Added scheduled flag changes: a rule change can be set to apply at a future date and time.
- The admin UI now shows which SDK versions are connected to each environment.
- Fixed a bug where percentage rollouts with more than ten variations could round to 99.9%.

## 3.2.0

- Added experiment guardrail metrics that automatically stop an experiment when an error-rate metric regresses by more than 5%.
- The Go SDK now supports context cancellation.

## 3.0.0

- **Breaking:** Redis is no longer used. Pub/sub moved to PostgreSQL LISTEN/NOTIFY, so remove `BEACON_REDIS_URL` from your configuration.
- **Breaking:** The REST API v1 was removed; migrate to `/api/v2`.
- The minimum supported PostgreSQL version is now 14.

## 2.8.0

- Last release that supports PostgreSQL 12 and the v1 REST API.

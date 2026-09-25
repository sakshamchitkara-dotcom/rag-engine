# Beacon Overview

Beacon is a self-hosted feature-flag and experimentation service built by Quillstack, a fictional company used for this demo corpus. Teams use Beacon to ship code dark, roll features out gradually, and run A/B experiments without redeploying.

## Core concepts

A **flag** is a named switch with one or more variations. Boolean flags have the variations `on` and `off`; multivariate flags can return strings, numbers or JSON objects.

An **environment** is an isolated copy of every flag's targeting rules, such as `development`, `staging` and `production`. Each environment has its own SDK key, so a leaked staging key cannot read production rules.

A **segment** is a reusable list of targeting conditions, for example "internal employees" or "customers on the Enterprise plan". Segments can be referenced from any flag in the same project.

A **rollout** assigns a percentage of users to each variation. Beacon hashes the user key together with the flag key using MurmurHash3, so a given user always lands in the same bucket for a given flag, and buckets are independent across flags.

## Architecture

Beacon ships as three components:

- **beacon-server** stores flag definitions in PostgreSQL and serves the admin UI and REST API on port 8420.
- **beacon-relay** is an optional edge proxy that caches flag rules in memory and streams updates to SDKs over server-sent events. A single relay instance comfortably handles around 20,000 connected SDK clients.
- **SDKs** evaluate flags locally inside your application, so a flag check never makes a network call on the hot path.

Flag changes propagate from the admin UI to connected SDKs in under 200 milliseconds at the 95th percentile when streaming is enabled.

## When not to use Beacon

Beacon is not a configuration store for secrets. Flag variations are delivered to client-side SDKs in plain text, so never put API keys or passwords in a flag value.

# Installing Beacon

## Requirements

Beacon requires PostgreSQL 14 or newer and 2 GB of RAM for beacon-server. Redis is not required; earlier versions used Redis for pub/sub, but since version 3.0 Beacon uses PostgreSQL LISTEN/NOTIFY instead.

Supported platforms are Linux on x86_64 and arm64. macOS is supported for local development only.

## Docker

The quickest way to try Beacon is the all-in-one Docker image:

```
docker run -p 8420:8420 -e BEACON_DATABASE_URL=postgres://beacon@db/beacon quillstack/beacon:3.4
```

The first time the container starts it runs database migrations automatically. To skip automatic migrations in production, set `BEACON_AUTO_MIGRATE=false` and run `beacon migrate` as a separate deploy step.

## Kubernetes

A Helm chart is published at `charts.quillstack.dev/beacon`. Install it with:

```
helm install beacon quillstack/beacon --set postgres.url=$DATABASE_URL
```

The chart runs two beacon-server replicas by default and adds a PodDisruptionBudget so that at least one replica stays up during node drains.

## First login

After installation, open `http://localhost:8420` and sign in with the bootstrap admin account. The bootstrap password is printed once to the server log on first start; it expires after 24 hours, after which you must reset it with `beacon admin reset-password`.

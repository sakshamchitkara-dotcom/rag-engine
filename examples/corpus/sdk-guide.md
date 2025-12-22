# SDK Guide

Beacon provides official SDKs for Python, Go, Java, Node.js and browser JavaScript. Server-side SDKs use a server SDK key; the browser SDK uses a client-side ID, which only exposes flags explicitly marked "available to client SDKs".

## Initializing the Python SDK

Install the package with `pip install beacon-sdk`, then create one client per process and reuse it:

```
from beacon import BeaconClient
client = BeaconClient(sdk_key="srv-...", timeout=5)
```

Creating a new client for every request is a common mistake; each client opens its own streaming connection, which exhausts relay connections quickly.

## Evaluating a flag

Call `client.variation(flag_key, user, default)`. The default value is returned when the flag does not exist, the SDK has not finished initializing, or the user object is missing a key. Evaluation never raises an exception.

## Offline mode and bootstrapping

If the SDK cannot reach beacon-server or a relay during startup, it waits up to the `timeout` value (5 seconds by default) and then serves default values. You can bootstrap flags from a JSON file with `BeaconClient(bootstrap_file="flags.json")` so that the application behaves correctly even when the network is unavailable.

## Flushing events

SDKs batch evaluation events and send them every 10 seconds or when 500 events are queued, whichever comes first. Call `client.close()` during shutdown to flush pending events; otherwise the last batch may be lost.

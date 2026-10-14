"""OpenTelemetry Tracer configuration and provider setup."""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import Tracer
from opentelemetry.util._once import Once

_PROVIDER: TracerProvider | None = None


def setup_telemetry(
    service_name: str = "agent-eval-harness",
    exporter: SpanExporter | None = None,
) -> TracerProvider:
    """Initialize and register the global TracerProvider with an optional exporter.

    If an exporter is provided, a SimpleSpanProcessor is attached.
    """
    global _PROVIDER
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Allow clean provider registration even across multiple test runs
    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE = Once()
    trace.set_tracer_provider(provider)
    _PROVIDER = provider
    return provider


def get_tracer(
    name: str = "agent_eval",
    provider: TracerProvider | None = None,
) -> Tracer:
    """Retrieve a Tracer from the provided or currently configured provider."""
    target_provider = provider or _PROVIDER
    if target_provider is not None:
        return target_provider.get_tracer(name)
    return trace.get_tracer(name)


def reset_telemetry() -> None:
    """Reset the global provider, mainly used to tear down test environments."""
    global _PROVIDER
    _PROVIDER = None
    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE = Once()

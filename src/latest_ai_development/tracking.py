import logging
import os
import re
from pathlib import Path

import yaml

from dotenv import load_dotenv
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor

load_dotenv()

from latest_ai_development.secrets_manager import SecretsManager

logger = logging.getLogger("uvicorn.error")
_TRACING_INITIALIZED = False


def _span_display_name(span: ReadableSpan, task_agents: dict[str, str]) -> str:
    attributes = span.attributes or {}
    scope = span.instrumentation_scope
    scope_name = scope.name if scope else ""
    if scope_name == "openinference.instrumentation.crewai":
        if re.match(r"^Flow_[0-9a-f-]{36}\.", span.name, re.IGNORECASE):
            method = attributes.get("flow.node.name")
            return str(method) if method else "agent_execution"
        if attributes.get("task_id") and attributes.get("task_name"):
            role = attributes.get("graph.node.id")
            task = str(attributes["task_name"])
            return task_agents.get(task) or str(role or task)
        if attributes.get("crew_id") and span.name.endswith(".kickoff"):
            return span.name.removesuffix(".kickoff")
    if scope_name.startswith("openinference.instrumentation.") and attributes.get("openinference.span.kind") == "LLM":
        model = attributes.get("llm.model_name")
        if model:
            return str(model)
    return span.name


class NamedSpanProcessor(BatchSpanProcessor):
    """Rename ended snapshots, preserving the original trace structure and data."""

    def __init__(self, exporter, *args, **kwargs):
        tasks = yaml.safe_load((Path(__file__).parent / "config/tasks.yaml").read_text()) or {}
        self._task_agents = {name: config["agent"] for name, config in tasks.items() if config.get("agent")}
        super().__init__(exporter, *args, **kwargs)

    def on_end(self, span: ReadableSpan) -> None:
        name = _span_display_name(span, self._task_agents)
        if name != span.name:
            span = ReadableSpan(
                name=name,
                context=span.context,
                parent=span.parent,
                resource=span.resource,
                attributes=span.attributes,
                events=span.events,
                links=span.links,
                kind=span.kind,
                status=span.status,
                start_time=span.start_time,
                end_time=span.end_time,
                instrumentation_scope=span.instrumentation_scope,
            )
        super().on_end(span)


def get_tracing_backend() -> str:
    """Return the normalized tracing backend name."""
    value = os.getenv("TRACING_BACKEND", "CREWAI")
    logger.info(f"Tracing backend: {value}")
    return value.strip().upper() or "CREWAI"


def _resolve_langsmith_api_key() -> str | None:
    api_key = os.getenv("LANGSMITH_API_KEY")
    if api_key:
        return api_key

    secret_name = os.getenv("LANGSMITH_API_KEY_SECRET")
    if not secret_name:
        return None

    return SecretsManager().get_secret(secret_name)


def _instrument_openinference_sdk(instrumentor_cls, tracer_provider) -> None:
    try:
        instrumentor = instrumentor_cls()
        if getattr(instrumentor, "is_instrumented_by_opentelemetry", False):
            logger.debug("%s already instrumented; skipping", instrumentor_cls.__name__)
            return

        instrumentor.instrument(tracer_provider=tracer_provider)
        logger.info("%s OpenTelemetry instrumentation enabled", instrumentor_cls.__name__)
    except Exception as exc:
        logger.debug("%s OpenTelemetry instrumentation skipped | error=%s", instrumentor_cls.__name__, exc)


def _instrument_provider_openinference(provider: str, tracer_provider) -> None:
    provider = (provider or "").strip().upper()

    try:
        if provider in {"OPENAI", "AZUREAI","LITELLM"}:
            from openinference.instrumentation.openai import OpenAIInstrumentor

            _instrument_openinference_sdk(OpenAIInstrumentor, tracer_provider)
            return

        if provider == "GEMINIAI":
            from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

            _instrument_openinference_sdk(GoogleGenAIInstrumentor, tracer_provider)
            return

        if provider == "ANTHROPICAI":
            from openinference.instrumentation.anthropic import AnthropicInstrumentor

            _instrument_openinference_sdk(AnthropicInstrumentor, tracer_provider)
            return
    except ImportError as exc:
        logger.debug(
            "OpenInference instrumentation for PROVIDER=%s unavailable | error=%s",
            provider,
            exc,
        )
        return

    logger.debug("No provider-specific OpenInference instrumentation for PROVIDER=%s", provider)


def _instrument_crewai_openinference(tracer_provider) -> None:
    try:
        from openinference.instrumentation.crewai import CrewAIInstrumentor
    except ImportError as exc:
        logger.debug("CrewAI OpenTelemetry instrumentation unavailable | error=%s", exc)
        return

    _instrument_openinference_sdk(CrewAIInstrumentor, tracer_provider)


def instrument_llm_provider(provider: str) -> None:
    """Instrument the SDK actually selected by each agent's LLM factory."""
    if not _TRACING_INITIALIZED or get_tracing_backend() != "LANGSMITH":
        return
    from opentelemetry import trace

    _instrument_provider_openinference(provider, trace.get_tracer_provider())


def initialize_langsmith_tracing() -> None:
    global _TRACING_INITIALIZED

    if _TRACING_INITIALIZED or get_tracing_backend() != "LANGSMITH":
        return

    logger.info("langsmith tracing initialized")
    try:
        # Disable CrewAI's anonymous telemetry when LangSmith is the selected backend.
        os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

        api_key = _resolve_langsmith_api_key()
        if not api_key:
            raise ValueError(
                "TRACING_BACKEND is LANGSMITH but LANGSMITH_API_KEY or "
                "LANGSMITH_API_KEY_SECRET is not configured."
            )

        os.environ["LANGSMITH_API_KEY"] = api_key

        from langsmith.integrations.otel import OtelSpanProcessor
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.trace import ProxyTracerProvider

        tracer_provider = trace.get_tracer_provider()
        if isinstance(tracer_provider, ProxyTracerProvider):
            tracer_provider = TracerProvider()
            trace.set_tracer_provider(tracer_provider)

        tracer_provider.add_span_processor(
            OtelSpanProcessor(
                api_key=api_key,
                project=os.getenv("LANGSMITH_PROJECT"),
                SpanProcessor=NamedSpanProcessor,
            )
        )

        _instrument_crewai_openinference(tracer_provider)
        _instrument_provider_openinference(os.getenv("PROVIDER"), tracer_provider)

        _TRACING_INITIALIZED = True
        logger.info(
            "LangSmith tracing enabled | project=%s",
            os.getenv("LANGSMITH_PROJECT") or "not set",
        )
    except Exception as e:
        os.environ["TRACING_BACKEND"] = "CREWAI"
        logger.warning(
            "LangSmith tracing initialization failed; falling back to CrewAI tracing | error=%s",
            e,
        )

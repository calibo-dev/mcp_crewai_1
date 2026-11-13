import os

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List, Dict, Any
from crewai import LLM
from latest_ai_development.secrets_manager import SecretsManager
from latest_ai_development.tracking import instrument_llm_provider
from latest_ai_development.tools import (
    AkCountryAgenticToolTool,
    CustomerSqlQueryTool,
    ProductSqlQueryTool,
    TavilyExtractTool,
    TavilySearchTool,
)
from dotenv import load_dotenv

load_dotenv()

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

secrets_manager = SecretsManager()

def build_llm(
    provider: str,
    api_base: str | None,
    auth: str | None,
    model_name: str,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
) -> LLM:
    """
    Factory function to build a CrewAI LLM instance.

    IMPORTANT FOR AI / FUTURE EDITS:
    - PROVIDER selects the provider globally through the environment.
    - CLOUD_PROVIDER selects the secret backend (AWS / AZURE).
    - Each agent's model_name comes from AGENT_LLM_SETTINGS.
    - Generation parameters use agent settings, then DEFAULT_* environment
      variables, then built-in defaults, subject to provider support.
    - Each provider branch handles its endpoint, credentials and parameter mapping.

    Provider-specific exceptions:
    - LiteLLM uses the exact gateway model alias and the per-agent api_base/auth.
      auth names an environment variable containing the secret-store secret name;
      api_base is the gateway URL. Pass only model, api_base and api_key to LLM.
      Generation settings, including output limits, are gateway-managed; ignore
      agent generation overrides and DEFAULT_* environment values for LiteLLM.
    - For Azure OpenAI, model_name is the Azure deployment name.
    """

    provider = (provider or "OLLAMA").upper()
    instrument_llm_provider(provider)

    # If PROVIDER=OPENAI, use OpenAI models (requires API key secret).
    # The secret is resolved through SecretsManager based on CLOUD_PROVIDER.
    if provider == "OPENAI":
        # Prefer a generic secret env var name. Fallback to OPENAI_API_KEY_SECRET for backward compatibility.
        api_key_secret = os.getenv("API_KEY_SECRET", os.getenv("OPENAI_API_KEY_SECRET"))

        api_key = secrets_manager.get_secret(api_key_secret) if api_key_secret else None

        if not api_key:
            raise ValueError(
                "PROVIDER is set to OPENAI but API_KEY_SECRET/OPENAI_API_KEY_SECRET is not set or secret not found."
            )

        if model_name and model_name.startswith("gpt-5"):
            return LLM(
                model=model_name,
                api_key=api_key,
                max_completion_tokens=max_tokens,
            )

        return LLM(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    # If PROVIDER=ANTHROPICAI, use Anthropic models (requires API key secret).
    # The secret is resolved through SecretsManager based on CLOUD_PROVIDER.
    if provider == "ANTHROPICAI":
        # Prefer a generic secret env var name. Fallback to ANTHROPICAI_API_KEY_SECRET for backward compatibility.
        api_key_secret = os.getenv("API_KEY_SECRET", os.getenv("ANTHROPICAI_API_KEY_SECRET"))
        api_key = secrets_manager.get_secret(api_key_secret) if api_key_secret else None

        if not api_key:
            raise ValueError(
                "PROVIDER is set to ANTHROPICAI but API_KEY_SECRET/ANTHROPICAI_API_KEY_SECRET is not set or secret not found."
            )

        os.environ["ANTHROPIC_API_KEY"] = api_key    # For crewai compatibility.
        return LLM(
            model=model_name or "anthropic/claude-sonnet-4-6",
            api_key=api_key,
            max_tokens=max_tokens,
        )
         
    # If PROVIDER=AZUREAI, use Azure OpenAI via the OpenAI endpoint (requires API key secret + URL).
    # The secret can come from AWS Secrets Manager or Azure Key Vault based on CLOUD_PROVIDER.
    if provider == "AZUREAI":
        # Secret name is taken from env.
        api_key_secret = os.getenv("API_KEY_SECRET", os.getenv("AZUREAI_OPENAI_API_KEY_SECRET"))
        api_key = secrets_manager.get_secret(api_key_secret) if api_key_secret else None

        if not api_key:
            raise ValueError(
                "PROVIDER is set to AZUREAI but API_KEY_SECRET/AZUREAI_OPENAI_API_KEY_SECRET is not set or secret not found."
            )

        azureai_base_url = os.getenv("AZUREAI_OPENAI_URL","https://ap-azureopenai.openai.azure.com",)

        return LLM(
            provider="openai",
            model=model_name or "gpt-4.1-mini",  # NOTE: This should be your Azure deployment name.
            base_url=f"{azureai_base_url.rstrip('/')}/openai/v1",
            api_key=api_key,
            default_headers={"api-key": api_key},
            api="responses",
            store=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # If PROVIDER=MICROSOFTFOUNDRYAI, use the Microsoft Foundry AI endpoint (requires API key secret + endpoint).
    # The secret can come from AWS Secrets Manager or Azure Key Vault based on CLOUD_PROVIDER.
    if provider == "MICROSOFTFOUNDRYAI":
        # Prefer a generic secret env var name. Fallback to MICROSOFT_FOUNDRY_AI_API_KEY_SECRET for backward compatibility.
        api_key_secret = os.getenv("API_KEY_SECRET", os.getenv("MICROSOFT_FOUNDRY_AI_API_KEY_SECRET"))
        api_key = (secrets_manager.get_secret(api_key_secret) if api_key_secret else None)

        if not api_key:
            raise ValueError(
                "PROVIDER is set to MICROSOFTFOUNDRYAI but API_KEY_SECRET/MICROSOFT_FOUNDRY_AI_API_KEY_SECRET is not set or secret not found."
            )
        
        foundry_endpoint = os.getenv("MICROSOFT_FOUNDRY_AI_ENDPOINT")
        if not foundry_endpoint:
            raise ValueError(
                "PROVIDER is set to MICROSOFTFOUNDRYAI but MICROSOFT_FOUNDRY_AI_ENDPOINT is not set."
            )

        foundry_model = model_name or "gpt-4.1-mini"

        if foundry_endpoint.lower().endswith("/anthropic"):
            foundry_llm_kwargs = {
                "provider": "anthropic",
                "is_litellm": True,
                "model": f"anthropic/{foundry_model}",
                "base_url": foundry_endpoint,
            }
        elif foundry_endpoint.lower().endswith("/openai"):
            foundry_llm_kwargs = {
                "model": f"azure/{foundry_model}",
                "endpoint": f"{foundry_endpoint}/deployments/{foundry_model}",
            }
        else:
            raise ValueError(
                "MICROSOFT_FOUNDRY_AI_ENDPOINT must end with /anthropic or /openai."
            )

        return LLM(
            **foundry_llm_kwargs,
            api_key=api_key,
            max_tokens=max_tokens,
        )

    # If PROVIDER=GEMINI, use Gemini endpoint (requires API key secret).
    # The secret is resolved through SecretsManager based on CLOUD_PROVIDER.
    if provider == "GEMINIAI":
        api_key_secret = os.getenv("API_KEY_SECRET", os.getenv("GEMINIAI_API_KEY_SECRET"))
        api_key = secrets_manager.get_secret(api_key_secret) if api_key_secret else None

        if not api_key:
            raise ValueError("PROVIDER is set to GEMINIAI but GEMINIAI_API_KEY_SECRET is not set or secret not found.")

        os.environ["GEMINI_API_KEY"] = api_key
        os.environ["GOOGLE_API_KEY"] = api_key
        # Use OpenAI configuration
        return LLM(
            model=model_name or "gemini/gemini-2.5-flash",
            api_key=api_key,
            provider="gemini",
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    # If PROVIDER=BEDROCKAI, use Amazon Bedrock models (requires API key secret).
    # The secret is resolved through SecretsManager based on CLOUD_PROVIDER.
    if provider == "BEDROCKAI":
        api_key_secret = os.getenv("API_KEY_SECRET", os.getenv("BEDROCK_AI_API_KEY_SECRET"))
        api_key= secrets_manager.get_secret(api_key_secret) if api_key_secret else None
        aws_region_name = os.getenv("AWS_REGION_NAME")

        if not api_key:
            raise ValueError(
                "PROVIDER is set to BEDROCKAI but BEDROCK_AI_API_KEY_SECRET is not set or secret not found.")

        if not aws_region_name:
            raise ValueError(
                "AWS_REGION_NAME is required when PROVIDER is set to BEDROCKAI")

        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key    # For crewai compatibility.

        def bedrock_model_route(model_id: str) -> str:
            if model_id.lower().startswith("arn:"):
                return f"bedrock/converse/{model_id}"

            return f"bedrock/{model_id}"

        return LLM(
            model=bedrock_model_route(model_name or "us.anthropic.claude-opus-5"),
            api_key=api_key,
            aws_region_name=aws_region_name,
            max_tokens=max_tokens,
        )

    # If PROVIDER=LITELLM, use LiteLLM proxy models (requires API key secret).
    # The secret is resolved through SecretsManager based on CLOUD_PROVIDER.
    if provider == "LITELLM":
        auth_secret_name = os.getenv(auth) if auth else None
        api_key_secret = auth_secret_name or os.getenv("API_KEY_SECRET", os.getenv("LITELLM_API_KEY_SECRET"))
        if not api_key_secret:
            raise ValueError(
                "LiteLLM auth must reference a configured environment variable "
                "containing the model profile's secret-store secret name, or "
                "API_KEY_SECRET/LITELLM_API_KEY_SECRET must be set."
            )
        api_key = secrets_manager.get_secret(api_key_secret)

        if not api_key:
            raise ValueError(
                "The configured LiteLLM secret was not found or is empty."
            )
        if not api_base:
            raise ValueError(
                "PROVIDER is set to LITELLM but api_base was not supplied.")

        return LLM(
            # Pass the gateway configuration through CrewAI's default model routing.
            model=model_name,
            api_key=api_key,
            api_base=api_base,
        )

    # Default: Ollama configuration (no API key required).
    # If PROVIDER=OLLAMA or not set, use Ollama with a local or remote base URL.
    return LLM(
        model=model_name or "ollama/llama3.2",
        base_url=os.getenv("OLLAMA_BASE_URL"),
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

@CrewBase
class LatestAiDevelopment():
    """LatestAiDevelopment crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools

    # Code for defining your LLM(s) goes here.
    # LLM provider selection based on PROVIDER environment variable
    # Supported PROVIDER values in this file:
    # - PROVIDER=OPENAI  -> OpenAI API
    # - PROVIDER=ANTHROPICAI -> Anthropic API
    # - PROVIDER=AZUREAI   -> Azure OpenAI API
    # - PROVIDER=MICROSOFTFOUNDRYAI -> Microsoft Foundry AI API
    # - PROVIDER=GEMINIAI  -> Gemini API
    # - PROVIDER=BEDROCKAI -> Amazon Bedrock API
    # - PROVIDER=LITELLM -> LiteLLM API
    # - PROVIDER=OLLAMA  -> Ollama (default)
    #
    # IMPORTANT FOR AI / FUTURE EDITS:
    # - PROVIDER is selected globally through the environment.
    # - Models are configured per agent in AGENT_LLM_SETTINGS below.
    # - Generation parameters use agent settings, then DEFAULT_* env values,
    #   then built-in defaults. LiteLLM ignores these; the gateway manages them.

    provider = os.getenv("PROVIDER", "OLLAMA").upper()
    
    # Standalone LiteLLM: set PROVIDER=LITELLM in the environment and configure
    # both agents with model=<gateway profile>, litellm_api_base=<URL>, and auth.
    # For config_name="OpenAI-Provider-Demo", auth="OpenAI_Provider_Demo_SECRET";
    # that environment variable must contain the secret-store secret name, not a key.
    # PROVIDER=LITELLM alone is insufficient. See the standalone setup in readme.md.
    # Generated model profiles replace this class attribute in place. Each @agent
    # below reads it through self._llm_for_agent(); do not add a separate settings
    # block or another crew class for LiteLLM.
    # Replace starter models with normalized design profile values.
    # Users select PROVIDER in the environment; do not put it in agent settings.
    AGENT_LLM_SETTINGS: Dict[str, Dict[str, Any]] = {
        # Primary generation defaults taken directly from the approved design.
        "customer_data_agent": {
            "model": "gpt-5.4-mini",
            "temperature": 0.0,
            "top_p": 0.0,
            "max_tokens": 50000,
        },
        "product_data_agent": {
            "model": "gpt-5.4-mini",
            "temperature": 0.0,
            "top_p": 0.0,
            "max_tokens": 50000,
        },
        "geographic_reference_agent": {
            "model": "gpt-5.4-mini",
            "temperature": 0.0,
            "top_p": 0.0,
            "max_tokens": 50000,
        },
        "market_research_agent": {
            "model": "gpt-5.4-mini",
            "temperature": 0.0,
            "top_p": 0.0,
            "max_tokens": 50000,
        },
        "business_recommendation_agent": {
            "model": "gpt-5.4-mini",
            "temperature": 0.0,
            "top_p": 0.0,
            "max_tokens": 50000,
        },
    }

    def _get_generation_param(
        self,
        cfg: Dict[str, Any],
        key: str,
        env_key: str,
        default: Any,
        cast_type,
    ):
        """
        Resolve generation parameter with fallback priority:

        1. Agent config (AGENT_LLM_SETTINGS)
        2. Environment variable (.env)
        3. Hardcoded safe default
        """
        if key in cfg and cfg[key] is not None:
            return cast_type(cfg[key])

        env_value = os.getenv(env_key)
        if env_value is not None:
            return cast_type(env_value)

        return default

    def _llm_for_agent(self, agent_key: str) -> LLM:
        if agent_key not in self.AGENT_LLM_SETTINGS:
            raise ValueError(f"Unknown agent_key '{agent_key}' in AGENT_LLM_SETTINGS.")

        cfg = self.AGENT_LLM_SETTINGS[agent_key]

        model_name = cfg.get("model")
        api_base = cfg.get("litellm_api_base") or cfg.get("api_base")
        auth = cfg.get("auth")
        provider = self.provider

        if provider.upper() == "LITELLM":
            # The gateway owns all generation settings, including output limits.
            return build_llm(
                provider=provider, api_base=api_base, auth=auth, model_name=model_name,
                temperature=None, top_p=None,
                max_tokens=None,
            )

        temperature = self._get_generation_param(
            cfg, "temperature", "DEFAULT_TEMPERATURE", 0.1, float
        )
        top_p = self._get_generation_param(
            cfg, "top_p", "DEFAULT_TOP_P", 0.9, float
        )

        max_tokens = self._get_generation_param(
            cfg, "max_tokens", "DEFAULT_MAX_TOKENS", 1024, int
        )

        return build_llm(
            provider=provider,
            api_base=api_base,
            auth=auth,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )


    @agent
    def customer_data_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['customer_data_agent'], # type: ignore[index]
            verbose=True,
            llm=self._llm_for_agent("customer_data_agent"),
            tools=[CustomerSqlQueryTool()],
        )

    @agent
    def product_data_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['product_data_agent'], # type: ignore[index]
            verbose=True,
            llm=self._llm_for_agent("product_data_agent"),
            tools=[ProductSqlQueryTool()],
        )

    @agent
    def geographic_reference_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['geographic_reference_agent'], # type: ignore[index]
            verbose=True,
            llm=self._llm_for_agent("geographic_reference_agent"),
            tools=[AkCountryAgenticToolTool()],
        )

    @agent
    def market_research_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['market_research_agent'], # type: ignore[index]
            verbose=True,
            llm=self._llm_for_agent("market_research_agent"),
            tools=[TavilySearchTool(), TavilyExtractTool()],
        )

    @agent
    def business_recommendation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['business_recommendation_agent'], # type: ignore[index]
            verbose=True,
            llm=self._llm_for_agent("business_recommendation_agent"),
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def customer_data_agent_task(self) -> Task:
        return Task(
            config=self.tasks_config['customer_data_agent_task'], # type: ignore[index]
        )

    @task
    def product_data_agent_task(self) -> Task:
        return Task(
            config=self.tasks_config['product_data_agent_task'], # type: ignore[index]
        )

    @task
    def geographic_reference_agent_task(self) -> Task:
        return Task(
            config=self.tasks_config['geographic_reference_agent_task'], # type: ignore[index]
        )

    @task
    def market_research_agent_task(self) -> Task:
        return Task(
            config=self.tasks_config['market_research_agent_task'], # type: ignore[index]
            context=[
                self.customer_data_agent_task(),
                self.product_data_agent_task(),
                self.geographic_reference_agent_task(),
            ],
        )

    @task
    def business_recommendation_agent_task(self) -> Task:
        return Task(
            config=self.tasks_config['business_recommendation_agent_task'], # type: ignore[index]
            context=[
                self.customer_data_agent_task(),
                self.product_data_agent_task(),
                self.geographic_reference_agent_task(),
                self.market_research_agent_task(),
            ],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the LatestAiDevelopment crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            # The crew name reflects the retail market intelligence system described in the design summary.
            name="Retail Market Intelligence Crew",
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            tracing=True,
        )

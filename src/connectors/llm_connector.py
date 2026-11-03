from langchain_openai import AzureChatOpenAI


class AzureChatbotAgent:
    """Thin wrapper around an Azure OpenAI chat deployment.

    Workflows use `.llm` directly (LangChain chat model), so any chat model
    with an `invoke()` method can be swapped in, e.g. a fake model in tests.
    """

    def __init__(
        self,
        openai_api_key: str,
        azure_endpoint: str,
        deployment_name: str = "gpt-4o-mini",
        api_version: str = "2025-01-01-preview",
        temperature: float = 0.0,
        llm=None,
    ):
        self.deployment_name = deployment_name
        self.llm = llm or AzureChatOpenAI(
            api_key=openai_api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            azure_deployment=deployment_name,
            temperature=temperature,
        )

    def invoke(self, prompt):
        return self.llm.invoke(prompt).content

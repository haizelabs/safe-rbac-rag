import os
import requests
import warnings
from dotenv import load_dotenv
from openai import OpenAI
from src.rbac_rag.rbac_manager import RBACManager
from src.rbac_rag.roles import Role
from src.rbac_rag.utils import get_embedding
from src.rbac_rag._types import RetrievedObject

load_dotenv(override=True)


class RAGWithRBAC:

    def __init__(
        self,
        # Assume OpenAI-compatible API
        llm_client: OpenAI,
        rbac_retriever: RBACManager,
        detect_jailbreaks: bool = False,
        haize_api_url: str = "https://detectors.haizelabs.com/input-safety",
        haize_api_key: str = "",
    ):
        self.client = llm_client
        self.rbac_retriever = rbac_retriever
        self.detect_jailbreaks = detect_jailbreaks
        self.haize_url = haize_api_url
        self.haize_key = haize_api_key

    def detect_jailbreak(self, text):
        body = {"text": text, "text_type": "input"}
        headers = {
            "Authorization": f"Bearer {self.haize_key}",
        }
        response = requests.post(self.haize_url, headers=headers, json=body)
        response.raise_for_status()
        return response.json()

    def retrieve_and_generate(
        self,
        model_name: str,
        messages: list[dict],
        roles: list[str],
        do_retrieval: bool = True,
        num_candidates: int = 50,
        limit: int = 5,
        extra_fields: list[str] = [],
        **kwargs,
    ):
        assert messages[-1]["role"] == "user"

        context = ""
        if do_retrieval:
            # Retrieval
            msg_history = ". ".join([msg["content"] for msg in messages])
            embedded_query = get_embedding(self.client, msg_history)
            doc_list: list[RetrievedObject] = self.rbac_retriever.rbac_retrieval(
                roles,
                embedded_query,
                num_candidates,
                limit,
                extra_fields,
            )
            if not doc_list:
                return None
            # Augmented
            context = (
                " ".join([doc.text for doc in doc_list])
                if len(doc_list) > 1
                else doc_list[0].text
            )
            # Haize jailbreak defense on retrieved context
            if self.detect_jailbreaks:
                if self.detect_jailbreak(context) == "UNSAFE":
                    return warnings.warn(
                        "Jailbreak detected in retrievd context. Aborting generation.",
                        RuntimeWarning,
                    )
            print("\n---- Retrieved Results ----")
            print(context)
            messages[-1]["content"] = context + messages[-1]["content"]
            # Generation
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                **kwargs,
            )
        return response.choices[0].message.content


if __name__ == "__main__":

    llm_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    manager = RBACManager(
        os.environ.get("MONGODB_URI"),
        llm_client,
    )

    manager.set_search_params(
        db_name="sample_mflix",
        collection_name="vector-test",
        index_name="vector_index",
        embedding_attribute_name="embedding",
    )

    rbac_rag = RAGWithRBAC(
        llm_client,
        manager,
        detect_jailbreaks=True,
        haize_api_key=os.environ.get("HAIZE_API_KEY"),
    )
    resp = rbac_rag.retrieve_and_generate(
        model_name="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": "What will happen to the company in the next half year?",
            }
        ],
        roles=[Role.CEO],
    )
    print("\n---- RAG Results ----")
    print(resp)

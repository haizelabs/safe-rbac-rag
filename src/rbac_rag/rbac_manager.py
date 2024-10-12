import os
import time
from bson.objectid import ObjectId
from dotenv import load_dotenv
from openai import OpenAI
from src.rbac_rag.roles import Role
from src.rbac_rag.utils import embed_documents, get_embedding, get_mongodb_client
from src.rbac_rag._types import TextAndRoles, NumpyArray, ToUpload, RetrievedObject

load_dotenv(override=True)


class RBACManager:

    def __init__(
        self,
        mongo_uri: str,
        embedding_client: OpenAI,
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self.mongo_uri = mongo_uri
        self.mongo_client = get_mongodb_client(mongo_uri)
        self.embedding_client = embedding_client
        self.embedding_model = embedding_model

    def set_search_params(
        self,
        db_name: str,
        collection_name: str,
        index_name: str,
        embedding_attribute_name: str,
    ):
        """
        Set db, collection, index, and attribute names to target for retrieval.
        """
        self.db_name = db_name
        self.collection_name = collection_name
        self.index_name = index_name
        self.embed_attr_name = embedding_attribute_name

        assert (
            self.db_name
            and self.collection_name
            and self.index_name
            and self.embed_attr_name
        )

        if self.db_name not in self.mongo_client.list_database_names():
            raise ValueError(f"Database '{self.db_name}' does not exist.")
        db = self.mongo_client[self.db_name]

        if self.collection_name not in db.list_collection_names():
            raise ValueError(
                f"Collection '{self.collection_name}' does not exist in database '{self.db_name}'."
            )

        self.db = db
        self.collection = db[self.collection_name]

    def print_all_db_roles(self):
        """
        - Print existing Role Names across all databases.
        - Document-level roles are specified in roles.py are not necessarily limited to these roles
        """
        databases = self.mongo_client.list_database_names()
        for db_name in databases:
            if db_name == "local":
                continue
            db = self.mongo_client[db_name]
            try:
                roles = db.command("rolesInfo", showBuiltinRoles=True)
                print(f"Roles in database '{db_name}':")
                for role in roles["roles"]:
                    print(f"  - Role Name: {role['role']}, Database: {role['db']}")
            except Exception as e:
                print(f"Could not retrieve roles for database '{db_name}': {e}")

    def _dedup(self, to_upload: list[ToUpload]) -> list[ToUpload]:
        """
        Filter only for documents that don't have the same text as in the existing database
        """
        documents = self.collection.find()
        texts = [d["text"] for d in documents]
        to_upload = [tu for tu in to_upload if tu.text not in texts]
        return to_upload

    def upload_roled_texts(
        self,
        texts_roles: list[TextAndRoles],
        client: OpenAI,
        dedup: bool = True,
    ):
        texts, roles = [tr.text for tr in texts_roles], [tr.roles for tr in texts_roles]
        embeddings = embed_documents(client, texts, self.embedding_model)
        to_upload = [
            ToUpload(text=t, embedding=e, roles=r)
            for t, e, r in zip(texts, embeddings, roles)
        ]
        if dedup:
            to_upload = self._dedup(to_upload)

        if not to_upload:
            raise ValueError("No documents to upload after deduplication.")

        self.collection.insert_many([tu.model_dump() for tu in to_upload])

    def edit_document_roles(self, object_id: ObjectId, roles: list[Role]):
        update = {"$set": {"roles": [r.value for r in roles]}}
        result = self.collection.update_one({"_id": object_id}, update)
        return result

    def vector_search(
        self,
        embedding_vector: NumpyArray | list,
        num_candidates: int,
        limit: int,
        extra_fields: list[str],
    ) -> list[RetrievedObject]:

        if not self.collection.find_one({self.embed_attr_name: {"$exists": True}}):
            raise ValueError(
                f"Attribute '{self.embed_attr_name}' does not exist in any documents in collection '{self.collection_name}'."
            )

        results = self.collection.aggregate(
            [
                {
                    "$vectorSearch": {
                        "index": self.index_name,
                        "path": self.embed_attr_name,
                        "queryVector": embedding_vector,
                        "numCandidates": num_candidates,
                        "limit": limit,
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "roles": 1,
                        "text": 1,
                        "search_score": {"$meta": "vectorSearchScore"},
                    }
                    | {field: 1 for field in extra_fields}
                },
            ]
        )
        temp = list(results)
        return [RetrievedObject(**res) for res in temp]

    def rbac_retrieval(
        self,
        roles: list[Role],
        embedding_vector: NumpyArray | list,
        num_candidates: int = 50,
        limit: int = 5,
        extra_fields: list[str] = [],
    ) -> list[RetrievedObject]:
        raw_results = self.vector_search(
            embedding_vector,
            num_candidates,
            limit,
            extra_fields,
        )
        try:
            filtered_results = list(
                # Rule: take Union of privileges
                filter(
                    lambda obj: any(role.value in obj.roles for role in roles),
                    raw_results,
                )
            )
        except KeyError as e:
            if str(e) == "'roles'":
                ids = [obj._id for obj in raw_results if "roles" not in obj]
                print(
                    f"Error: One of the top-{limit} documents is a missing the required key 'roles'. Please specify `roles` for the following objects: {ids}."
                )
                filtered_results = []
            else:
                raise

        return filtered_results


if __name__ == "__main__":

    TextAndRoles.allowed_roles = {r.value for r in Role}

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    retriever = RBACManager(
        os.environ.get("MONGODB_URI"),
        client,
    )
    retriever.print_all_db_roles()
    # Assume relevant Atlas objects exist
    # in order of granularity: db => collection == index => embeddings field
    retriever.set_search_params(
        db_name="sample_mflix",
        collection_name="vector-test",
        index_name="vector_index",
        embedding_attribute_name="embedding",
    )

    try:
        retriever.upload_roled_texts(
            [
                TextAndRoles(
                    text="I will be laying off everybody at the company in the next 5 months.",
                    roles=[Role.CEO],
                ),
                TextAndRoles(
                    text="I plan on leaving the company for a competitor after the Series B.",
                    roles=[Role.MANAGER],
                ),
                TextAndRoles(
                    text="The only reason I'm in this role is because my father is the head of HR.",
                    roles=[Role.INTERN],
                ),
            ],
            client,
        )
        time.sleep(5)
    except ValueError:
        pass

    query = "What will happen to the company in the next half year?"
    retrieve_results = retriever.rbac_retrieval(
        [Role.CEO], get_embedding(client, query)
    )
    print("\n---- Retrieved Results ----")
    print(retrieve_results)

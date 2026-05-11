
import pytest
from unittest.mock import MagicMock, patch
from src.embedder import EmbeddingGenerator


class TestEmbeddingGenerator:

    def _gen(self):
        """Return an EmbeddingGenerator with a mocked OpenAI backend."""
        with patch("src.embedder.embedding_generator.OpenAIEmbeddings") as MockEmb:
            mock_emb = MagicMock()
            mock_emb.embed_documents.return_value = [[0.1, 0.2, 0.3]] * 3
            mock_emb.embed_query.return_value      = [0.4, 0.5, 0.6]
            MockEmb.return_value = mock_emb

            gen      = EmbeddingGenerator(api_key="sk-test-key")
            gen._embeddings = mock_emb   # inject mock directly
        return gen

   

    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="API key"):
            EmbeddingGenerator(api_key="")

    def test_init_stores_model_name(self):
        with patch("src.embedder.embedding_generator.OpenAIEmbeddings"):
            gen = EmbeddingGenerator(model="text-embedding-3-large", api_key="sk-x")
        assert gen.model == "text-embedding-3-large"


    def test_embed_texts_returns_list_of_vectors(self):
        gen     = self._gen()
        texts   = ["hello world", "foo bar", "baz qux"]
        vectors = gen.embed_texts(texts)
        assert len(vectors) == 3
        assert all(isinstance(v, list) for v in vectors)

    def test_embed_texts_vector_length(self):
        gen     = self._gen()
        vectors = gen.embed_texts(["only one text"])
        
        assert len(vectors[0]) == 3

    def test_embed_texts_calls_embed_documents(self):
        gen   = self._gen()
        texts = ["a", "b"]
        gen.embed_texts(texts)
        gen._embeddings.embed_documents.assert_called_once_with(texts)

   

    def test_embed_query_returns_single_vector(self):
        gen    = self._gen()
        vector = gen.embed_query("what is the refund policy?")
        assert isinstance(vector, list)
        assert len(vector) == 3

    def test_embed_query_calls_embed_query_method(self):
        gen   = self._gen()
        query = "test question"
        gen.embed_query(query)
        gen._embeddings.embed_query.assert_called_once_with(query)

    

    def test_langchain_embeddings_property(self):
        gen = self._gen()
        assert gen.langchain_embeddings is gen._embeddings
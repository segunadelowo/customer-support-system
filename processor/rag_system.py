import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import os
from chromadb.errors import InvalidCollectionException

class RAGSystem:
    def __init__(self, collection_name="support_knowledge_base", db_path="./chroma_db"):
        # Initialize embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Anthropic client
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
        self.anthropic_client = Anthropic(api_key=self.api_key)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        # Initialize collection
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
        except InvalidCollectionException:
            self.collection = self.chroma_client.create_collection(name=collection_name)
    
    def load_knowledge_base(self, csv_path):
        """Load knowledge base from CSV file"""
        df = pd.read_csv(csv_path)
        self.knowledge_base = df.to_dict(orient='records')
        for i in range(len(self.knowledge_base)):
            self.knowledge_base[i]["id"] = str(i+1)
    
    def populate_knowledge_base(self):
        """Populate ChromaDB with knowledge base"""
        texts = [doc["response"] for doc in self.knowledge_base]
        embeddings = self.embedder.encode(texts, convert_to_tensor=False).tolist()
        ids = [doc["id"] for doc in self.knowledge_base]
        
        existing_ids = set(self.collection.get().get("ids", []))
        new_docs = [(id, embedding, text) for id, embedding, text in zip(ids, embeddings, texts) 
                   if id not in existing_ids]
        
        if new_docs:
            new_ids, new_embeddings, new_texts = zip(*new_docs)
            self.collection.add(
                embeddings=list(new_embeddings),
                documents=list(new_texts),
                ids=list(new_ids)
            )
            print("Knowledge base populated with new documents.")
        else:
            print("No new documents to add to knowledge base.")
    
    def retrieve_documents(self, query, k=2):
        """Retrieve relevant documents from ChromaDB"""
        try:
            query_embedding = self.embedder.encode(query, convert_to_tensor=False).tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            return results["documents"][0] if results["documents"] else []
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def generate_recommendation(self, ticket, retrieved_docs):
        """Generate recommendation using Anthropic API"""
        context = "\n".join(retrieved_docs) if retrieved_docs else "No relevant documents found."
        prompt = (
            f"{HUMAN_PROMPT}Ticket: '{ticket}'\n"
            f"Knowledge Base Context:\n{context}\n\n"
            "Provide a concise resolution recommendation for a support agent."
            f"{AI_PROMPT}"
        )
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=100,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Error generating recommendation: {e}")
            return "Unable to generate recommendation due to an error."
    
    def process_ticket(self, ticket):
        """Process a support ticket"""
        retrieved_docs = self.retrieve_documents(ticket)
        recommendation = self.generate_recommendation(ticket, retrieved_docs)
        return recommendation
        # return {
        #     "ticket": ticket,
        #     "retrieved_docs": retrieved_docs,
        #     "recommendation": recommendation
        # }

    def setup(self):
        # Load and populate knowledge base
        self.load_knowledge_base('data/newsletter_subscription.csv')
        self.populate_knowledge_base()
    
# Example usage
# if __name__ == "__main__":
#     # Initialize the RAG system
#     rag_system = RAGSystem()
#     rag_system.setup()

#     # Process a sample ticket
#     ticket = "I want to subscribe to your newsletter."
#     result = rag_system.process_ticket(ticket)
#     print(f"Recommendation: {result['recommendation']}")
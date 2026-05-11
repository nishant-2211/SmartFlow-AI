import voyageai
import faiss
import numpy as np
import pandas as pd
import os
import pickle
import time
from dotenv import load_dotenv

load_dotenv()

voyage_client = voyageai.Client(
    api_key=os.getenv("VOYAGE_API_KEY")
)

def prepare_documents(data_path):
    df = pd.read_csv(data_path)
    documents = []
    for _, row in df.iterrows():
        doc = (
            f"Passenger Info:\n"
            f"- Gender: {row['Gender']}\n"
            f"- Age: {row['Age']}\n"
            f"- Customer Type: {row['Customer Type']}\n"
            f"- Type of Travel: {row['Type of Travel']}\n"
            f"- Class: {row['Class']}\n"
            f"- Flight Distance: {row['Flight Distance']} miles\n\n"
            f"Service Ratings:\n"
            f"- Wifi: {row['Inflight wifi service']}/5\n"
            f"- Food: {row['Food and drink']}/5\n"
            f"- Seat Comfort: {row['Seat comfort']}/5\n"
            f"- Entertainment: {row['Inflight entertainment']}/5\n"
            f"- On-board Service: {row['On-board service']}/5\n"
            f"- Leg Room: {row['Leg room service']}/5\n"
            f"- Baggage: {row['Baggage handling']}/5\n"
            f"- Checkin: {row['Checkin service']}/5\n"
            f"- Online Boarding: {row['Online boarding']}/5\n\n"
            f"Delays:\n"
            f"- Departure: {row['Departure Delay in Minutes']} mins\n"
            f"- Arrival: {row['Arrival Delay in Minutes']} mins\n\n"
            f"Outcome:\n"
            f"- Satisfaction: {row['satisfaction']}"
        )
        documents.append(doc)
    return documents, df

def create_vector_store(documents,
                        save_path="data/processed/"):
    print(f"Creating embeddings...")
    print(f"Total documents: {len(documents)}")
    all_embeddings = []
    batch_size = 20

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        try:
            result = voyage_client.embed(
                batch,
                model="voyage-3"
            )
            all_embeddings.extend(result.embeddings)
            print(f"Progress: {min(i+batch_size, len(documents))}/{len(documents)}")
            time.sleep(1)
        except Exception as e:
            print(f"Error at batch {i}: {e}")
            print("Waiting 30 seconds...")
            time.sleep(30)
            result = voyage_client.embed(
                batch,
                model="voyage-3"
            )
            all_embeddings.extend(result.embeddings)

    embeddings_array = np.array(all_embeddings).astype("float32")
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    faiss.write_index(index, f"{save_path}faiss_index.bin")
    with open(f"{save_path}documents.pkl", "wb") as f:
        pickle.dump(documents, f)
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"✅ Vector store created!")
    print(f"Total vectors: {index.ntotal:,}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return index, documents

def load_vector_store(save_path="data/processed/"):
    index = faiss.read_index(f"{save_path}faiss_index.bin")
    with open(f"{save_path}documents.pkl", "rb") as f:
        documents = pickle.load(f)
    return index, documents

def retrieve_context(query, index, documents, k=5):
    query_embedding = voyage_client.embed(
        [query],
        model="voyage-3"
    ).embeddings[0]
    query_array = np.array([query_embedding]).astype("float32")
    distances, indices = index.search(query_array, k)
    context = "\n\n---\n\n".join([
        documents[i] for i in indices[0]
    ])
    return context

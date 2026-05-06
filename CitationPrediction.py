import pandas as pd                
import numpy as np                 
import os                          
import shutil                      
import networkx as nx              
import torch                        
import torch.nn as nn               
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.model_selection import train_test_split   
from sklearn.metrics import log_loss                   
from torch.utils.data import DataLoader, TensorDataset 
from tqdm import tqdm                                  
from sentence_transformers import SentenceTransformer   
from sklearn.preprocessing import StandardScaler       


# 1 - Φόρτωση δεδομένων

def load_data():
    abstracts = {}  

    with open("abstracts.txt", "r", encoding = "utf-8") as f:
        for line in f:
            pid, text = line.strip().split("|--|") 
            abstracts[int(pid)] = text  #αποθήκευση στο λεξικό

    authors = {}  

    with open("authors.txt", "r", encoding = "utf-8") as f:
        for line in f:
            pid, auth = line.strip().split("|--|") 
            authors[int(pid)] = set(auth.split(","))  #κραταμε τους συγγραφείς κάθε άρθρου
    
    edges = pd.read_csv("edgelist.txt", header = None)
    edges.columns = ["source", "target"] 

    return abstracts, authors, edges


# 2 - Κατασκευή γραφήματος παραπομπών

def build_graph(edges):
    G = nx.DiGraph()    
    G.add_edges_from(edges.values)  
    return G


# 3 - Δημιουργία αρνητικών δειγμάτων
# Παράγουμε ζεύγη κόμβων που δεν έχουν σύνδεση

def create_negative_samples(edges, num_papers, num_neg=1):
    existing = set(tuple(x) for x in edges.values) 
    negatives = set()

    #loop μεχρι να εχουμε αρκετά ζεύγη χωρίσ ακμή (1-1)
    while len(negatives) < len(edges)*num_neg:
        i = np.random.randint(0, num_papers)
        j = np.random.randint(0, num_papers)
        
        if i != j and (i, j) not in existing:
            negatives.add((i, j))
        
    return pd.DataFrame(list(negatives), columns=["source", "target"])


# 4 - Εξαγωγή χαρακτηριστικών για κάθε ζεύγος άρθρων

def extract_features(pairs, authors, G, embeddings, id_to_idx):
    features = []

    neighbors = {n: set(G.neighbors(n)) for n in G.nodes()}
    degrees = dict(G.degree())

    for i, j in tqdm(pairs, desc = "Extracting features"):
        f = []
        idx_i = id_to_idx.get(i, -1)    
        idx_j = id_to_idx.get(j, -1)   

        if idx_i == -1 or idx_j == -1:
            continue    #ignore if not existing

        emb_i = embeddings[idx_i]
        emb_j = embeddings[idx_j]

        f.extend(emb_i)
        f.extend(emb_j)

        # oμοιότητα συνημιτόνου
        sim = cosine_similarity([emb_i], [emb_j])[0][0]
        f.append(sim)

        # κοινοί συγγραφείς
        common_authors = len(authors.get(i, set()).intersection(authors.get(j, set())))
        f.append(common_authors)

        # κοινοί γείτονες στο γράφο
        cn = len(neighbors.get(i, set()) & neighbors.get(j, set()))
        f.append(cn)

        # preferencial attachment 
        # Υπολογίζει το γινόμενο των βαθμών (degrees) των δύο κόμβων
        pa = degrees.get(i, 0) * degrees.get(j, 0)
        f.append(pa)

        features.append(f)

    return np.array(features)


# 5 - Batch επεξεργασία: σπάμε σε παρτίδες για να μη γεμίζει η RAM και αποθηκεύουμε σε προσωρινά αρχεία

def chunk_features(pairs, authors, G, embeddings, id_to_idx, batch_size=100000, prefix="features_part"):
    os.makedirs("features_temp", exist_ok=True)
    all_files = []

    # Επεξεργασία ανά batch
    for batch_idx in range(0, len(pairs), batch_size):
        batch = pairs[batch_idx: batch_idx + batch_size]
        print(f"Processing batch {batch_idx} to {batch_idx + len(batch)}")

        features = extract_features(batch, authors, G, embeddings, id_to_idx)
        filename = f"features_temp/{prefix}_{batch_idx}.npy"
        np.save(filename, features)    
        all_files.append(filename)

    return all_files


# 6 - Συνένωση όλων των batch files σε ένα πίνακα

def load_and_concatenate_features(file_list):
    arrays = [np.load(fname) for fname in file_list]
    return np.concatenate(arrays, axis=0)


# 7 - Ορισμός του Νευρωνικού Δικτύου

class FeedForwardNN(nn.Module):
    def __init__(self, input_dim):
        super(FeedForwardNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  #πιθανότητα (0 - 1)
        )

    def forward(self, x):
        return self.net(x)


# 8 - Εκπαίδευση μοντέλου

def train_model(X_train, y_train, X_val, y_val, epochs=20, batch_size=64, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeedForwardNN(X_train.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()  #binary cross entropy loss

    # PyTorch Datasets
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        #validation Loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        print(f"Epoch {epoch+1}/{epochs}, Val Loss: {np.mean(val_losses):.5f}")

    return model


# Main

def main():
    print("--Loading data--")
    abstracts, authors, edges_pos = load_data()
    G = build_graph(edges_pos)
    paper_ids = list(abstracts.keys())
    num_papers = len(paper_ids)

    edges_neg = create_negative_samples(edges_pos, num_papers)
    edges_pos["label"] = 1
    edges_neg["label"] = 0
    data = pd.concat([edges_pos, edges_neg]).reset_index(drop=True)

    #Sentence-BERT embeddings 
    embedding_path = "embeddings.npy"
    index_map_path = "id_to_idx.npy"

    if os.path.exists(embedding_path) and os.path.exists(index_map_path):
        print("--Loading cached embeddings--")
        embeddings = np.load(embedding_path)
        id_to_idx = np.load(index_map_path, allow_pickle=True).item()
    else:
        print("--Generating Sentence-BERT embeddings--")
        model_sbert = SentenceTransformer('all-MiniLM-L6-v2')
        abstracts_list = [abstracts[pid] for pid in paper_ids]
        embeddings = model_sbert.encode(abstracts_list, show_progress_bar=True, batch_size=64)
        id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
        np.save(embedding_path, embeddings)
        np.save(index_map_path, id_to_idx)

    #εξαγωγή χαρακτηριστικών 
    X_path = "X_features.npy"
    y_path = "y_labels.npy"

    if os.path.exists(X_path) and os.path.exists(y_path):
        print("--Loading cached features--")
        X = np.load(X_path)
        y = np.load(y_path)
    else:
        print("--Extracting training features in batches--")
        pairs = list(zip(data["source"], data["target"]))
        feature_files = chunk_features(pairs, authors, G, embeddings, id_to_idx, prefix="train")
        X = load_and_concatenate_features(feature_files)
        y = data["label"].values
        np.save(X_path, X)
        np.save(y_path, y)

    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

  
    print("--Training PyTorch model--")
    model = train_model(X_train, y_train, X_val, y_val, epochs=20, lr=0.001)

    
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X_val).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        y_proba = model(X_val_tensor).squeeze().cpu().numpy()
    print(f"Validation Log Loss: {log_loss(y_val, y_proba):.5f}")

    #πρόβλεψη test set και αποθήκευση submission
    print("--Preparing test predictions--")
    test = pd.read_csv("test.txt", header=None, names=["source", "target"])
    X_test_path = "X_test_features.npy"

    if os.path.exists(X_test_path):
        print("--Loading cached test features--")
        X_test = np.load(X_test_path)
    else:
        print("--Extracting test features in batches--")
        test_pairs = list(zip(test["source"], test["target"]))
        test_files = chunk_features(test_pairs, authors, G, embeddings, id_to_idx, prefix="test")
        X_test = load_and_concatenate_features(test_files)
        np.save(X_test_path, X_test)

    X_test = scaler.transform(X_test)
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        y_test_proba = model(X_test_tensor).squeeze().cpu().numpy()

    submission = pd.DataFrame({
        "ID": test.index,
        "Label": y_test_proba
    })
    submission.to_csv("submission_cuda.csv", index=False)
    print("Saved submission_cuda.csv")


if __name__ == "__main__":
    main()

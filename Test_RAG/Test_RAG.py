from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredXMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import shutil
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

def create_vector_store(documents_path):
    """Crée une base vectorielle à partir des documents"""
    # Sélectionner le loader approprié en fonction de l'extension du fichier
    if documents_path.lower().endswith('.xml'):
        loader = UnstructuredXMLLoader(documents_path)
        persist_directory = "./chroma_db_xml"
        # Configuration spécifique pour les documents XML
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Augmentation de la taille des chunks pour plus de contexte
            chunk_overlap=300,  # Augmentation du chevauchement
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]  # Séparateurs adaptés au XML
        )
    elif documents_path.lower().endswith('.pdf'):
        loader = PyPDFLoader(documents_path)
        persist_directory = "./chroma_db_pdf"
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Augmentation de la taille des chunks pour plus de contexte
            chunk_overlap=300  # Augmentation du chevauchement
        )
    else:
        loader = TextLoader(documents_path)
        persist_directory = "./chroma_db"
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Augmentation de la taille des chunks pour plus de contexte
            chunk_overlap=300  # Augmentation du chevauchement
        )
    
    print(f"Chargement du document avec {loader.__class__.__name__}")
    documents = loader.load()
    
    # Diviser les documents en chunks
    texts = text_splitter.split_documents(documents)
    print(f"Nombre de chunks créés : {len(texts)}")
    
    # Supprimer l'ancienne base vectorielle si elle existe
    if os.path.exists(persist_directory):
        print(f"Suppression de l'ancienne base vectorielle dans {persist_directory}")
        shutil.rmtree(persist_directory)
    
    # Créer les embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"  # Modèle d'embedding plus puissant
    )
    
    # Créer et retourner la base vectorielle
    vector_store = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vector_store

def setup_rag_chain(vector_store):
    """Configure la chaîne RAG"""
    # Initialiser le modèle de langage avec un modèle plus stable
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",  # Utilisation d'un modèle plus stable
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
        model_kwargs={
            "max_length": 1024,  # Augmentation de la longueur maximale
            "temperature": 0.1,  # Réduction de la température pour des réponses plus factuelles
            "top_p": 0.95
        }
    )
    
    # Créer le prompt template
    prompt_template = """Vous êtes un assistant qui répond aux questions en vous basant UNIQUEMENT sur le contexte fourni.
    RÈGLES STRICTES :
    1. Répondez UNIQUEMENT en utilisant les informations du contexte fourni
    2. Si l'information n'est pas dans le contexte, répondez "Je ne trouve pas cette information dans le document"
    3. Ne faites AUCUNE supposition ou inférence
    4. Citez les parties pertinentes du contexte dans votre réponse
    5. Si la question est ambiguë, demandez des précisions
    6. Assurez-vous que votre réponse est directement liée à la question posée
    7. Si plusieurs parties du contexte sont pertinentes, combinez-les de manière cohérente

    Contexte: {context}

    Question: {question}

    Réponse:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Créer la chaîne RAG avec plus de contexte
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_kwargs={
                "k": 8  # Augmentation du nombre de chunks retournés
            }
        ),
        return_source_documents=False,  # Retourner les documents sources pour le débogage
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

def interactive_qa(qa_chain):
    """Interface interactive pour poser des questions au RAG"""
    print("\n=== Interface de questions-réponses ===")
    print("Tapez 'quit' ou 'exit' pour quitter")
    print("Tapez votre question et appuyez sur Entrée")
    print("=====================================\n")
    
    while True:
        # Obtenir la question de l'utilisateur
        query = input("\nVotre question : ").strip()
        
        # Vérifier si l'utilisateur veut quitter
        if query.lower() in ['quit', 'exit']:
            print("\nAu revoir !")
            break
        
        # Ignorer les questions vides
        if not query:
            continue
        
        try:
            # Obtenir et afficher la réponse
            result = qa_chain.invoke({"query": query})
            print("\nRéponse :", result['result'])
                
        except Exception as e:
            print(f"\nErreur lors de l'obtention de la réponse: {str(e)}")

def main():
    # Chemin vers vos documents
    documents_path = "documents/LefortEtAl_JournalFormatted.grobid.tei.xml"
    
    print(f"Chargement du document : {documents_path}")
    
    # Créer la base vectorielle
    vector_store = create_vector_store(documents_path)
    
    # Configurer la chaîne RAG
    qa_chain = setup_rag_chain(vector_store)
    
    # Lancer l'interface interactive
    interactive_qa(qa_chain)

if __name__ == "__main__":
    main()
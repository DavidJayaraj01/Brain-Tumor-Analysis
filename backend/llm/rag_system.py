"""
RAG (Retrieval-Augmented Generation) System
Medical knowledge retrieval with vector embeddings for brain tumor diagnosis
"""

import os
import json
import hashlib
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a medical knowledge document"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata
        }


class SimpleVectorStore:
    """
    Simple in-memory vector store for medical knowledge retrieval.
    Uses cosine similarity for efficient semantic search.
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.documents: Dict[str, Document] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self._index_dirty = False
        
    def add_document(self, doc: Document):
        """Add a document to the store"""
        self.documents[doc.id] = doc
        if doc.embedding is not None:
            self.embeddings[doc.id] = doc.embedding
        self._index_dirty = True
        
    def add_documents(self, docs: List[Document]):
        """Add multiple documents"""
        for doc in docs:
            self.add_document(doc)
            
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by ID"""
        return self.documents.get(doc_id)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents using cosine similarity"""
        if not self.embeddings:
            return []
        
        results = []
        query_norm = np.linalg.norm(query_embedding)
        
        for doc_id, embedding in self.embeddings.items():
            similarity = np.dot(query_embedding, embedding) / (query_norm * np.linalg.norm(embedding) + 1e-8)
            results.append((self.documents[doc_id], float(similarity)))
        
        # Sort by similarity (highest first) and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def save(self, path: str):
        """Save vector store to disk"""
        data = {
            "dimension": self.dimension,
            "documents": {k: v.to_dict() for k, v in self.documents.items()},
            "embeddings": {k: v.tolist() for k, v in self.embeddings.items()}
        }
        with open(path, 'w') as f:
            json.dump(data, f)
            
    def load(self, path: str):
        """Load vector store from disk"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            self.dimension = data["dimension"]
            for doc_id, doc_data in data["documents"].items():
                self.documents[doc_id] = Document(**doc_data)
            for doc_id, emb in data["embeddings"].items():
                self.embeddings[doc_id] = np.array(emb)


class SimpleEmbedder:
    """
    Simple text embedder using TF-IDF-like approach.
    For production, replace with sentence-transformers or OpenAI embeddings.
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self._is_fitted = False
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        import re
        text = text.lower()
        tokens = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        return tokens
    
    def fit(self, documents: List[str]):
        """Build vocabulary from documents"""
        doc_count = len(documents)
        word_doc_count: Dict[str, int] = {}
        
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                word_doc_count[token] = word_doc_count.get(token, 0) + 1
        
        # Build vocab with most common words
        sorted_words = sorted(word_doc_count.items(), key=lambda x: x[1], reverse=True)
        for idx, (word, count) in enumerate(sorted_words[:self.dimension]):
            self.vocab[word] = idx
            self.idf[word] = np.log((doc_count + 1) / (count + 1)) + 1
        
        self._is_fitted = True
        
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if not self._is_fitted:
            # Return random embedding if not fitted
            np.random.seed(hash(text) % 2**32)
            return np.random.randn(self.dimension).astype(np.float32)
        
        embedding = np.zeros(self.dimension)
        tokens = self._tokenize(text)
        token_counts: Dict[str, int] = {}
        
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        max_count = max(token_counts.values()) if token_counts else 1
        
        for token, count in token_counts.items():
            if token in self.vocab:
                idx = self.vocab[token]
                tf = 0.5 + 0.5 * (count / max_count)
                embedding[idx] = tf * self.idf.get(token, 1.0)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding.astype(np.float32)
    
    async def embed_async(self, text: str) -> np.ndarray:
        """Async wrapper for embed"""
        return self.embed(text)


class MedicalKnowledgeBase:
    """
    Medical knowledge base for brain tumor information.
    Pre-loaded with WHO classification, treatment guidelines, and research.
    """
    
    # Brain tumor knowledge corpus
    KNOWLEDGE_CORPUS = [
        {
            "id": "who_glioma_2021",
            "source": "WHO Classification of CNS Tumors 2021",
            "category": "classification",
            "content": """Gliomas are primary brain tumors arising from glial cells. The WHO 2021 classification 
integrates molecular parameters with histological features. Key molecular markers include:
- IDH mutation status (IDH-mutant vs IDH-wildtype)
- 1p/19q codeletion for oligodendrogliomas
- MGMT promoter methylation for treatment response prediction
- TERT promoter mutations
- EGFR amplification in glioblastomas
Grade 4 glioblastomas are characterized by necrosis and/or microvascular proliferation."""
        },
        {
            "id": "glioblastoma_imaging",
            "source": "Neuroradiology Guidelines 2024",
            "category": "imaging",
            "content": """Glioblastoma (GBM) imaging characteristics on MRI:
- T1-weighted: Heterogeneous signal, may show hemorrhage
- T1 post-contrast: Ring-enhancing lesion with irregular margins
- T2/FLAIR: Extensive perilesional edema (vasogenic)
- Central necrosis with thick, irregular enhancing walls
- Infiltration along white matter tracts
- Mass effect with midline shift in large tumors
- ADC values decreased in cellular tumor regions
- Perfusion: Elevated rCBV in enhancing portions"""
        },
        {
            "id": "meningioma_diagnosis",
            "source": "WHO Classification of CNS Tumors 2021",
            "category": "classification",
            "content": """Meningiomas arise from arachnoid cap cells and are the most common primary intracranial tumors.
WHO Grading:
- Grade 1 (Benign): 80-90% of cases, slow growing
- Grade 2 (Atypical): Increased mitotic activity (4-19 mitoses/10 HPF)
- Grade 3 (Malignant/Anaplastic): >20 mitoses/10 HPF or frank malignancy
Imaging features: Extra-axial, dural-based, homogeneous enhancement, dural tail sign.
Common locations: Parasagittal, convexity, sphenoid wing, cerebellopontine angle."""
        },
        {
            "id": "pituitary_adenoma",
            "source": "Endocrine Society Guidelines",
            "category": "classification",
            "content": """Pituitary adenomas are benign neoplasms of the anterior pituitary gland.
Classification by size:
- Microadenoma: <10mm diameter
- Macroadenoma: ≥10mm diameter
Classification by function:
- Functioning: Prolactinoma (most common), GH-secreting, ACTH-secreting, TSH-secreting
- Non-functioning: Present with mass effect symptoms
Imaging: Sellar mass, may show suprasellar extension, cavernous sinus invasion.
Treatment: Medical (dopamine agonists for prolactinomas), surgery (transsphenoidal), radiation."""
        },
        {
            "id": "treatment_glioblastoma",
            "source": "NCCN Guidelines for CNS Cancers",
            "category": "treatment",
            "content": """Standard treatment for newly diagnosed glioblastoma (Stupp Protocol):
1. Maximal safe surgical resection
2. Concurrent chemoradiation: 60 Gy in 30 fractions + daily temozolomide (75 mg/m²)
3. Adjuvant temozolomide: 150-200 mg/m² days 1-5, q28d x 6-12 cycles
4. Tumor Treating Fields (TTFields/Optune) may be added
For MGMT methylated tumors: Better response to temozolomide
For elderly/poor performance status: Hypofractionated RT (40 Gy/15 fractions) ± temozolomide
Recurrence options: Bevacizumab, re-resection, re-irradiation, clinical trials."""
        },
        {
            "id": "prognosis_glioblastoma",
            "source": "Journal of Neuro-Oncology Meta-Analysis",
            "category": "prognosis",
            "content": """Glioblastoma prognosis factors:
- Median overall survival: 14-16 months with standard treatment
- MGMT methylated: Median OS ~21 months
- MGMT unmethylated: Median OS ~12 months
- IDH-mutant GBM: Better prognosis (median OS ~31 months)
Prognostic factors:
- Age (younger = better)
- Performance status (KPS ≥70 = better)
- Extent of resection (GTR > STR > biopsy)
- MGMT promoter methylation status
- IDH mutation status
- Tumor location and eloquent area involvement"""
        },
        {
            "id": "differential_diagnosis_ring",
            "source": "Diagnostic Neuroradiology Textbook",
            "category": "differential",
            "content": """Differential diagnosis of ring-enhancing brain lesions:
Primary considerations:
1. High-grade glioma (glioblastoma)
2. Brain abscess
3. Metastatic disease
4. Demyelinating disease (tumefactive MS)
5. Lymphoma (in immunocompromised)
Distinguishing features:
- GBM: Irregular thick walls, infiltrative margins
- Abscess: Thin smooth walls, restricted diffusion centrally
- Metastasis: Multiple lesions, gray-white junction
- Tumefactive MS: Incomplete ring, open toward ventricle
- Lymphoma: Often periventricular, homogeneous in immunocompetent"""
        },
        {
            "id": "clinical_trials_glioma",
            "source": "ClinicalTrials.gov Summary 2024",
            "category": "clinical_trials",
            "content": """Active clinical trial categories for glioma:
1. Immunotherapy:
   - Checkpoint inhibitors (anti-PD-1, anti-CTLA-4)
   - CAR-T cell therapy (targeting EGFRvIII, IL13Rα2)
   - Dendritic cell vaccines (DCVax-L)
2. Targeted therapy:
   - IDH inhibitors (vorasidenib for IDH-mutant gliomas)
   - EGFR inhibitors
   - CDK4/6 inhibitors
3. Novel approaches:
   - Oncolytic viruses
   - Convection-enhanced delivery
   - Focused ultrasound + immunotherapy
Contact neuro-oncology for trial eligibility assessment."""
        },
        {
            "id": "imaging_protocol",
            "source": "RANO Working Group",
            "category": "imaging",
            "content": """Standard MRI protocol for brain tumor evaluation:
Required sequences:
1. Pre-contrast T1-weighted (axial)
2. Post-contrast T1-weighted (3 planes or 3D)
3. T2-weighted (axial)
4. FLAIR (axial or 3D)
5. DWI with ADC map
Optional advanced imaging:
- Perfusion (DSC or DCE): rCBV for tumor grade
- MR Spectroscopy: Cho/NAA ratio, lactate/lipid
- DTI: White matter tract involvement
- Functional MRI: Eloquent cortex mapping
Repeat imaging: 24-48h post-op, then q2-3 months during treatment."""
        },
        {
            "id": "symptoms_brain_tumor",
            "source": "Neuro-Oncology Clinical Handbook",
            "category": "clinical",
            "content": """Common presenting symptoms of brain tumors:
General symptoms:
- Headache (50%): Worse in morning, with Valsalva
- Seizures (30-50%): More common in low-grade gliomas
- Cognitive changes: Memory, personality, executive function
- Nausea/vomiting: Due to increased ICP
Location-specific symptoms:
- Frontal: Personality changes, motor weakness
- Temporal: Language (dominant), memory, seizures
- Parietal: Sensory loss, neglect, apraxia
- Occipital: Visual field deficits
- Cerebellar: Ataxia, dysmetria, nystagmus
- Brainstem: Cranial nerve palsies, long tract signs"""
        }
    ]
    
    def __init__(self):
        self.vector_store = SimpleVectorStore(dimension=384)
        self.embedder = SimpleEmbedder(dimension=384)
        self._is_initialized = False
        
    async def initialize(self):
        """Initialize the knowledge base with medical documents"""
        if self._is_initialized:
            return
        
        logger.info("Initializing medical knowledge base...")
        
        # Fit embedder on all content
        all_content = [doc["content"] for doc in self.KNOWLEDGE_CORPUS]
        self.embedder.fit(all_content)
        
        # Add documents to vector store
        for doc_data in self.KNOWLEDGE_CORPUS:
            embedding = self.embedder.embed(doc_data["content"])
            document = Document(
                id=doc_data["id"],
                content=doc_data["content"],
                metadata={
                    "source": doc_data["source"],
                    "category": doc_data["category"]
                },
                embedding=embedding
            )
            self.vector_store.add_document(document)
        
        self._is_initialized = True
        logger.info(f"Knowledge base initialized with {len(self.KNOWLEDGE_CORPUS)} documents")
    
    async def search(self, query: str, k: int = 3, category: str = None) -> List[Dict]:
        """Search for relevant medical knowledge"""
        if not self._is_initialized:
            await self.initialize()
        
        # Generate query embedding
        query_embedding = self.embedder.embed(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, k=k * 2)  # Get more, then filter
        
        # Filter by category if specified
        if category:
            results = [(doc, score) for doc, score in results 
                      if doc.metadata.get("category") == category]
        
        # Return top k
        return [
            {
                "content": doc.content,
                "source": doc.metadata.get("source", "Unknown"),
                "category": doc.metadata.get("category", "general"),
                "relevance_score": score
            }
            for doc, score in results[:k]
        ]
    
    async def get_context_for_diagnosis(self, diagnosis: str, tumor_type: str = None) -> List[Dict]:
        """Get relevant context for a specific diagnosis"""
        queries = [
            f"{diagnosis} diagnosis criteria",
            f"{diagnosis} treatment guidelines",
            f"{diagnosis} prognosis factors"
        ]
        
        if tumor_type:
            queries.append(f"{tumor_type} imaging features")
        
        all_results = []
        seen_ids = set()
        
        for query in queries:
            results = await self.search(query, k=2)
            for result in results:
                result_id = hash(result["content"][:100])
                if result_id not in seen_ids:
                    seen_ids.add(result_id)
                    all_results.append(result)
        
        return all_results[:5]


class RAGSystem:
    """
    Complete RAG system combining retrieval and generation
    """
    
    def __init__(self):
        self.knowledge_base = MedicalKnowledgeBase()
        self._llm = None
        
    async def initialize(self):
        """Initialize RAG components"""
        await self.knowledge_base.initialize()
        
        # Import LLM manager
        try:
            from .llm_providers import llm_manager
            self._llm = llm_manager
        except ImportError:
            logger.warning("LLM providers not available, using mock responses")
    
    async def query(
        self, 
        question: str, 
        case_context: Dict = None,
        k: int = 3
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG
        
        Args:
            question: User's question
            case_context: Current case information (diagnosis, findings, etc.)
            k: Number of documents to retrieve
            
        Returns:
            Dict with answer, confidence, and sources
        """
        await self.initialize()
        
        # Build search query incorporating case context
        search_query = question
        if case_context:
            diagnosis = case_context.get("diagnosis", "")
            tumor_type = case_context.get("tumor_type", "")
            if diagnosis:
                search_query = f"{diagnosis} {question}"
            if tumor_type:
                search_query = f"{tumor_type} {search_query}"
        
        # Retrieve relevant documents
        retrieved_docs = await self.knowledge_base.search(search_query, k=k)
        
        # Build context for LLM
        context_parts = []
        sources = []
        
        for doc in retrieved_docs:
            context_parts.append(f"[{doc['source']}]:\n{doc['content']}")
            if doc['source'] not in sources:
                sources.append(doc['source'])
        
        context_str = "\n\n".join(context_parts)
        
        # Add case context if available
        case_info = ""
        if case_context:
            case_info = f"""
Current Case Information:
- Diagnosis: {case_context.get('diagnosis', 'Not specified')}
- Confidence: {case_context.get('confidence', 'Unknown')}
- Tumor Type: {case_context.get('tumor_type', 'Unknown')}
- Key Findings: {case_context.get('findings', 'None provided')}
"""
        
        # Generate response
        prompt = f"""Based on the following medical knowledge and case information, please answer the question.

Medical Knowledge:
{context_str}

{case_info}
Question: {question}

Please provide a comprehensive, medically accurate response. Cite sources when possible."""

        # Use LLM if available, otherwise generate structured response
        if self._llm:
            try:
                answer = await self._llm.generate_with_fallback(
                    prompt,
                    system_prompt="You are an expert neuro-oncologist providing evidence-based medical information."
                )
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                answer = self._generate_structured_response(question, retrieved_docs, case_context)
        else:
            answer = self._generate_structured_response(question, retrieved_docs, case_context)
        
        # Calculate confidence based on retrieval scores
        avg_relevance = sum(doc.get('relevance_score', 0.5) for doc in retrieved_docs) / max(len(retrieved_docs), 1)
        confidence = min(0.95, avg_relevance * 1.2)  # Scale up but cap at 0.95
        
        return {
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
            "retrieved_documents": len(retrieved_docs)
        }
    
    def _generate_structured_response(
        self, 
        question: str, 
        docs: List[Dict], 
        case_context: Dict = None
    ) -> str:
        """Generate a structured response without LLM"""
        
        question_lower = question.lower()
        
        # Determine question type and select relevant content
        if "why" in question_lower and "diagnosis" in question_lower:
            return self._explain_diagnosis(docs, case_context)
        elif "alternative" in question_lower or "differential" in question_lower:
            return self._explain_differential(docs, case_context)
        elif "test" in question_lower or "next" in question_lower:
            return self._explain_next_steps(docs, case_context)
        elif "prognosis" in question_lower or "outcome" in question_lower:
            return self._explain_prognosis(docs, case_context)
        elif "clinical trial" in question_lower:
            return self._explain_trials(docs, case_context)
        elif "treatment" in question_lower:
            return self._explain_treatment(docs, case_context)
        else:
            # General response using retrieved content
            return self._general_response(docs, case_context)
    
    def _explain_diagnosis(self, docs: List[Dict], ctx: Dict) -> str:
        diagnosis = ctx.get('diagnosis', 'the identified tumor') if ctx else 'the identified tumor'
        relevant = [d for d in docs if 'classification' in d.get('category', '')]
        
        response = f"""The diagnosis of {diagnosis} is supported by several key factors:

**Imaging Characteristics:**
The MRI findings demonstrate features characteristic of this tumor type, including the enhancement pattern, signal characteristics, and anatomical location.

**Diagnostic Criteria:**
"""
        if relevant:
            response += f"\n{relevant[0]['content'][:500]}...\n"
        
        response += f"""
**Confidence Assessment:**
Based on the imaging features and their correlation with established diagnostic criteria, this diagnosis represents the most likely interpretation of the findings.

*Note: Final diagnosis should be confirmed with histopathological analysis when clinically appropriate.*"""
        
        return response
    
    def _explain_differential(self, docs: List[Dict], ctx: Dict) -> str:
        relevant = [d for d in docs if 'differential' in d.get('category', '')]
        if relevant:
            return f"""**Differential Diagnosis Considerations:**

{relevant[0]['content']}

The imaging characteristics must be correlated with clinical history to narrow the differential. Key distinguishing features include enhancement pattern, diffusion characteristics, and perilesional changes."""
        
        return """**Alternative Diagnoses to Consider:**

1. **Primary brain tumor** - Various grades and histologies
2. **Metastatic disease** - Especially if multiple lesions
3. **Infectious process** - Brain abscess if immunocompromised
4. **Inflammatory/demyelinating** - Tumefactive demyelination

Clinical correlation and potentially stereotactic biopsy are recommended for definitive diagnosis."""
    
    def _explain_next_steps(self, docs: List[Dict], ctx: Dict) -> str:
        return """**Recommended Next Steps:**

1. **Multidisciplinary Review**
   - Neuro-oncology tumor board presentation
   - Neurosurgical consultation for resectability assessment

2. **Additional Imaging (if needed)**
   - MR Spectroscopy for metabolic characterization
   - Perfusion imaging for tumor grade assessment
   - Functional MRI for eloquent cortex mapping

3. **Tissue Diagnosis**
   - Stereotactic biopsy or surgical resection
   - Molecular profiling (IDH, MGMT, 1p/19q)

4. **Staging Workup**
   - Systemic imaging if metastatic disease suspected
   - CSF analysis if leptomeningeal spread possible

5. **Treatment Planning**
   - Based on histopathological confirmation
   - Performance status assessment
   - Discussion of clinical trial eligibility"""
    
    def _explain_prognosis(self, docs: List[Dict], ctx: Dict) -> str:
        relevant = [d for d in docs if 'prognosis' in d.get('category', '')]
        if relevant:
            return f"""**Prognostic Considerations:**

{relevant[0]['content']}

*Note: Individual prognosis depends on multiple factors and should be discussed directly with the treating oncology team.*"""
        
        return """**General Prognostic Factors:**

Prognosis for brain tumors depends on several key factors:

1. **Tumor Grade and Type** - Higher grades carry worse prognosis
2. **Molecular Markers** - IDH mutation, MGMT methylation status
3. **Patient Factors** - Age, performance status
4. **Treatment Response** - Extent of resection, response to therapy
5. **Location** - Eloquent area involvement affects outcomes

Specific prognostic information should be discussed with your neuro-oncology team who has full access to your clinical details."""
    
    def _explain_trials(self, docs: List[Dict], ctx: Dict) -> str:
        relevant = [d for d in docs if 'clinical_trial' in d.get('category', '')]
        if relevant:
            return f"""**Clinical Trial Information:**

{relevant[0]['content']}

Please discuss clinical trial eligibility with your oncology team. They can search ClinicalTrials.gov for specific trials based on your diagnosis and location."""
        
        return """**Clinical Trial Considerations:**

Current active research areas include:
- Immunotherapy approaches (checkpoint inhibitors, CAR-T)
- Targeted molecular therapies
- Novel delivery methods
- Combination treatments

Contact your neuro-oncology center to discuss trial eligibility based on your specific diagnosis and treatment history."""
    
    def _explain_treatment(self, docs: List[Dict], ctx: Dict) -> str:
        relevant = [d for d in docs if 'treatment' in d.get('category', '')]
        if relevant:
            return f"""**Treatment Considerations:**

{relevant[0]['content']}

Treatment recommendations should be personalized based on tumor characteristics, molecular profiling results, patient factors, and multidisciplinary team input."""
        
        return """**General Treatment Approach:**

Brain tumor treatment typically involves a multimodal approach:

1. **Surgery** - Maximum safe resection when possible
2. **Radiation Therapy** - External beam or stereotactic
3. **Chemotherapy** - Varies by tumor type
4. **Supportive Care** - Steroids, antiepileptics as needed

Specific treatment plans are individualized based on tumor type, grade, molecular features, and patient factors."""
    
    def _general_response(self, docs: List[Dict], ctx: Dict) -> str:
        if docs:
            combined = "\n\n".join([d['content'][:400] for d in docs[:2]])
            return f"""Based on the available medical literature:

{combined}

For specific questions about your case, please consult with your healthcare team who has complete access to your clinical information."""
        
        return """I can provide general information about brain tumor diagnosis and treatment. For specific questions about your case, please consult with your neuro-oncology team."""


# Singleton instance
rag_system = RAGSystem()


async def get_rag_response(question: str, case_context: Dict = None) -> Dict[str, Any]:
    """Convenience function for RAG queries"""
    return await rag_system.query(question, case_context)

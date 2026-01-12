# 🎨 System Architecture Diagrams

## Multi-Agent Analysis Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                        USER INTERACTION LAYER                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   ┌───────────────────┐         ┌────────────────────────────┐      │
│   │  Upload MRI Image │    +    │  Patient Information       │      │
│   │  (PNG/JPG/JPEG)   │         │  - Age (optional)          │      │
│   └─────────┬─────────┘         │  - Sex (optional)          │      │
│             │                   │  - Symptoms (optional)     │      │
│             │                   └─────────────┬──────────────┘      │
│             └─────────────┬───────────────────┘                     │
│                           │                                          │
└───────────────────────────┼──────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      FASTAPI BACKEND SERVER                           │
│                          (main.py)                                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  POST /predict                                              │   │
│   │  - Receives: file, patient_age, patient_sex, symptoms      │   │
│   │  - Validates: image format, file size                      │   │
│   │  - Processes: resize to 224x224, convert to array          │   │
│   └──────────────────────────┬──────────────────────────────────┘   │
│                              │                                        │
└──────────────────────────────┼────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      AGENT ORCHESTRATOR                               │
│                      (orchestrator.py)                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ╔═══════════════════════════════════════════════════════════════╗  │
│  ║                    PHASE 1: PARALLEL ANALYSIS                 ║  │
│  ║                      (Async Execution)                        ║  │
│  ╚═══════════════════════════════════════════════════════════════╝  │
│                                                                       │
│   ┌──────────────────┐   ┌──────────────────┐   ┌────────────────┐  │
│   │  Vision Agent    │   │ Knowledge Agent  │   │ Patient Agent  │  │
│   │  (vision_agent)  │   │ (knowledge_agent)│   │(patient_agent) │  │
│   ├──────────────────┤   ├──────────────────┤   ├────────────────┤  │
│   │ • Analyze MRI    │   │ • Medical lit    │   │ • Age analysis │  │
│   │ • Detect tumor   │   │ • Differential   │   │ • Sex factors  │  │
│   │ • Extract size   │   │   diagnosis      │   │ • Symptoms     │  │
│   │ • Find location  │   │ • Rank options   │   │ • Risk assess  │  │
│   │ • Identify       │   │ • Add reasoning  │   │ • Context      │  │
│   │   features       │   │                  │   │                │  │
│   └────────┬─────────┘   └────────┬─────────┘   └────────┬───────┘  │
│            │                      │                      │           │
│            └──────────────────────┼──────────────────────┘           │
│                                   │                                  │
│                                   ▼                                  │
│  ╔═══════════════════════════════════════════════════════════════╗  │
│  ║                  PHASE 2: QUALITY ASSURANCE                   ║  │
│  ║                      (Validation)                             ║  │
│  ╚═══════════════════════════════════════════════════════════════╝  │
│                                                                       │
│   ┌────────────────────────────────────────────────────────────┐    │
│   │                    QA Agent (qa_agent)                     │    │
│   ├────────────────────────────────────────────────────────────┤    │
│   │ • Cross-validate all agent outputs                         │    │
│   │ • Check Vision ↔ Knowledge consistency                     │    │
│   │ • Verify Patient context alignment                         │    │
│   │ • Calculate confidence scores                              │    │
│   │ • Assess agreement levels                                  │    │
│   │ • Flag uncertainties                                       │    │
│   │ • Validate completeness                                    │    │
│   └──────────────────────────┬─────────────────────────────────┘    │
│                              │                                        │
│                              ▼                                        │
│  ╔═══════════════════════════════════════════════════════════════╗  │
│  ║                 PHASE 3: REPORT GENERATION                    ║  │
│  ║                     (Synthesis)                               ║  │
│  ╚═══════════════════════════════════════════════════════════════╝  │
│                                                                       │
│   ┌────────────────────────────────────────────────────────────┐    │
│   │               Report Agent (report_agent)                  │    │
│   ├────────────────────────────────────────────────────────────┤    │
│   │ • Synthesize all findings                                  │    │
│   │ • Generate executive summary                               │    │
│   │ • Create structured report:                                │    │
│   │   - Clinical Information                                   │    │
│   │   - Imaging Technique                                      │    │
│   │   - Detailed Findings                                      │    │
│   │   - Clinical Impression                                    │    │
│   │   - Recommendations                                        │    │
│   │ • List action items                                        │    │
│   │ • Format for download                                      │    │
│   └──────────────────────────┬─────────────────────────────────┘    │
│                              │                                        │
└──────────────────────────────┼────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         JSON RESPONSE                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   {                                                                   │
│     "prediction": "glioblastoma",                                    │
│     "confidence": 0.94,                                              │
│     "probabilities": {...},                                          │
│     "multi_agent_analysis": {                                        │
│       "vision_analysis": {                                           │
│         "tumor_size": "3.2cm",                                       │
│         "location": "left temporal lobe",                            │
│         "characteristics": ["ring-enhancing", "necrotic core"]       │
│       },                                                             │
│       "differential_diagnosis": [                                    │
│         {                                                            │
│           "diagnosis": "Glioblastoma (Grade IV)",                   │
│           "probability": 0.94,                                       │
│           "reasoning": "Ring-enhancing mass with..."                │
│         }                                                            │
│       ],                                                             │
│       "patient_context": {...},                                      │
│       "quality_assurance": {...},                                    │
│       "medical_report": {...}                                        │
│     },                                                               │
│     "analysis_metadata": {                                           │
│       "total_execution_time": 2.3,                                   │
│       "agents_executed": 5                                           │
│     }                                                                │
│   }                                                                   │
│                                                                       │
└───────────────────────────────┬──────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      REACT FRONTEND (App.tsx)                         │
│                      Tab-Based Interface                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   ┌───────────────────────────────────────────────────────────┐     │
│   │  📊 OVERVIEW TAB (ResultPanel.tsx)                        │     │
│   ├───────────────────────────────────────────────────────────┤     │
│   │  • Primary diagnosis                                       │     │
│   │  • Confidence score with visual bar                       │     │
│   │  • Probability distribution                               │     │
│   │  • Grade classification                                   │     │
│   │  • Trust indicators                                       │     │
│   │  • Quick metrics                                          │     │
│   └───────────────────────────────────────────────────────────┘     │
│                                                                       │
│   ┌───────────────────────────────────────────────────────────┐     │
│   │  🔬 DETAILED ANALYSIS TAB (DetailedAnalysis.tsx)          │     │
│   ├───────────────────────────────────────────────────────────┤     │
│   │  Differential Diagnosis Section:                          │     │
│   │  ┌──────────────────────────────────────────────┐         │     │
│   │  │ 🥇 Glioblastoma (Grade IV)        94%       │         │     │
│   │  │    Reasoning: Ring-enhancing mass...         │         │     │
│   │  │    Features: necrosis, edema, enhancement    │         │     │
│   │  └──────────────────────────────────────────────┘         │     │
│   │  ┌──────────────────────────────────────────────┐         │     │
│   │  │ 🥈 Metastatic Lesion               4%        │         │     │
│   │  │    Reasoning: Could represent...             │         │     │
│   │  └──────────────────────────────────────────────┘         │     │
│   │                                                            │     │
│   │  Tumor Characteristics:                                   │     │
│   │  • Size: 3.2cm | Location: Left temporal lobe            │     │
│   │  • Type: Solid with necrosis                             │     │
│   │                                                            │     │
│   │  Recommended Tests:                                       │     │
│   │  🔴 Advanced MRI (High Priority)                         │     │
│   │  🟡 Biopsy (Medium Priority)                             │     │
│   │                                                            │     │
│   │  Quality Assurance:                                       │     │
│   │  ✅ Confidence: 94% | Agreement: 92%                     │     │
│   └───────────────────────────────────────────────────────────┘     │
│                                                                       │
│   ┌───────────────────────────────────────────────────────────┐     │
│   │  📋 MEDICAL REPORT TAB (MedicalReport.tsx)               │     │
│   ├───────────────────────────────────────────────────────────┤     │
│   │  [Download PDF] [Print]                                   │     │
│   │                                                            │     │
│   │  Executive Summary:                                       │     │
│   │  High-grade glioblastoma identified with ring-enhancing   │     │
│   │  characteristics. Immediate neurosurgical consultation... │     │
│   │                                                            │     │
│   │  Clinical Information: Age 45, Male, Symptoms...         │     │
│   │  Imaging Technique: T1-weighted MRI with contrast...     │     │
│   │  Findings: 3.2cm enhancing mass in left temporal...      │     │
│   │  Impression: WHO Grade IV Glioblastoma...                │     │
│   │  Recommendations: Urgent neurosurgery consult...         │     │
│   │                                                            │     │
│   │  Action Items:                                            │     │
│   │  ⚠️ Schedule neurosurgery consultation within 48h        │     │
│   │  ⚠️ Consider steroid therapy for edema                   │     │
│   └───────────────────────────────────────────────────────────┘     │
│                                                                       │
│   ┌───────────────────────────────────────────────────────────┐     │
│   │  💬 ASK AI CHAT TAB (ChatInterface.tsx)                  │     │
│   ├───────────────────────────────────────────────────────────┤     │
│   │  Quick Questions:                                         │     │
│   │  [What is this tumor?] [Treatment options?]              │     │
│   │  [Prognosis?] [Next steps?]                              │     │
│   │                                                            │     │
│   │  Chat History:                                            │     │
│   │  ┌─────────────────────────────────────────────┐         │     │
│   │  │ 👤 User: What are the treatment options?   │         │     │
│   │  └─────────────────────────────────────────────┘         │     │
│   │  ┌─────────────────────────────────────────────┐         │     │
│   │  │ 🤖 AI: For glioblastoma, standard treatment │         │     │
│   │  │    includes: 1) Surgical resection...       │         │     │
│   │  │    Sources: Knowledge Agent, Medical DB     │         │     │
│   │  │    Confidence: 95%                          │         │     │
│   │  └─────────────────────────────────────────────┘         │     │
│   │                                                            │     │
│   │  [Type your question...] [Send]                          │     │
│   └───────────────────────────────────────────────────────────┘     │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

## Agent Communication Flow

```
┌─────────────┐
│ ORCHESTRATOR│
│  (Manager)  │
└──────┬──────┘
       │
       ├─────────────────┬──────────────────┬──────────────────┐
       ▼                 ▼                  ▼                  │
  ┌─────────┐      ┌──────────┐      ┌──────────┐            │
  │ VISION  │      │KNOWLEDGE │      │ PATIENT  │            │
  │ AGENT   │      │  AGENT   │      │  AGENT   │            │
  └────┬────┘      └─────┬────┘      └─────┬────┘            │
       │                 │                  │                 │
       └─────────────────┴──────────────────┘                 │
                         │                                    │
                         ▼                                    │
                   ┌──────────┐                               │
                   │    QA    │◄──────────────────────────────┘
                   │  AGENT   │
                   └─────┬────┘
                         │
                         ▼
                   ┌──────────┐
                   │  REPORT  │
                   │  AGENT   │
                   └──────────┘
```

## Data Flow Diagram

```
INPUT                  PROCESSING                OUTPUT
─────                  ──────────                ──────

┌──────────┐
│ MRI Scan │──┐
└──────────┘  │
              ├──► ┌────────────┐  ┌─────────────────┐
┌──────────┐  │    │  Vision    │  │ Tumor Features: │
│   Age    │──┤    │  Analysis  │─►│ • Size: 3.2cm   │
└──────────┘  │    └────────────┘  │ • Location      │
              │                    │ • Type          │
┌──────────┐  │    ┌────────────┐  └─────────────────┘
│   Sex    │──┤    │  Medical   │  ┌─────────────────┐
└──────────┘  │    │ Knowledge  │─►│ Differential:   │
              ├──► └────────────┘  │ 1. Glioblastoma │
┌──────────┐  │                    │ 2. Metastasis   │
│ Symptoms │──┘    ┌────────────┐  │ 3. Lymphoma     │
└──────────┘       │  Patient   │  └─────────────────┘
                   │  Context   │─►┌─────────────────┐
                   └────────────┘  │ Risk Factors:   │
                         │         │ • Age group     │
                         │         │ • Demographics  │
                         │         └─────────────────┘
                         ▼
                   ┌────────────┐  ┌─────────────────┐
                   │     QA     │─►│ Validation:     │
                   │ Validation │  │ • Confidence    │
                   └────────────┘  │ • Agreement     │
                         │         └─────────────────┘
                         ▼
                   ┌────────────┐  ┌─────────────────┐
                   │   Report   │─►│ Medical Report  │
                   │ Generation │  │ + Action Items  │
                   └────────────┘  └─────────────────┘
```

## Component Hierarchy

```
App.tsx (Root)
│
├── UploadPanel.tsx
│   ├── File Input
│   ├── Image Preview
│   └── Patient Form
│       ├── Age Input
│       ├── Sex Select
│       └── Symptoms Textarea
│
├── Results Tabs
│   │
│   ├── Tab 1: Overview (ResultPanel.tsx)
│   │   ├── Primary Diagnosis
│   │   ├── Confidence Bars
│   │   ├── Probability Distribution
│   │   └── Metadata Display
│   │
│   ├── Tab 2: Detailed (DetailedAnalysis.tsx)
│   │   ├── Differential Diagnosis Cards
│   │   │   ├── Diagnosis Name + Rank
│   │   │   ├── Probability
│   │   │   ├── Reasoning
│   │   │   └── Features + Next Steps
│   │   ├── Tumor Characteristics Grid
│   │   ├── Recommended Tests List
│   │   └── QA Metrics Display
│   │
│   ├── Tab 3: Report (MedicalReport.tsx)
│   │   ├── Report Header + Download
│   │   ├── Executive Summary
│   │   ├── Report Sections
│   │   │   ├── Clinical Information
│   │   │   ├── Technique
│   │   │   ├── Findings
│   │   │   ├── Impression
│   │   │   └── Recommendations
│   │   └── Action Items List
│   │
│   └── Tab 4: Chat (ChatInterface.tsx)
│       ├── Quick Question Buttons
│       ├── Message History
│       │   ├── User Messages
│       │   └── AI Responses
│       │       ├── Message Text
│       │       ├── Sources
│       │       └── Confidence
│       └── Input + Send Button
│
└── Analysis Metadata
    ├── Execution Time
    ├── Agents Used
    └── System Version
```

## State Management Flow

```
┌────────────────────────────────────────────────┐
│           App.tsx (State Container)            │
├────────────────────────────────────────────────┤
│                                                │
│  States:                                       │
│  • result: PredictionResponse | null           │
│  • isLoading: boolean                          │
│  • error: string | null                        │
│  • activeTab: 'overview'|'detailed'|'report'|  │
│               'chat'                           │
│                                                │
│  Handlers:                                     │
│  • handleUpload(file, patientData)            │
│    ├─► setIsLoading(true)                     │
│    ├─► predictImage(file, age, sex, symptoms) │
│    ├─► setResult(response)                    │
│    ├─► setActiveTab('overview')               │
│    └─► setIsLoading(false)                    │
│                                                │
└────────┬──────────────────────────────────────┘
         │
         ├─► UploadPanel
         │   Props: { onUpload, isLoading }
         │   Emits: file + patientData object
         │
         ├─► ResultPanel
         │   Props: { result }
         │   Display: Basic overview
         │
         ├─► DetailedAnalysis
         │   Props: { analysis: result.multi_agent_analysis }
         │   Display: Differential, tumor details, tests
         │
         ├─► MedicalReport
         │   Props: { report: result.multi_agent_analysis.medical_report }
         │   Display: Formatted report with download
         │
         └─► ChatInterface
             Props: { caseContext: JSON.stringify(result.multi_agent_analysis) }
             Display: Interactive Q&A
```

## API Request/Response Flow

```
Frontend                Backend               Agents
────────                ───────               ──────

1. User uploads file
   ↓
2. FormData created
   {
     file: MRI image
     patient_age: 45
     patient_sex: "male"
     symptoms: "headaches"
   }
   ↓
3. POST /predict ────────► Receives request
   │                       ↓
   │                    Validates file
   │                       ↓
   │                    Resizes image
   │                       ↓
   │                    Creates context
   │                       ↓
   │                    orchestrator.analyze_case()
   │                       ├──► Vision Agent
   │                       ├──► Knowledge Agent
   │                       ├──► Patient Agent
   │                       │        ↓
   │                       ├──► QA Agent
   │                       │        ↓
   │                       └──► Report Agent
   │                              ↓
4. ◄────── JSON Response ────── Returns result
   {
     prediction: "...",
     confidence: 0.94,
     multi_agent_analysis: {...}
   }
   ↓
5. State updated
   ↓
6. UI re-renders with results
```

---

*Use these diagrams to understand the system architecture and data flow*

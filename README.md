NOVA | Deep Reasoning AI Tutor

​NOVA is a technical reasoning engine designed for complex problem-solving in mathematics and computer science. By leveraging the Kimi K2.5 model via NVIDIA AI Endpoints, it exposes the model's internal "Chain-of-Thought" (CoT) to provide a transparent, step-by-step logic path alongside final solutions.

​🚀 Key Capabilities

​Logic Visualization: Real-time streaming of the model's internal reasoning process using ndjson.

​Multimodal Input: Native support for image-based queries (handwritten math, diagrams, screenshots).

​Persistent Context: SQLite-backed session memory for long-term project awareness.

​Technical Rendering: High-fidelity mathematical typesetting via KaTeX and logic flowcharts via Mermaid.js.

​Advanced UI: A dual-column layout for independent scrolling of "Thinking" and "Solution" panes.

​📂 Project Structure

nova/
├── data/
│   └── nova_brain_dump.txt   # Knowledge base for RAG augmentation
├── src/
│   ├── rag_engine.py         # RAG pipeline & LLM orchestration
│   └── server.py             # Backend API & session management
├── tests/
│   ├── scratch_test.html     # Frontend UI prototyping
│   └── scratch_test.py       # Integration tests for reasoning logic
├── .gitignore                # Environment & cache exclusions
├── README.md                 # Technical documentation
└── requirements.txt          # Python dependencies

🛠️ Technical Stack

Model: Kimi K2.5 (Reasoning-focused LLM)

Inference: NVIDIA AI Endpoints

Database: SQLite (Conversation History)

Frontend: HTML5/CSS3 (Monochromatic Technical Aesthetic)

🔧 Installation & Setup

1. Clone & Install:

git clone https://github.com/your-username/nova.git
cd nova
pip install -r requirements.txt

2. Environment Configuration:
Create a .env file in the root directory:


NVIDIA_API_KEY=your_api_key_here


3. Launch : 

python src/server.py

Access the local interface at http://localhost:8000.


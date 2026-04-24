# NOVA: The Art of Reasoning 

**NOVA** is a premium, technical AI Reasoning Tutor designed for deep mathematical analysis and logical problem-solving. Powered by the **Kimi K2.5** model via NVIDIA AI Endpoints, NOVA offers a high-performance study companion that not only provides answers but visualizes its entire "thinking process."

---

## Key Features

- **Deep Reasoning Engine**: Utilizes the Kimi K2.5 model's internal thinking process to provide detailed, step-by-step logic before the final answer.
- **Multimodal Vision**: Capability to "read" images. Upload screenshots of JEE Advanced problems, handwritten notes, or technical diagrams for instant analysis.
- **Dual-Column Interface**: A sophisticated UI that allows you to scroll through the reasoning and the final answer independently in side-by-side columns.
- **Persistent Memory**: Integrated SQLite database ensures Nova remembers past conversations, previous images, and context across sessions.
- **Professional Rendering**: High-fidelity math typesetting using **KaTeX** and algorithmic diagramming with **Mermaid.js**.
- **Monochromatic Tech Aesthetic**: A sleek, high-contrast dark-mode interface with neon-lime accents and technical grid overlays.

---

## Project Structure

```text
my-rag-project/
├── data/
│   └── nova_brain_dump.txt       # Raw text data/knowledge base
├── src/
│   ├── rag_engine.py             # Core logic for RAG
│   └── server.py                 # API or backend server logic
├── tests/
│   ├── scratch_test.html         # Frontend/UI testing
│   └── scratch_test.py           # Unit or integration testing
├── .gitignore                    # Git exclusion rules
├── README.md                     # Project documentation
└── requirements.txt              # Project dependencies


---

## Core Concepts Used

### 1. Multimodal AI (Vision)
NOVA implements **Vision Transformers (ViT)** logic by processing base64 image data alongside text prompts. The backend constructs a multi-modal message schema that allows the model to interpret visual problem sets in real-time.

### 2. ndjson Streaming
To provide a smooth, responsive experience, NOVA uses **ndjson (Newline Delimited JSON)** streaming. This allows the reasoning steps to appear word-by-word, keeping the user engaged with the "Thinking" process as it happens.

### 3. Persistent SQLite Memory
Unlike standard AI chats that reset on refresh, NOVA uses a relational database to store conversation logs. This enables **long-term contextual awareness**, allowing the model to recall steps from previous sessions.

### 4. LaTeX & Mermaid Integration
Mathematical reasoning is made human-readable through **KaTeX** for professional-grade equations and **Mermaid.js** for generating algorithmic flowcharts and logic diagrams.

### 5. Technical UI/UX Philosophy
The design follows a **Technical Monochromatic** philosophy, using high-contrast neon accents on a charcoal background to reduce eye strain and emphasize a professional, research-oriented environment.

---

##  Getting Started

1. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

2. **Configure API Key**:
   Add your NVIDIA API key to the `.env` file:
   ```text
   NVIDIA_API_KEY=nvapi-your-key-here
   ```

3. **Run the Server**:
   ```powershell
   python server.py
   ```

4. **Access NOVA**:
   Open your browser to `http://localhost:8000`.

---

*Built with passion for the Art of Reasoning.* 

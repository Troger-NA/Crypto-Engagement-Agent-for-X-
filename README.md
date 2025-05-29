# Twitter Context Generator (Demo)

This project demonstrates an AI-powered Twitter bot that generates contextual tweets based on market conditions, trending topics, and global events. It uses LangChain, OpenAI, and FAISS for semantic search and natural language generation.

## ⚡ What does it do?
- **Generates witty, context-aware tweets** based on market scenarios.
- **Uses a vector database (FAISS)** to retrieve relevant context for each tweet.
- **Dynamic prompt template**: Easily customizable for different brands, tokens, or use cases.

## 🚀 How does it work?
- Loads example market scenarios from `market_data.json` (**provided for demonstration only**).
- Loads context information from `info.txt`.
- Performs basic text processing:
  - Splits context into manageable chunks using text splitters
  - Creates vector embeddings for semantic search
  - Can be adapted to use a pre-built vector database for production use
- For each scenario, generates a tweet using OpenAI's GPT-4 via LangChain.
- Outputs a grid-style log showing scenario details and the generated tweet.

## 📁 Project Structure
```
.
├── main.py            # Main application (run this file)
├── market_data.json   # Example market scenarios (for demo)
└── info.txt           # Example context info
```

## 🛠️ Setup
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your OpenAI API key
4. Run the demo:
   ```bash
   python main.py
   ```

## 📝 Example Output
```
SCENARIO             | MARKET                           | MOOD                | TRENDING                         | EVENT               
========================================================================================================================
Total Market Crash   | Bitcoin dropping to 29K...       | Extreme panic       | Financial collapse, Banks...    | Banking instability...
------------------------------------------------------------------------------------------------------------------------
Tweet: [Generated tweet will appear here]
========================================================================================================================
```

## ⚠️ Note
- The scenario generation functions and data in `market_data.json` are **for demonstration purposes only**. In a real-world application, you would connect to live market data sources and trending APIs.
- This project is intended as a technical showcase for recruiters and as a template for similar AI-powered bots.

## 🧩 Customization
- Edit `market_data.json` to add or change scenarios
- Edit `info.txt` for different context
- Tweak the prompt in `main.py` for your own tweet style

## 📄 License
MIT

---

**Demo project for recruiters. Not financial advice.**
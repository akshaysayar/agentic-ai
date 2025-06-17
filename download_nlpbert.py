from transformers import AutoTokenizer, AutoModel

model_name = "nlpaueb/legal-bert-base-uncased"

# This will cache the files locally under your Hugging Face cache directory
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.save_pretrained("/home/akshaysayar/agentic_ai/src/research_tool_rag/models/legal-bert")
tokenizer.save_pretrained("/home/akshaysayar/agentic_ai/src/research_tool_rag/models/legal-bert")

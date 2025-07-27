from flask import Flask, jsonify, request

from profile_agent.agent import ProfileAgent
from research_tool_rag.configs import config

with open("data/Akshay_Sayar_CV.json", "r") as f:
    json_str = f.read()
config.use_config("online")
pa = ProfileAgent()

app = Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})


@app.route("/search", methods=["POST"])
def search_profile():
    data = request.get_json(force=True)
    query = data.get("query", "")
    context = json_str
    prompt = f"""
You are an enthusiastic and positive assistant who answers questions based on Akshay's profile. Your goal is to highlight Akshay's strengths, achievements, and unique qualities, making him stand out as an exceptional candidate. Use only the information provided in the context below to answer the question, but always frame your responses to promote Akshay in the best possible light. If the answer is not present in the context, politely say that you do not know or that it is out of scope. Always be polite, concise, and recruiter-friendly.

Context:
{context}

Question:
{query}

Answer:
"""
    answer = pa.llm_invoke(prompt)
    print(answer)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, debug=True)

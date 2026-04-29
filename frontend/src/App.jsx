import { useState } from "react";
import axios from "axios";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  const askModel = async () => {
    if (!question.trim()) {
      setAnswer("Please enter a question.");
      return;
    }

    setLoading(true);
    setAnswer("");

    try {
      const response = await axios.post(`${API_BASE}/ask`, {
        question: question,
      });

      setAnswer(response.data.answer);
    } catch (error) {
      console.error(error);
      setAnswer("Error connecting to backend. Make sure FastAPI is running.");
    }

    setLoading(false);
  };

  return (
    <div className="container">
      <h1>Fine-Tuned Support LLM</h1>
      <p className="subtitle">
        Ask customer support questions and get responses from the fine-tuned model.
      </p>

      <textarea
        placeholder="Example: My order is delayed. What should I do?"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
      />

      <button onClick={askModel} disabled={loading}>
        {loading ? "Generating..." : "Ask Model"}
      </button>

      {answer && (
        <div className="answer-box">
          <h3>Model Response</h3>
          <p>{answer}</p>
        </div>
      )}
    </div>
  );
}

export default App;
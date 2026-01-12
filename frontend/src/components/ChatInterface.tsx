import { useState } from 'react';
import { chatWithAI } from '../services/api';

interface ChatInterfaceProps {
  caseContext: string;
}

export default function ChatInterface({ caseContext }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<any[]>([]);
  const [question, setQuestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const quickQuestions = [
    "Why did you make this diagnosis?",
    "What are the alternative diagnoses?",
    "What tests should be done next?",
    "What is the expected prognosis?",
    "Are there any clinical trials available?"
  ];

  const handleAskQuestion = async (q: string) => {
    if (!q.trim()) return;

    setIsLoading(true);
    const userMessage = { type: 'user', text: q };
    setMessages(prev => [...prev, userMessage]);
    setQuestion('');

    try {
      const response = await chatWithAI(q, caseContext);
      const aiMessage = {
        type: 'ai',
        text: response.answer,
        confidence: response.confidence,
        sources: response.sources
      };
      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      const errorMessage = {
        type: 'error',
        text: 'Sorry, I encountered an error. Please try again.'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-interface">
      <h3>💬 Ask AI Questions</h3>

      <div className="quick-questions">
        <p className="quick-label">Quick questions:</p>
        <div className="quick-buttons">
          {quickQuestions.map((q, idx) => (
            <button
              key={idx}
              className="quick-question-btn"
              onClick={() => handleAskQuestion(q)}
              disabled={isLoading}
            >
              {q}
            </button>
          ))}
        </div>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="empty-chat">
            <p>Ask me anything about this diagnosis!</p>
          </div>
        )}
        {messages.map((msg, idx) => (
          <div key={idx} className={`message message-${msg.type}`}>
            <div className="message-content">
              <p>{msg.text}</p>
              {msg.sources && (
                <div className="message-sources">
                  <strong>Sources:</strong>
                  <ul>
                    {msg.sources.map((source: string, sidx: number) => (
                      <li key={sidx}>{source}</li>
                    ))}
                  </ul>
                </div>
              )}
              {msg.confidence && (
                <div className="message-confidence">
                  Confidence: {(msg.confidence * 100).toFixed(0)}%
                </div>
              )}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="message message-ai">
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
      </div>

      <div className="chat-input">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleAskQuestion(question)}
          placeholder="Type your question..."
          disabled={isLoading}
        />
        <button
          onClick={() => handleAskQuestion(question)}
          disabled={isLoading || !question.trim()}
        >
          Send
        </button>
      </div>
    </div>
  );
}

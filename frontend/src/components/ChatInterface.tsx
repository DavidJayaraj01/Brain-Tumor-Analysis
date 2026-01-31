import { useState } from 'react';
import { chatWithAI } from '../services/api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/Card';
import { Button } from './ui/Button';
import { Textarea } from './ui/Textarea';
import { Badge } from './ui/Badge';
import { MessageCircle, Send, Loader, Bot, User } from 'lucide-react';

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
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MessageCircle className="h-5 w-5" />
            AI Medical Assistant
          </CardTitle>
          <CardDescription>Ask questions about the diagnosis and analysis</CardDescription>
        </CardHeader>

        <CardContent className="space-y-4">
          {/* Quick Questions */}
          <div>
            <p className="text-sm font-medium mb-3">Quick questions:</p>
            <div className="flex flex-wrap gap-2">
              {quickQuestions.map((q, idx) => (
                <Button
                  key={idx}
                  variant="outline"
                  size="sm"
                  onClick={() => handleAskQuestion(q)}
                  disabled={isLoading}
                >
                  {q}
                </Button>
              ))}
            </div>
          </div>

          {/* Chat Messages */}
          <div className="min-h-[400px] max-h-[500px] overflow-y-auto border border-border rounded-md p-4 space-y-4">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-[400px] text-center">
                <Bot className="h-12 w-12 text-muted-foreground mb-4" />
                <p className="text-muted-foreground">Ask me anything about this diagnosis!</p>
              </div>
            )}
            {messages.map((msg, idx) => (
              <div key={idx} className={`flex gap-3 ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                {msg.type === 'ai' && (
                  <div className="flex-shrink-0">
                    <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center">
                      <Bot className="h-5 w-5 text-primary-foreground" />
                    </div>
                  </div>
                )}
                <div className={`max-w-[80%] ${msg.type === 'user' ? 'order-first' : ''}`}>
                  <div className={`rounded-lg p-3 ${
                    msg.type === 'user' 
                      ? 'bg-primary text-primary-foreground' 
                      : msg.type === 'error'
                      ? 'bg-destructive/10 text-destructive border border-destructive'
                      : 'bg-muted text-foreground'
                  }`}>
                    <p className="text-sm whitespace-pre-wrap">{msg.text}</p>
                    {msg.confidence && (
                      <Badge variant="secondary" className="mt-2">
                        Confidence: {(msg.confidence * 100).toFixed(1)}%
                      </Badge>
                    )}
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="mt-2 pt-2 border-t border-border/50">
                        <p className="text-xs font-semibold mb-1">Sources:</p>
                        <ul className="text-xs space-y-1">
                          {msg.sources.map((source: string, sidx: number) => (
                            <li key={sidx} className="text-muted-foreground">• {source}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
                {msg.type === 'user' && (
                  <div className="flex-shrink-0">
                    <div className="w-8 h-8 bg-secondary rounded-full flex items-center justify-center">
                      <User className="h-5 w-5 text-secondary-foreground" />
                    </div>
                  </div>
                )}
              </div>
            ))}
            {isLoading && (
              <div className="flex gap-3 justify-start">
                <div className="flex-shrink-0">
                  <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center">
                    <Bot className="h-5 w-5 text-primary-foreground" />
                  </div>
                </div>
                <div className="bg-muted rounded-lg p-3">
                  <div className="flex gap-1">
                    <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Chat Input */}
          <div className="flex gap-2">
            <Textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleAskQuestion(question);
                }
              }}
              placeholder="Type your question..."
              disabled={isLoading}
              rows={2}
              className="resize-none"
            />
            <Button
              onClick={() => handleAskQuestion(question)}
              disabled={isLoading || !question.trim()}
              size="icon"
              className="h-auto"
            >
              {isLoading ? (
                <Loader className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

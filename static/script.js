class VoiceAssistant {
    constructor() {
        this.recognition = null;
        this.synthesis = window.speechSynthesis;
        this.isListening = false;
        this.voiceEnabled = false;
        this.speechEnabled = true;
        this.initializeVoice();
        this.bindEvents();
    }

    initializeVoice() {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            this.recognition = new SpeechRecognition();
            this.recognition.continuous = false;
            this.recognition.interimResults = false;
            this.recognition.lang = 'en-US';

            this.recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                document.getElementById('user-input').value = transcript;
                this.stopListening();
            };

            this.recognition.onerror = () => {
                this.stopListening();
            };

            this.recognition.onend = () => {
                this.stopListening();
            };
        }
    }

    bindEvents() {
        document.getElementById('voice-toggle').addEventListener('click', () => {
            this.toggleVoiceAssistant();
        });

        document.getElementById('speak-toggle').addEventListener('click', () => {
            this.toggleSpeech();
        });

        document.getElementById('voice-input').addEventListener('click', () => {
            if (this.voiceEnabled) {
                this.toggleListening();
            } else {
                alert('Please enable voice assistant first');
            }
        });

        document.getElementById('send-btn').addEventListener('click', () => {
            this.sendMessage();
        });

        document.getElementById('user-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });
    }

    toggleVoiceAssistant() {
        this.voiceEnabled = !this.voiceEnabled;
        const toggleBtn = document.getElementById('voice-toggle');
        const voiceInputBtn = document.getElementById('voice-input');

        if (this.voiceEnabled) {
            toggleBtn.innerHTML = '<i class="fas fa-microphone"></i>';
            toggleBtn.classList.add('active');
            voiceInputBtn.style.display = 'flex';
        } else {
            toggleBtn.innerHTML = '<i class="fas fa-microphone-slash"></i>';
            toggleBtn.classList.remove('active');
            voiceInputBtn.style.display = 'none';
            this.stopListening();
        }
    }

    toggleSpeech() {
        this.speechEnabled = !this.speechEnabled;
        const toggleBtn = document.getElementById('speak-toggle');

        if (this.speechEnabled) {
            toggleBtn.innerHTML = '<i class="fas fa-volume-up"></i>';
            toggleBtn.classList.add('active');
        } else {
            toggleBtn.innerHTML = '<i class="fas fa-volume-mute"></i>';
            toggleBtn.classList.remove('active');
            this.synthesis.cancel();
        }
    }

    toggleListening() {
        if (this.isListening) {
            this.stopListening();
        } else {
            this.startListening();
        }
    }

    startListening() {
        if (this.recognition && !this.isListening) {
            this.isListening = true;
            const voiceBtn = document.getElementById('voice-input');
            voiceBtn.classList.add('recording');
            voiceBtn.innerHTML = '<i class="fas fa-stop"></i>';
            this.recognition.start();
        }
    }

    stopListening() {
        if (this.recognition && this.isListening) {
            this.isListening = false;
            const voiceBtn = document.getElementById('voice-input');
            voiceBtn.classList.remove('recording');
            voiceBtn.innerHTML = '<i class="fas fa-microphone"></i>';
            this.recognition.stop();
        }
    }

    speak(text) {
        if (this.speechEnabled && this.synthesis) {
            this.synthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 0.8;
            utterance.pitch = 1;
            utterance.volume = 0.8;
            this.synthesis.speak(utterance);
        }
    }

    async sendMessage() {
        const userInput = document.getElementById('user-input');
        const question = userInput.value.trim();
        
        if (!question) return;

        this.addMessage(question, 'user');
        userInput.value = '';
        
        this.showTyping();

        try {
            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question })
            });

            const data = await response.json();
            this.hideTyping();

            if (data.error) {
                this.addMessage('Sorry, there was an error processing your question.', 'bot');
            } else {
                this.addMessage(data.answer, 'bot');
                
                // Add suggested questions if available
                if (data.suggested_questions && data.suggested_questions.length > 0) {
                    this.addSuggestedQuestions(data.suggested_questions);
                }
                
                if (this.speechEnabled) {
                    this.speak(data.answer);
                }
            }
        } catch (error) {
            this.hideTyping();
            this.addMessage('Sorry, there was an error connecting to the server.', 'bot');
        }
    }

    addMessage(message, sender) {
        const chatMessages = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        messageDiv.textContent = message;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    addSuggestedQuestions(questions) {
        const chatMessages = document.getElementById('chat-messages');
        const container = document.createElement('div');
        container.className = 'suggested-questions';
        
        const title = document.createElement('div');
        title.className = 'suggested-title';
        title.textContent = 'Suggested follow-up questions:';
        container.appendChild(title);
        
        questions.forEach(q => {
            const btn = document.createElement('button');
            btn.className = 'suggested-question';
            btn.textContent = q;
            btn.onclick = () => {
                document.getElementById('user-input').value = q;
                this.sendMessage();
            };
            container.appendChild(btn);
        });
        
        chatMessages.appendChild(container);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    showTyping() {
        document.getElementById('typing-indicator').style.display = 'block';
    }

    hideTyping() {
        document.getElementById('typing-indicator').style.display = 'none';
    }
}

// Initialize the voice assistant when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new VoiceAssistant();
});
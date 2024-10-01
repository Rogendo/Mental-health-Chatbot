document.addEventListener('DOMContentLoaded', (event) => {
    const sendButton = document.getElementById('sendButton');
    const textInput = document.getElementById('textInput');
    const chatbox = document.getElementById('chatbox');

    function addMessage(message, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${isUser ? 'user' : 'bot'}`;
        
        const img = document.createElement('img');
        img.src = isUser ? '/static/img/person.png' : '/static/img/mhcicon.png';
        img.alt = isUser ? 'User' : 'psychAI';
        
        const p = document.createElement('p');
        p.textContent = message;
        
        messageDiv.appendChild(img);
        messageDiv.appendChild(p);
        chatbox.appendChild(messageDiv);
        
        // Scroll to the bottom of the chat
        chatbox.scrollTop = chatbox.scrollHeight;
    }

    function sendMessage() {
        const message = textInput.value.trim();
        if (message) {
            addMessage(message, true);
            textInput.value = '';
            
            fetch(`/get?msg=${encodeURIComponent(message)}`)
                .then(response => response.text())
                .then(data => {
                    addMessage(data);
                })
                .catch(error => {
                    console.error('Error:', error);
                    addMessage('Sorry, I encountered an error. Please try again.');
                });
        }
    }

    sendButton.addEventListener('click', sendMessage);
    textInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});

// Add this at the beginning of your script
let isWaiting = false;

// Modify your existing getBotResponse function
function getBotResponse() {
    if (isWaiting) return;
    
    var rawText = $("#textInput").val();
    var userHtml = '<p class="userText"><span>' + rawText + '</span></p>';
    $("#textInput").val("");
    $("#chatbox").append(userHtml);
    document.getElementById('userInput').scrollIntoView({block: 'start', behavior: 'smooth'});
    
    isWaiting = true;
    $("#chatbox").append('<p class="botText"><span id="loading">Thinking...</span></p>');
    document.getElementById('userInput').scrollIntoView({block: 'start', behavior: 'smooth'});

    $.get("/get", { msg: rawText }).done(function(data) {
        $("#loading").remove();
        var botHtml = '<p class="botText"><span>' + data + '</span></p>';
        $("#chatbox").append(botHtml);
        document.getElementById('userInput').scrollIntoView({block: 'start', behavior: 'smooth'});
        isWaiting = false;
    });
}

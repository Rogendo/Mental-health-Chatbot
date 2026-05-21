document.addEventListener("DOMContentLoaded", () => {
  const chatForm = document.getElementById("chatForm");
  const sendButton = document.getElementById("sendButton");
  const textInput = document.getElementById("textInput");
  const chatbox = document.getElementById("chatbox");
  const moodButtons = document.querySelectorAll(".mood-chip");
  const promptCards = document.querySelectorAll(".prompt-card");
  const breathButton = document.getElementById("breathButton");
  const breathPhase = document.getElementById("breathPhase");
  const breathCount = document.getElementById("breathCount");

  const history = [];
  let selectedMood = "steady";
  let isWaiting = false;
  let breathTimer = null;

  function addMessage(message, role = "assistant", meta = "") {
    const messageEl = document.createElement("article");
    messageEl.className = `chat-message ${role === "user" ? "user" : "bot"}`;

    const avatar = document.createElement("img");
    avatar.src = role === "user" ? "/static/img/person.png" : "/static/img/mhcicon.png";
    avatar.alt = "";

    const body = document.createElement("div");
    const text = document.createElement("p");
    text.textContent = message;
    body.appendChild(text);

    if (meta) {
      const small = document.createElement("small");
      small.textContent = meta;
      body.appendChild(small);
    }

    messageEl.appendChild(avatar);
    messageEl.appendChild(body);
    chatbox.appendChild(messageEl);
    chatbox.scrollTop = chatbox.scrollHeight;
    return messageEl;
  }

  function setWaiting(waiting) {
    isWaiting = waiting;
    sendButton.disabled = waiting;
    textInput.disabled = waiting;
    sendButton.textContent = waiting ? "..." : "Send";
  }

  async function sendMessage(message) {
    const cleanMessage = message.trim();
    if (!cleanMessage || isWaiting) return;

    addMessage(cleanMessage, "user");
    history.push({ role: "user", content: cleanMessage });
    textInput.value = "";
    setWaiting(true);

    const thinking = addMessage("Thinking with you...", "assistant");
    thinking.classList.add("thinking");

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: cleanMessage,
          mood: selectedMood,
          history: history.slice(-10),
        }),
      });

      if (!response.ok) {
        throw new Error("Request failed");
      }

      const data = await response.json();
      thinking.remove();
      addMessage(data.reply, "assistant", `Model: ${data.model}`);
      history.push({ role: "assistant", content: data.reply });
    } catch (error) {
      thinking.remove();
      addMessage("I had trouble responding just now. Try again in a moment, and if this is urgent please contact local emergency support.", "assistant");
    } finally {
      setWaiting(false);
      textInput.focus();
    }
  }

  moodButtons.forEach((button) => {
    button.addEventListener("click", () => {
      moodButtons.forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      selectedMood = button.dataset.mood;
    });
  });

  promptCards.forEach((card) => {
    card.addEventListener("click", () => {
      textInput.value = card.dataset.prompt;
      textInput.focus();
    });
  });

  chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage(textInput.value);
  });

  function runBreathingReset() {
    if (breathTimer) {
      clearInterval(breathTimer);
      breathTimer = null;
    }

    const pattern = [
      { phase: "Breathe in", seconds: 4 },
      { phase: "Hold", seconds: 2 },
      { phase: "Breathe out", seconds: 6 },
    ];
    let patternIndex = 0;
    let remaining = pattern[patternIndex].seconds;
    let cycles = 0;

    breathButton.disabled = true;
    breathButton.textContent = "Reset in progress";

    breathTimer = setInterval(() => {
      const current = pattern[patternIndex];
      breathPhase.textContent = current.phase;
      breathCount.textContent = remaining;
      remaining -= 1;

      if (remaining < 0) {
        patternIndex = (patternIndex + 1) % pattern.length;
        if (patternIndex === 0) cycles += 1;
        remaining = pattern[patternIndex].seconds;
      }

      if (cycles >= 5) {
        clearInterval(breathTimer);
        breathTimer = null;
        breathPhase.textContent = "Nice work";
        breathCount.textContent = "1";
        breathButton.disabled = false;
        breathButton.textContent = "Start 60 second reset";
      }
    }, 1000);
  }

  breathButton.addEventListener("click", runBreathingReset);
});

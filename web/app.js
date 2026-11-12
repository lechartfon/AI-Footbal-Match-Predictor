(function () {
  "use strict";

  var overlay = document.getElementById("overlay");
  var acceptBtn = document.getElementById("accept-btn");
  var statusDot = document.getElementById("status-dot");
  var statusText = document.getElementById("status-text");
  var reconnectBtn = document.getElementById("reconnect-btn");

  var ws = null;
  var term = null;

  function setStatus(text, online) {
    statusText.textContent = text;
    statusDot.className = "dot " + (online ? "on" : "off");
  }

  function createTerminal() {
    term = new Terminal({
      cursorBlink: true,
      fontFamily: "'Cascadia Mono', Consolas, 'Courier New', monospace",
      fontSize: 14,
      scrollback: 2000,
      theme: {
        background: "#0d1117",
        foreground: "#e6edf3",
        cursor: "#58a6ff",
      },
    });

    var fit = new FitAddon.FitAddon();
    term.loadAddon(fit);
    term.open(document.getElementById("terminal"));
    fit.fit();

    // Local echo: what the user types is drawn immediately. The server never
    // echoes the line back, it only answers with prompts/results.
    term.onData(function (data) {
      if (data === "\r") {
        term.write("\r\n");
      } else if (data === "\x7f") {
        term.write("\b \b");
      } else {
        term.write(data);
      }
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(data);
      }
    });

    window.addEventListener("resize", function () {
      fit.fit();
    });
  }

  function connect() {
    var proto = location.protocol === "https:" ? "wss" : "ws";
    setStatus("Conectando...", false);
    reconnectBtn.hidden = true;

    ws = new WebSocket(proto + "://" + location.host + "/ws/predict");

    ws.onopen = function () {
      setStatus("Conectado", true);
    };

    ws.onmessage = function (event) {
      term.write(event.data);
    };

    ws.onclose = function () {
      setStatus("Desconectado", false);
      reconnectBtn.hidden = false;
      term.write(
        "\r\n\x1b[93m[Se perdió la conexión. Pulsa Reconectar para continuar.]\x1b[0m\r\n"
      );
    };

    ws.onerror = function () {
      if (ws) {
        ws.close();
      }
    };
  }

  acceptBtn.addEventListener("click", function () {
    overlay.style.display = "none";
    createTerminal();
    connect();
  });

  reconnectBtn.addEventListener("click", function () {
    reconnectBtn.hidden = true;
    connect();
  });
})();
